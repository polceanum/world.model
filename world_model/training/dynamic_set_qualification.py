"""Compact governed qualification shell for specification 1.61.

This module deliberately contains orchestration, not another materializer or
security kernel.  The reusable :mod:`qualification_core` owns bounded
artifacts and exactly-once split permits; the dynamic-set evaluator owns the
public/private data boundary; this shell binds their evidence into one frozen
campaign and promotion record.

Development is restartable only at explicit durable boundaries.  Protected
evaluation is not even initialised until an independently supplied receipt has
re-identified the passed development report, checkpoint, and ledger bytes.
Once initialised, the protected ledger admits exactly
``selector -> confirmation -> final_test -> compositional_ood`` and becomes
terminal on the first failed gate.
"""

from __future__ import annotations

import io
import json
import math
import os
import stat
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from world_model.runtime.online_world_model import OnlineWorldModel
from world_model.training.convergence import CampaignInspection, ValidationCandidate
from world_model.training.dynamic_set_adapter import DynamicSetEpisodeObjectiveAdapter
from world_model.training.dynamic_set_bootstrap import (
    PlanningTaskScoreEvidence,
    pooled_paired_improvement_evidence,
)
from world_model.training.dynamic_set_campaign import (
    DEFAULT_CAMPAIGN,
    DisposableScreenMetrics,
    LimitHitReason,
    configure_second_attempt,
    decide_dynamic_set_campaign,
    disposable_screen_failures,
    project_minimum_update_feasibility,
)
from world_model.training.dynamic_set_config import OrpheusConfig, load_config
from world_model.training.dynamic_set_evaluation import (
    PHYSICAL_POPULATION_CLAIM_PURPOSE,
    DynamicSetAdditiveSnapshot,
    DynamicSetEvaluationResult,
    DynamicSetExampleScoreEvidence,
    DynamicSetPairedEvaluationResult,
    dynamic_set_truth_leakage_count,
    evaluate_authorized_dynamic_set_rows,
    evaluate_authorized_paired_dynamic_set_rows,
    evaluate_dynamic_set_materializations,
    evaluate_paired_dynamic_set_materializations,
    pooled_selection_score_evidence,
    protected_physical_population_claim_binding,
    screen_snapshot_from_result,
)
from world_model.training.dynamic_set_evaluation_cache import (
    DEVELOPMENT_EVALUATION_CACHE_DIRECTORY_NAME,
    DynamicSetDevelopmentEvaluationCache,
    DynamicSetDevelopmentEvaluationCacheBinding,
    DynamicSetDevelopmentEvaluationCacheEvidence,
)
from world_model.training.dynamic_set_gates import (
    HORIZON_POSITION_LIMITS_M,
    PhysicalCellMetrics,
    PlanningInvariantMetrics,
    PlanningSliceMetrics,
    PromotionIntegrityMetrics,
    ResourceMetrics,
    SupportedScalar,
    physical_cell_gate_failures,
    planning_gate_failures,
    promotion_gate_failures,
)
from world_model.training.dynamic_set_materializer import (
    DynamicSetPublicBoundaryEvidence,
    materialize_dynamic_set_episode,
)
from world_model.training.dynamic_set_objectives import (
    dynamic_set_objective,
    dynamic_set_objective_lower_bound,
)
from world_model.training.dynamic_set_planning import (
    PlanningTaskEvaluation,
    PlanningTaskOutcome,
    reduce_planning_task_outcomes,
)
from world_model.training.dynamic_set_planning_materializer import (
    PLANNING_POPULATION_CLAIM_PURPOSE,
    PlanningPairedPopulationEvaluationResult,
    PlanningPairProvenance,
    PlanningPopulationEvaluationResult,
    evaluate_authorized_paired_planning_rows,
    evaluate_paired_development_planning_materializations,
    planning_error,
    protected_planning_population_claim_binding,
    validate_paired_planning_population_evaluation_result,
    validate_planning_population_evaluation_result,
)
from world_model.training.dynamic_set_protocol import (
    FROZEN_PHYSICAL_MANIFEST_SHA256,
    FROZEN_PLANNING_MANIFEST_SHA256,
    MICROBATCH_SIZE,
    PHYSICAL_CELLS,
    PHYSICAL_SPLIT_SIZES,
    PLANNING_SPLIT_SIZES,
    SELECTION_SCORE_WEIGHTS,
    SIMULATOR_VERSION,
    SPECIFICATION_VERSION,
    PhysicalCell,
    PhysicalManifestRow,
    PlanningManifestRow,
    physical_manifest,
    planning_manifest,
    selection_score,
    validate_frozen_manifests,
)
from world_model.training.dynamic_set_resources import (
    FRESH_RESOURCE_WORKLOAD_SHA256,
    FreshWorkerResourceEvidence,
    measure_fresh_worker_resources,
)
from world_model.training.dynamic_set_selection import (
    AggregateScoreComparison,
    DynamicSetCheckpointScore,
    build_promotion_integrity_metrics_from_paired_evidence,
    select_dynamic_set_incumbent,
    selection_guardrail_failures,
)
from world_model.training.dynamic_set_trainer import (
    CHECKPOINT_SCHEMA,
    SCREEN_MANIFEST_SHA256,
    SCREEN_ROWS,
    DynamicSetTrainer,
    DynamicSetTrainingMicrobatch,
    dynamic_set_model_state_sha256,
    dynamic_set_perception_frame_index,
    dynamic_set_training_protocol_sha256,
)
from world_model.training.dynamic_set_training_cache import (
    TRAINING_CACHE_DIRECTORY_NAME,
    DynamicSetTrainingCache,
    DynamicSetTrainingCacheBinding,
    validate_training_cache_directory,
)
from world_model.training.qualification_core import (
    OrderedSplitLedger,
    QualificationArtifactDirectory,
    SplitPermit,
    canonical_json_bytes,
    canonical_sha256,
    capture_clean_published_git_state,
    sha256_bytes,
    validated_sha256,
)

QualificationStatus = Literal[
    "qualified_convergence",
    "objective_plateau",
    "failed_to_improve",
    "limit_hit",
]
ProtectedSplit = Literal["selector", "confirmation", "final_test", "compositional_ood"]

PROTECTED_SPLIT_ORDER: tuple[ProtectedSplit, ...] = (
    "selector",
    "confirmation",
    "final_test",
    "compositional_ood",
)
PAIRED_SCORE_SCHEMA = "pooled_additive_physical_0.85_independent_planning_0.15_v2"
SCREEN_OBJECTIVE_REGRET_SCHEMA = "signed_objective_lower_envelope_regret_v1"
ACCEPTED_LIKE_SCHEMA = "legacy_static_action_free_rgbd_v1"
ACCEPTED_LIKE_POPULATION_SHA256: Mapping[str, Mapping[int, str]] = {
    "development": {
        1: "87b42b84d0a4eb4685aad1eae1adcb5a83b013f78ee345e3d94ee515de2a663a",
        2: "519968d0426ad041ee3941770feacea380189accb688b89a2f07d31c52f39c74",
    },
    "selector": {
        1: "1144fa18019306f8ee190478e7b911321c06405adb255f51a28a24d867f949b2",
        2: "7682a0882dfaf1963a34d4a4c6ca0254b2cd8435a9cdecfa0f39ef143f7d29a9",
    },
    "confirmation": {
        1: "f07a7c21a1270d41cf0a738bf00987a00481e241aa27e386fd67f9d2802415d9",
        2: "0f12efc736cd5267723d4a1a2f2465fd6b7637f5a3f1d514863bab2133685058",
    },
    "final_test": {
        1: "544e8db88aedc5fb2230b3581c02d5d26449f66707e5fdb4eb51b55bd5a38a55",
        2: "7f2ca7fa210be45929bbeb23a8471a60efe6bfa85bee74f139d4765d75d1908e",
    },
    "compositional_ood": {
        1: "44a8e049112edaf938f1cdac98c7d1c12baa2967ca02a1e015c6550e4cea79d9",
        2: "de21ce87cb4bbc0a33fabaf7d8d15497914a211b5a6188053b0d3ba1b62bf1e6",
    },
}
TERMINAL_STATUSES = frozenset(
    {"qualified_convergence", "objective_plateau", "failed_to_improve", "limit_hit"}
)
FOUNDATION_ARTIFACT_NAMES = (
    "development_report.json",
    "development_model.pt",
    "development_attempt_1_access.json",
    "qualification_report.json",
    "qualification_attempt_1_access.json",
)
ARTIFACT_NAMES = (
    "protocol.json",
    "known_action_foundation.json",
    "development_ledger.json",
    "campaign_state.json",
    "screen_result.json",
    "screen_attempt_1_result.json",
    "development_error.json",
    "development_model.pt",
    "development_report.json",
    "development_review_receipt.json",
    "protected_ledger.json",
    "selector_result.json",
    "confirmation_result.json",
    "final_test_result.json",
    "compositional_ood_result.json",
    "qualification_report.json",
    *(
        f"development_scored_evidence_{completed_updates:06d}.json"
        for completed_updates in range(512, 32_768 + 1, 512)
    ),
    *(f"{split}_scored_evidence.json" for split in PROTECTED_SPLIT_ORDER),
)
MAXIMUM_ARTIFACT_BYTES = 32 * 1024 * 1024
SCORED_EVIDENCE_SCHEMA = "dynamic_set_scored_sufficient_statistics_v1"
EXECUTION_PROGRESS_SCHEMA = "dynamic_set_campaign_progress_v3"
EXECUTION_TIMING_EVIDENCE_SCHEMA = "dynamic_set_update_timing_v3"
VALIDATION_TIMING_EVIDENCE_SCHEMA = "dynamic_set_validation_timing_v1"
EXECUTION_SCHEDULE_SEED = 161_061
_ZERO_SHA256 = "0" * 64
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_CANONICAL_CONFIG_PATH = _REPOSITORY_ROOT / "configs" / "rgbd_dynamic_set_planning_cpu.yaml"
_REPOSITORY_EVALUATOR_SCHEMA = "dynamic_set_repository_evaluator_v1"
_FORMAL_EVALUATION_AUTHORITY = "repository_owned"
_TEST_EVALUATION_AUTHORITY = "test_only_injected"
_ARCHITECTURE_EXECUTION_KEYS = frozenset(
    {
        "architecture_attempt_index",
        "architecture_choice",
        "base_config_sha256",
        "resolved_config_sha256",
        "prior_attempt_cumulative_seconds",
        "architecture_attempt_sha256",
    }
)
_EXECUTION_RECEIPT_EVIDENCE_KEYS = frozenset(
    {
        "execution_progress_path",
        "execution_progress_sha256",
        "execution_progress_bytes",
        "execution_progress_record_sha256",
        "execution_protocol_sha256",
        "execution_config_sha256",
        "execution_source_sha256",
        "execution_schedule_seed",
        "execution_active_resume_name",
        "execution_completed_updates",
        "execution_checkpoint_sha256",
        "execution_model_state_sha256",
        *_ARCHITECTURE_EXECUTION_KEYS,
        "cumulative_training_seconds",
        "screen_wall_seconds",
        "completed_update_timing_count",
        "timing_evidence_sha256",
        "discarded_attempt_seconds",
        "execution_validation_timing_count",
        "execution_cumulative_validation_seconds",
        "execution_validation_timing_sha256",
        "campaign_envelope_limit_seconds",
        "training_mutation_limit_seconds",
        "reserved_audit_seconds",
        "projection_support_satisfied",
        "conservative_update_seconds",
        "conservative_validation_seconds",
        "projected_remaining_validation_seconds",
        "projected_minimum_training_seconds",
        "projected_minimum_envelope_seconds",
        "minimum_update_feasible",
        "rejected_update_count",
        "training_limit_reached",
        "execution_status",
        "limit_hit_reason",
        "execution_validated_candidate_count",
        "execution_validated_boundary_updates",
        "execution_validated_candidate_sha256",
        "execution_validated_checkpoint_sha256",
        "execution_validated_model_state_sha256",
    }
)


def _exact_dict(value: object, *, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{label} must be one exact JSON object")
    # Canonicalisation is the common JSON-native/nonfinite check.
    canonical_sha256(value)
    return dict(value)


def _record_with_digest(body: Mapping[str, Any]) -> dict[str, Any]:
    native = _exact_dict(dict(body), label="record")
    return {**native, "record_sha256": canonical_sha256(native)}


def _validate_record(value: object, *, schema: str) -> dict[str, Any]:
    record = _exact_dict(value, label=schema)
    supplied = validated_sha256(record.get("record_sha256"), label=f"{schema} record")
    body = {key: item for key, item in record.items() if key != "record_sha256"}
    if body.get("schema") != schema or canonical_sha256(body) != supplied:
        raise ValueError(f"{schema} record binding differs")
    return record


def _stable_regular_bytes(path: str | Path, *, maximum: int = MAXIMUM_ARTIFACT_BYTES) -> bytes:
    resolved = Path(path).absolute()
    before = os.lstat(resolved)
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size <= 0
        or before.st_size > maximum
    ):
        raise ValueError(f"{resolved} must be one bounded single-link regular file")
    flags = os.O_RDONLY | (getattr(os, "O_NOFOLLOW", 0))
    descriptor = os.open(resolved, flags)
    try:
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    final = os.lstat(resolved)
    identities = {
        (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
        for item in (before, opened, after, final)
    }
    if len(identities) != 1:
        raise OSError(f"{resolved} changed during stable read")
    contents = b"".join(chunks)
    if len(contents) != final.st_size or len(contents) > maximum:
        raise ValueError(f"{resolved} differs from its bounded file metadata")
    return contents


def _passed_formal_report(report: object, *, stage: str, splits: Sequence[str]) -> dict[str, Any]:
    value = _exact_dict(report, label=f"known-action {stage} report")
    expected = {
        "artifact_kind": "rgbd_known_action_qualification_report",
        "schema": "rgbd_known_action_qualification_report_v2",
        "stage": stage,
        "execution_mode": "formal",
        "passed": True,
        "outcome": "passed",
        "access_completed": True,
        "error": None,
        "opened_splits": list(splits),
        "stopped_after": splits[-1],
    }
    if any(value.get(name) != expected_value for name, expected_value in expected.items()):
        raise PermissionError(f"known-action {stage} report is not a completed formal pass")
    if type(value.get("formal_authorization")) is not dict:
        raise PermissionError(f"known-action {stage} report lacks formal authorization")
    results = value.get("results")
    if (
        type(results) is not list
        or [item.get("split") if type(item) is dict else None for item in results] != list(splits)
        or any(item.get("passed") is not True for item in results)
    ):
        raise PermissionError(f"known-action {stage} split evidence is not a complete pass")
    protocol = _exact_dict(value.get("protocol"), label="known-action protocol")
    supplied_protocol = validated_sha256(
        protocol.get("protocol_sha256"), label="known-action protocol"
    )
    unsigned_protocol = {key: item for key, item in protocol.items() if key != "protocol_sha256"}
    if (
        protocol.get("name") != "rgbd_known_action_planning_v2"
        or canonical_sha256(unsigned_protocol) != supplied_protocol
    ):
        raise PermissionError("known-action protocol binding differs")
    return value


def _passed_formal_ledger(
    ledger: object,
    *,
    stage: str,
    order: Sequence[str],
    report_sha256: str,
) -> dict[str, Any]:
    value = _exact_dict(ledger, label=f"known-action {stage} ledger")
    supplied = validated_sha256(value.get("record_sha256"), label=f"known-action {stage} ledger")
    unsigned = {key: item for key, item in value.items() if key != "record_sha256"}
    if canonical_sha256(unsigned) != supplied:
        raise ValueError(f"known-action {stage} ledger self-hash differs")
    publication = value.get("publication")
    if (
        value.get("schema") != "rgbd_known_action_access_ledger_v2"
        or value.get("artifact_kind") != "rgbd_known_action_exactly_once_access_ledger"
        or value.get("stage") != stage
        or value.get("execution_mode") != "formal"
        or value.get("status") != "complete_passed"
        or value.get("order") != list(order)
        or type(publication) is not dict
        or publication.get("state") != "normal_bound"
        or publication.get("report_sha256") != report_sha256
    ):
        raise PermissionError(f"known-action {stage} ledger is not a published formal pass")
    return value


def validate_known_action_foundation(directory: str | Path) -> dict[str, Any]:
    """Validate and bind the complete formal 1.60 development/qualification bundle."""

    artifacts = QualificationArtifactDirectory.attach(
        Path(directory).absolute(),
        allowed_names=FOUNDATION_ARTIFACT_NAMES,
        maximum_file_bytes=MAXIMUM_ARTIFACT_BYTES,
    )
    if artifacts.inventory() != frozenset(FOUNDATION_ARTIFACT_NAMES):
        raise PermissionError("specification-1.60 foundation bundle is incomplete")
    contents = {name: artifacts.read_bytes(name) for name in FOUNDATION_ARTIFACT_NAMES}
    hashes = {name: sha256_bytes(blob) for name, blob in contents.items()}
    development = _passed_formal_report(
        artifacts.read_json("development_report.json"),
        stage="development",
        splits=("development",),
    )
    qualification = _passed_formal_report(
        artifacts.read_json("qualification_report.json"),
        stage="qualification",
        splits=("selector", "confirmation", "final_test"),
    )
    if development.get("protocol") != qualification.get("protocol"):
        raise PermissionError("known-action reports bind different protocols")
    if development.get("source_provenance") != qualification.get("source_provenance"):
        raise PermissionError("known-action reports bind different source provenance")
    if development.get("resolved_config_sha256") != qualification.get("resolved_config_sha256"):
        raise PermissionError("known-action reports bind different configurations")
    validated_sha256(development.get("resolved_config_sha256"), label="known-action config")
    source = _exact_dict(development.get("source_provenance"), label="known-action source")
    remote = source.get("remote_publication")
    if (
        source.get("dirty") is not False
        or source.get("ahead") != 0
        or source.get("behind") != 0
        or source.get("commit") != source.get("upstream_commit")
        or type(remote) is not dict
        or remote.get("advertised_commit") != source.get("commit")
    ):
        raise PermissionError("known-action source was not a clean published freeze")
    checkpoint_record = development.get("checkpoint")
    if (
        type(checkpoint_record) is not dict
        or checkpoint_record.get("sha256") != hashes["development_model.pt"]
    ):
        raise ValueError("known-action development checkpoint byte binding differs")
    try:
        checkpoint_payload = torch.load(
            io.BytesIO(contents["development_model.pt"]),
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        raise ValueError(
            "known-action foundation checkpoint is not a safe tensor payload"
        ) from error
    if (
        type(checkpoint_payload) is not dict
        or checkpoint_payload.get("artifact_kind") != "rgbd_known_action_empty_state_checkpoint"
        or checkpoint_payload.get("execution_mode") != "formal"
        or checkpoint_payload.get("specification_version") != "1.60"
        or checkpoint_payload.get("simulator_version") != "sphere_world_v7"
        or checkpoint_payload.get("device") != "cpu"
        or checkpoint_payload.get("precision") != "float32"
        or checkpoint_payload.get("optimizer_updates") != 0
        or checkpoint_payload.get("model_state") != {}
        or checkpoint_payload.get("protocol_sha256") != development["protocol"]["protocol_sha256"]
        or checkpoint_payload.get("resolved_config_sha256") != development["resolved_config_sha256"]
        or checkpoint_payload.get("source_provenance") != source
        or checkpoint_payload.get("development_result") != development["results"][0]
        or checkpoint_payload.get("model_state_sha256")
        != checkpoint_record.get("model_state_sha256")
    ):
        raise PermissionError("known-action foundation checkpoint is not a formal 1.60 pass")
    expected_reviewed = {
        "checkpoint_sha256": hashes["development_model.pt"],
        "report_sha256": hashes["development_report.json"],
        "ledger_sha256": hashes["development_attempt_1_access.json"],
    }
    if qualification.get("reviewed_development") != expected_reviewed:
        raise PermissionError("known-action qualification lacks the reviewed development trio")
    development_ledger = _passed_formal_ledger(
        artifacts.read_json("development_attempt_1_access.json"),
        stage="development",
        order=("development",),
        report_sha256=hashes["development_report.json"],
    )
    qualification_ledger = _passed_formal_ledger(
        artifacts.read_json("qualification_attempt_1_access.json"),
        stage="qualification",
        order=("selector", "confirmation", "final_test"),
        report_sha256=hashes["qualification_report.json"],
    )
    qualification_bindings = qualification_ledger.get("bindings")
    if (
        type(qualification_bindings) is not dict
        or qualification_bindings.get("reviewed_development") != expected_reviewed
    ):
        raise PermissionError("known-action qualification ledger lacks the reviewed trio")
    body = {
        "schema": "dynamic_set_known_action_foundation_v1",
        "specification_version": "1.60",
        "qualified": True,
        "artifact_sha256": hashes,
        "protocol_sha256": development["protocol"]["protocol_sha256"],
        "resolved_config_sha256": development["resolved_config_sha256"],
        "source_provenance_sha256": canonical_sha256(source),
        "published_commit": source["commit"],
        "development_ledger_record_sha256": development_ledger["record_sha256"],
        "qualification_ledger_record_sha256": qualification_ledger["record_sha256"],
    }
    return _record_with_digest(body)


@dataclass(frozen=True, slots=True)
class DynamicSetSourceFreeze:
    source_sha256: str
    commit: str
    tree: str
    upstream_commit: str
    clean: bool
    published: bool

    def validate(self) -> DynamicSetSourceFreeze:
        supplied = validated_sha256(self.source_sha256, label="dynamic-set source")
        for name in ("commit", "tree", "upstream_commit"):
            value = getattr(self, name)
            if type(value) is not str or len(value) not in {40, 64}:
                raise ValueError(f"{name} must be one Git object identifier")
            try:
                int(value, 16)
            except ValueError as error:
                raise ValueError(f"{name} must be hexadecimal") from error
        if (
            self.clean is not True
            or self.published is not True
            or self.commit != self.upstream_commit
        ):
            raise PermissionError("specification 1.61 requires one clean published source freeze")
        expected = dynamic_set_source_sha256(
            commit=self.commit,
            tree=self.tree,
            upstream_commit=self.upstream_commit,
            clean=self.clean,
            published=self.published,
        )
        if supplied != expected:
            raise ValueError("dynamic-set source digest differs from its canonical derivation")
        return self


def dynamic_set_source_sha256(
    *,
    commit: str,
    tree: str,
    upstream_commit: str,
    clean: bool,
    published: bool,
) -> str:
    """Derive the sole source digest from the complete public freeze fields."""

    return canonical_sha256(
        {
            "schema": "dynamic_set_source_freeze_v1",
            "commit": commit,
            "tree": tree,
            "upstream_commit": upstream_commit,
            "clean": clean,
            "published": published,
        }
    )


def capture_dynamic_set_source_freeze(
    repository_root: str | Path = _REPOSITORY_ROOT,
) -> DynamicSetSourceFreeze:
    """Read-only capture of the exact clean HEAD/tree/upstream source freeze."""

    current = capture_clean_published_git_state(repository_root)
    body = {
        "commit": current.commit,
        "tree": current.tree,
        "upstream_commit": current.upstream_commit,
        "clean": current.clean,
        "published": current.published,
    }
    return DynamicSetSourceFreeze(
        source_sha256=dynamic_set_source_sha256(**body),
        **body,
    ).validate()


def authenticate_dynamic_set_source_freeze(
    source: DynamicSetSourceFreeze,
    repository_root: str | Path = _REPOSITORY_ROOT,
) -> DynamicSetSourceFreeze:
    """Require a supplied freeze to equal a fresh read-only Git capture."""

    expected = source.validate()
    current = capture_dynamic_set_source_freeze(repository_root)
    if current != expected:
        raise PermissionError("current source differs from the dynamic-set source freeze")
    return current


@dataclass(frozen=True, slots=True)
class _ExecutionProgressReceipt:
    path: Path
    contents: bytes
    record_sha256: str
    protocol_sha256: str
    config_sha256: str
    source_sha256: str
    active_checkpoint_path: Path
    active_checkpoint_contents: bytes
    active_resume_name: str
    checkpoint_sha256: str
    model_state_sha256: str
    completed_updates: int
    architecture_attempt_index: int
    architecture_choice: str
    base_config_sha256: str
    resolved_config_sha256: str
    prior_attempt_cumulative_seconds: float
    architecture_attempt_sha256: str
    cumulative_training_seconds: float
    screen_wall_seconds: float
    completed_update_timing_count: int
    timing_evidence_sha256: str
    discarded_attempt_seconds: float
    execution_validation_timing_count: int
    execution_cumulative_validation_seconds: float
    execution_validation_timing_sha256: str
    campaign_envelope_limit_seconds: float
    training_mutation_limit_seconds: float
    reserved_audit_seconds: float
    projection_support_satisfied: bool
    conservative_update_seconds: float | None
    conservative_validation_seconds: float | None
    projected_remaining_validation_seconds: float | None
    projected_minimum_training_seconds: float | None
    projected_minimum_envelope_seconds: float | None
    minimum_update_feasible: bool | None
    rejected_update_count: int
    training_limit_reached: bool
    execution_status: str
    limit_hit_reason: LimitHitReason
    validated_candidate_count: int
    validated_boundary_updates: int
    validated_candidate_sha256: str
    validated_checkpoint_sha256: str
    validated_model_state_sha256: str
    completed_update_seconds: tuple[float, ...]
    completed_validation_seconds: tuple[float, ...]

    @property
    def contents_sha256(self) -> str:
        return sha256_bytes(self.contents)

    def evidence(self) -> dict[str, Any]:
        return {
            "execution_progress_path": str(self.path),
            "execution_progress_sha256": self.contents_sha256,
            "execution_progress_bytes": len(self.contents),
            "execution_progress_record_sha256": self.record_sha256,
            "execution_protocol_sha256": self.protocol_sha256,
            "execution_config_sha256": self.config_sha256,
            "execution_source_sha256": self.source_sha256,
            "execution_schedule_seed": EXECUTION_SCHEDULE_SEED,
            "execution_active_resume_name": self.active_resume_name,
            "execution_completed_updates": self.completed_updates,
            "execution_checkpoint_sha256": self.checkpoint_sha256,
            "execution_model_state_sha256": self.model_state_sha256,
            "architecture_attempt_index": self.architecture_attempt_index,
            "architecture_choice": self.architecture_choice,
            "base_config_sha256": self.base_config_sha256,
            "resolved_config_sha256": self.resolved_config_sha256,
            "prior_attempt_cumulative_seconds": (self.prior_attempt_cumulative_seconds),
            "architecture_attempt_sha256": self.architecture_attempt_sha256,
            "cumulative_training_seconds": self.cumulative_training_seconds,
            "screen_wall_seconds": self.screen_wall_seconds,
            "completed_update_timing_count": self.completed_update_timing_count,
            "timing_evidence_sha256": self.timing_evidence_sha256,
            "discarded_attempt_seconds": self.discarded_attempt_seconds,
            "execution_validation_timing_count": self.execution_validation_timing_count,
            "execution_cumulative_validation_seconds": (
                self.execution_cumulative_validation_seconds
            ),
            "execution_validation_timing_sha256": (self.execution_validation_timing_sha256),
            "campaign_envelope_limit_seconds": self.campaign_envelope_limit_seconds,
            "training_mutation_limit_seconds": self.training_mutation_limit_seconds,
            "reserved_audit_seconds": self.reserved_audit_seconds,
            "projection_support_satisfied": self.projection_support_satisfied,
            "conservative_update_seconds": self.conservative_update_seconds,
            "conservative_validation_seconds": self.conservative_validation_seconds,
            "projected_remaining_validation_seconds": (self.projected_remaining_validation_seconds),
            "projected_minimum_training_seconds": self.projected_minimum_training_seconds,
            "projected_minimum_envelope_seconds": self.projected_minimum_envelope_seconds,
            "minimum_update_feasible": self.minimum_update_feasible,
            "rejected_update_count": self.rejected_update_count,
            "training_limit_reached": self.training_limit_reached,
            "execution_status": self.execution_status,
            "limit_hit_reason": self.limit_hit_reason,
            "execution_validated_candidate_count": self.validated_candidate_count,
            "execution_validated_boundary_updates": self.validated_boundary_updates,
            "execution_validated_candidate_sha256": self.validated_candidate_sha256,
            "execution_validated_checkpoint_sha256": self.validated_checkpoint_sha256,
            "execution_validated_model_state_sha256": self.validated_model_state_sha256,
        }


def _expected_execution_lineage(
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[int, int, str, str, str]:
    count = len(candidates)
    if not candidates:
        return (0, 0, _ZERO_SHA256, _ZERO_SHA256, _ZERO_SHA256)
    latest = candidates[-1]
    return (
        count,
        count * DEFAULT_CAMPAIGN.validation_interval_updates,
        latest["candidate_sha256"],
        latest["checkpoint_sha256"],
        latest["model_state_sha256"],
    )


def _validation_timing_fields(
    *,
    candidates: Sequence[Mapping[str, Any]],
    completed_updates: int,
    checkpoint_sha256: str,
    model_state_sha256: str,
    execution_progress_record_sha256: str,
    callback_binding_sha256: str,
    validation_wall_seconds: float,
) -> dict[str, Any]:
    """Build one append-only validation timing link bound to evaluated evidence."""

    if (
        type(validation_wall_seconds) is not float
        or not math.isfinite(validation_wall_seconds)
        or validation_wall_seconds < 0.0
    ):
        raise ValueError("validation wall timing must be finite and nonnegative")
    sequence = len(candidates)
    previous_sha256 = (
        _ZERO_SHA256
        if not candidates
        else validated_sha256(
            candidates[-1]["validation_timing_sha256"],
            label="previous validation timing",
        )
    )
    previous_durations = tuple(
        float(candidate["validation_wall_seconds"]) for candidate in candidates
    )
    cumulative = float(math.fsum((*previous_durations, validation_wall_seconds)))
    body = {
        "schema": VALIDATION_TIMING_EVIDENCE_SCHEMA,
        "sequence": sequence,
        "completed_updates": completed_updates,
        "checkpoint_sha256": checkpoint_sha256,
        "model_state_sha256": model_state_sha256,
        "execution_progress_record_sha256": execution_progress_record_sha256,
        "callback_binding_sha256": callback_binding_sha256,
        "validation_wall_seconds": validation_wall_seconds,
        "validation_timing_count": sequence + 1,
        "cumulative_validation_wall_seconds": cumulative,
        "previous_validation_timing_sha256": previous_sha256,
    }
    return {
        "validation_wall_seconds": validation_wall_seconds,
        "validation_timing_count": sequence + 1,
        "cumulative_validation_wall_seconds": cumulative,
        "previous_validation_timing_sha256": previous_sha256,
        "validation_timing_sha256": canonical_sha256(body),
    }


def _checkpoint_timing_progress_fields(
    payload: Mapping[str, Any],
    *,
    completed_updates: int,
) -> dict[str, Any]:
    """Authenticate checkpoint timing and rebuild every v3 progress field."""

    raw = payload.get("execution_timing")
    expected_keys = {
        "schema",
        *_ARCHITECTURE_EXECUTION_KEYS,
        "completed_update_seconds",
        "discarded_attempt_seconds",
        "screen_wall_seconds",
        "completed_validation_seconds",
        "validation_timing_sha256",
        "cumulative_training_seconds",
        "evidence_sha256",
    }
    if type(raw) is not dict or set(raw) != expected_keys:
        raise ValueError("execution checkpoint timing schema differs")
    body = {key: item for key, item in raw.items() if key != "evidence_sha256"}
    timing_sha256 = validated_sha256(
        raw["evidence_sha256"],
        label="execution checkpoint timing evidence",
    )
    samples = raw["completed_update_seconds"]
    discarded = raw["discarded_attempt_seconds"]
    screen_wall = raw["screen_wall_seconds"]
    validation_samples = raw["completed_validation_seconds"]
    validation_timing_sha256 = validated_sha256(
        raw["validation_timing_sha256"],
        label="execution validation timing",
    )
    architecture_attempt_index = raw["architecture_attempt_index"]
    architecture_choice = raw["architecture_choice"]
    base_config_sha256 = validated_sha256(
        raw["base_config_sha256"],
        label="execution base config",
    )
    resolved_config_sha256 = validated_sha256(
        raw["resolved_config_sha256"],
        label="execution resolved config",
    )
    prior_attempt_cumulative_seconds = raw["prior_attempt_cumulative_seconds"]
    architecture_attempt_sha256 = validated_sha256(
        raw["architecture_attempt_sha256"],
        label="execution architecture attempt",
    )
    if (
        raw["schema"] != EXECUTION_TIMING_EVIDENCE_SCHEMA
        or canonical_sha256(body) != timing_sha256
        or type(samples) is not list
        or type(validation_samples) is not list
        or len(samples) != completed_updates
        or len(samples) > DEFAULT_CAMPAIGN.maximum_updates
        or any(
            type(sample) is not float or not math.isfinite(sample) or sample < 0.0
            for sample in samples
        )
        or type(discarded) is not float
        or not math.isfinite(discarded)
        or discarded < 0.0
        or type(screen_wall) is not float
        or not math.isfinite(screen_wall)
        or screen_wall < 0.0
        or len(validation_samples)
        > completed_updates // DEFAULT_CAMPAIGN.validation_interval_updates
        or any(
            type(sample) is not float or not math.isfinite(sample) or sample < 0.0
            for sample in validation_samples
        )
        or bool(validation_samples) is (validation_timing_sha256 == _ZERO_SHA256)
        or type(architecture_attempt_index) is not int
        or architecture_attempt_index not in {1, 2}
        or architecture_choice not in {"base", "widen_perception", "widen_relation"}
        or (architecture_attempt_index == 1) is not (architecture_choice == "base")
        or type(prior_attempt_cumulative_seconds) is not float
        or not math.isfinite(prior_attempt_cumulative_seconds)
        or prior_attempt_cumulative_seconds < 0.0
        or (architecture_attempt_index == 1 and prior_attempt_cumulative_seconds != 0.0)
        or (architecture_attempt_index == 1 and resolved_config_sha256 != base_config_sha256)
        or (architecture_attempt_index == 2 and resolved_config_sha256 == base_config_sha256)
        or architecture_attempt_sha256 == _ZERO_SHA256
    ):
        raise ValueError("execution checkpoint timing binding differs")
    cumulative_validation = float(math.fsum(validation_samples))
    cumulative = float(
        math.fsum(
            (
                prior_attempt_cumulative_seconds,
                screen_wall,
                *samples,
                discarded,
                *validation_samples,
            )
        )
    )
    if (
        type(raw["cumulative_training_seconds"]) is not float
        or raw["cumulative_training_seconds"] != cumulative
    ):
        raise ValueError("execution checkpoint cumulative timing differs")
    projection = project_minimum_update_feasibility(
        completed_update_seconds=samples,
        discarded_attempt_seconds=discarded,
        prior_attempt_cumulative_seconds=prior_attempt_cumulative_seconds,
        screen_wall_seconds=screen_wall,
        completed_validation_seconds=validation_samples,
        config=DEFAULT_CAMPAIGN,
    )
    return {
        "architecture_attempt_index": architecture_attempt_index,
        "architecture_choice": architecture_choice,
        "base_config_sha256": base_config_sha256,
        "resolved_config_sha256": resolved_config_sha256,
        "prior_attempt_cumulative_seconds": prior_attempt_cumulative_seconds,
        "architecture_attempt_sha256": architecture_attempt_sha256,
        "cumulative_training_seconds": cumulative,
        "screen_wall_seconds": screen_wall,
        "completed_update_timing_count": len(samples),
        "timing_evidence_sha256": timing_sha256,
        "discarded_attempt_seconds": discarded,
        "execution_validation_timing_count": len(validation_samples),
        "execution_cumulative_validation_seconds": cumulative_validation,
        "execution_validation_timing_sha256": validation_timing_sha256,
        "campaign_envelope_limit_seconds": projection.envelope_limit_seconds,
        "training_mutation_limit_seconds": projection.mutation_limit_seconds,
        "reserved_audit_seconds": DEFAULT_CAMPAIGN.reserved_audit_seconds,
        "projection_support_satisfied": projection.support_satisfied,
        "conservative_update_seconds": projection.conservative_update_seconds,
        "conservative_validation_seconds": projection.conservative_validation_seconds,
        "projected_remaining_validation_seconds": (
            projection.projected_remaining_validation_seconds
        ),
        "projected_minimum_training_seconds": projection.projected_minimum_training_seconds,
        "projected_minimum_envelope_seconds": projection.projected_minimum_envelope_seconds,
        "minimum_update_feasible": projection.minimum_update_feasible,
        "training_limit_reached": projection.limit_hit,
        "execution_status": "limit_hit" if projection.limit_hit else "continue",
        "limit_hit_reason": projection.limit_hit_reason,
        "_completed_update_seconds": tuple(samples),
        "_completed_validation_seconds": tuple(validation_samples),
    }


def _validated_execution_progress_receipt(
    path: str | Path,
    *,
    protocol: Mapping[str, Any],
    lineage_candidates: Sequence[Mapping[str, Any]],
    qualification_root: Path,
    minimum_cumulative_training_seconds: float,
) -> _ExecutionProgressReceipt:
    progress_path = Path(path).absolute()
    if progress_path.name != "progress.json":
        raise ValueError("execution receipt must be the exact progress.json artifact")
    work_root = progress_path.parent.resolve(strict=True)
    progress_path = work_root / progress_path.name
    sealed_root = qualification_root.resolve(strict=True)
    if (
        work_root == sealed_root
        or work_root in sealed_root.parents
        or sealed_root in work_root.parents
    ):
        raise ValueError("execution receipt directory overlaps sealed qualification artifacts")
    contents = _stable_regular_bytes(progress_path)
    try:
        value = json.loads(contents.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("execution progress receipt is not strict JSON") from error
    expected_keys = {
        "schema",
        "protocol_sha256",
        "config_sha256",
        "source_sha256",
        "active_resume_name",
        "checkpoint_sha256",
        "model_state_sha256",
        "completed_updates",
        *_ARCHITECTURE_EXECUTION_KEYS,
        "cumulative_training_seconds",
        "screen_wall_seconds",
        "completed_update_timing_count",
        "timing_evidence_sha256",
        "discarded_attempt_seconds",
        "execution_validation_timing_count",
        "execution_cumulative_validation_seconds",
        "execution_validation_timing_sha256",
        "campaign_envelope_limit_seconds",
        "training_mutation_limit_seconds",
        "reserved_audit_seconds",
        "projection_support_satisfied",
        "conservative_update_seconds",
        "conservative_validation_seconds",
        "projected_remaining_validation_seconds",
        "projected_minimum_training_seconds",
        "projected_minimum_envelope_seconds",
        "minimum_update_feasible",
        "rejected_update_count",
        "training_limit_reached",
        "execution_status",
        "limit_hit_reason",
        "validated_candidate_count",
        "validated_boundary_updates",
        "validated_candidate_sha256",
        "validated_checkpoint_sha256",
        "validated_model_state_sha256",
        "record_sha256",
    }
    if type(value) is not dict or set(value) != expected_keys:
        raise ValueError("execution progress receipt schema differs")
    body = {key: item for key, item in value.items() if key != "record_sha256"}
    record_sha256 = validated_sha256(value["record_sha256"], label="execution progress record")
    if value["schema"] != EXECUTION_PROGRESS_SCHEMA or canonical_sha256(body) != record_sha256:
        raise ValueError("execution progress receipt digest differs")
    for name in (
        "protocol_sha256",
        "config_sha256",
        "source_sha256",
        "checkpoint_sha256",
        "model_state_sha256",
        "timing_evidence_sha256",
        "execution_validation_timing_sha256",
        "base_config_sha256",
        "resolved_config_sha256",
        "architecture_attempt_sha256",
        "validated_candidate_sha256",
        "validated_checkpoint_sha256",
        "validated_model_state_sha256",
    ):
        validated_sha256(value[name], label=f"execution progress {name}")
    if (
        value["protocol_sha256"] != protocol["protocol_sha256"]
        or value["config_sha256"] != protocol["config_sha256"]
        or value["source_sha256"] != canonical_sha256(protocol["source"])
    ):
        raise ValueError("execution progress protocol/config/source binding differs")
    completed = value["completed_updates"]
    rejected = value["rejected_update_count"]
    if (
        isinstance(completed, bool)
        or not isinstance(completed, int)
        or completed < 0
        or isinstance(rejected, bool)
        or not isinstance(rejected, int)
        or rejected < 0
    ):
        raise ValueError("execution progress counters differ")
    expected_lineage = _expected_execution_lineage(lineage_candidates)
    actual_lineage = (
        value["validated_candidate_count"],
        value["validated_boundary_updates"],
        value["validated_candidate_sha256"],
        value["validated_checkpoint_sha256"],
        value["validated_model_state_sha256"],
    )
    if actual_lineage != expected_lineage:
        raise ValueError("execution progress candidate ancestry differs")
    lower = expected_lineage[1]
    upper = min(
        lower + DEFAULT_CAMPAIGN.validation_interval_updates,
        DEFAULT_CAMPAIGN.maximum_updates,
    )
    if not lower <= completed <= upper:
        raise ValueError("execution progress cursor lies outside its candidate interval")
    active_name = value["active_resume_name"]
    if active_name not in {"resume_a.pt", "resume_b.pt"}:
        raise ValueError("execution progress names an invalid resume slot")
    active_path = work_root / active_name
    checkpoint_contents = _stable_regular_bytes(active_path)
    checkpoint_sha256 = sha256_bytes(checkpoint_contents)
    if checkpoint_sha256 != value["checkpoint_sha256"]:
        raise ValueError("execution progress checkpoint digest differs")
    try:
        payload = torch.load(io.BytesIO(checkpoint_contents), map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("execution progress checkpoint is not a safe tensor payload") from error
    trainer_state = payload.get("trainer_state") if type(payload) is dict else None
    next_sample = payload.get("next_sample_state") if type(payload) is dict else None
    model_state = payload.get("model_state") if type(payload) is dict else None
    if (
        type(payload) is not dict
        or payload.get("schema") != CHECKPOINT_SCHEMA
        or not isinstance(trainer_state, Mapping)
        or not isinstance(next_sample, Mapping)
        or not isinstance(model_state, Mapping)
        or trainer_state.get("completed_updates") != completed
        or trainer_state.get("rejected_update_count") != rejected
        or next_sample.get("absolute_update_index") != completed
        or next_sample.get("schedule_seed") != EXECUTION_SCHEDULE_SEED
        or payload.get("model_state_sha256") != value["model_state_sha256"]
        or dynamic_set_model_state_sha256(model_state) != value["model_state_sha256"]
    ):
        raise ValueError("execution progress checkpoint/cursor/fixed-seed binding differs")
    timing_fields = _checkpoint_timing_progress_fields(
        payload,
        completed_updates=completed,
    )
    if any(
        type(value[name]) is not type(expected) or value[name] != expected
        for name, expected in timing_fields.items()
        if not name.startswith("_")
    ):
        raise ValueError("execution progress timing/projection fields differ from checkpoint")
    expected_validation_seconds = tuple(
        float(candidate["validation_wall_seconds"]) for candidate in lineage_candidates
    )
    expected_validation_sha256 = (
        _ZERO_SHA256
        if not lineage_candidates
        else validated_sha256(
            lineage_candidates[-1]["validation_timing_sha256"],
            label="latest candidate validation timing",
        )
    )
    if (
        timing_fields["_completed_validation_seconds"] != expected_validation_seconds
        or timing_fields["execution_validation_timing_sha256"] != expected_validation_sha256
    ):
        raise ValueError("execution progress validation timing ancestry differs")
    elapsed = timing_fields["cumulative_training_seconds"]
    if elapsed < minimum_cumulative_training_seconds:
        raise ValueError("execution progress cumulative training time moved backwards")
    return _ExecutionProgressReceipt(
        path=progress_path,
        contents=contents,
        record_sha256=record_sha256,
        protocol_sha256=value["protocol_sha256"],
        config_sha256=value["config_sha256"],
        source_sha256=value["source_sha256"],
        active_checkpoint_path=active_path,
        active_checkpoint_contents=checkpoint_contents,
        active_resume_name=active_name,
        checkpoint_sha256=checkpoint_sha256,
        model_state_sha256=value["model_state_sha256"],
        completed_updates=completed,
        architecture_attempt_index=timing_fields["architecture_attempt_index"],
        architecture_choice=timing_fields["architecture_choice"],
        base_config_sha256=timing_fields["base_config_sha256"],
        resolved_config_sha256=timing_fields["resolved_config_sha256"],
        prior_attempt_cumulative_seconds=timing_fields["prior_attempt_cumulative_seconds"],
        architecture_attempt_sha256=timing_fields["architecture_attempt_sha256"],
        cumulative_training_seconds=elapsed,
        screen_wall_seconds=timing_fields["screen_wall_seconds"],
        completed_update_timing_count=timing_fields["completed_update_timing_count"],
        timing_evidence_sha256=timing_fields["timing_evidence_sha256"],
        discarded_attempt_seconds=timing_fields["discarded_attempt_seconds"],
        execution_validation_timing_count=timing_fields["execution_validation_timing_count"],
        execution_cumulative_validation_seconds=timing_fields[
            "execution_cumulative_validation_seconds"
        ],
        execution_validation_timing_sha256=timing_fields["execution_validation_timing_sha256"],
        campaign_envelope_limit_seconds=timing_fields["campaign_envelope_limit_seconds"],
        training_mutation_limit_seconds=timing_fields["training_mutation_limit_seconds"],
        reserved_audit_seconds=timing_fields["reserved_audit_seconds"],
        projection_support_satisfied=timing_fields["projection_support_satisfied"],
        conservative_update_seconds=timing_fields["conservative_update_seconds"],
        conservative_validation_seconds=timing_fields["conservative_validation_seconds"],
        projected_remaining_validation_seconds=timing_fields[
            "projected_remaining_validation_seconds"
        ],
        projected_minimum_training_seconds=timing_fields["projected_minimum_training_seconds"],
        projected_minimum_envelope_seconds=timing_fields["projected_minimum_envelope_seconds"],
        minimum_update_feasible=timing_fields["minimum_update_feasible"],
        rejected_update_count=rejected,
        training_limit_reached=timing_fields["training_limit_reached"],
        execution_status=timing_fields["execution_status"],
        limit_hit_reason=timing_fields["limit_hit_reason"],
        validated_candidate_count=actual_lineage[0],
        validated_boundary_updates=actual_lineage[1],
        validated_candidate_sha256=actual_lineage[2],
        validated_checkpoint_sha256=actual_lineage[3],
        validated_model_state_sha256=actual_lineage[4],
        completed_update_seconds=timing_fields["_completed_update_seconds"],
        completed_validation_seconds=timing_fields["_completed_validation_seconds"],
    )


def _assert_execution_receipt_stable(receipt: _ExecutionProgressReceipt) -> None:
    if (
        _stable_regular_bytes(receipt.path) != receipt.contents
        or _stable_regular_bytes(receipt.active_checkpoint_path)
        != receipt.active_checkpoint_contents
    ):
        raise OSError("execution progress/checkpoint changed during governed use")


def _receipt_evidence_matches(
    receipt: _ExecutionProgressReceipt,
    evidence: Mapping[str, Any],
) -> bool:
    expected = receipt.evidence()
    return set(evidence) >= set(expected) and all(
        evidence[name] == value for name, value in expected.items()
    )


def _validate_stored_execution_timing(
    evidence: Mapping[str, Any],
    *,
    minimum_cumulative_training_seconds: float,
) -> None:
    """Validate every timing/projection field retained in campaign state."""

    completed = evidence["execution_completed_updates"]
    architecture_attempt_index = evidence["architecture_attempt_index"]
    architecture_choice = evidence["architecture_choice"]
    base_config_sha256 = evidence["base_config_sha256"]
    resolved_config_sha256 = evidence["resolved_config_sha256"]
    prior_attempt_cumulative_seconds = evidence["prior_attempt_cumulative_seconds"]
    architecture_attempt_sha256 = evidence["architecture_attempt_sha256"]
    cumulative = evidence["cumulative_training_seconds"]
    screen_wall = evidence["screen_wall_seconds"]
    timing_count = evidence["completed_update_timing_count"]
    discarded = evidence["discarded_attempt_seconds"]
    validation_count = evidence["execution_validation_timing_count"]
    cumulative_validation = evidence["execution_cumulative_validation_seconds"]
    validation_timing_sha256 = evidence["execution_validation_timing_sha256"]
    envelope_limit = evidence["campaign_envelope_limit_seconds"]
    mutation_limit = evidence["training_mutation_limit_seconds"]
    reserve = evidence["reserved_audit_seconds"]
    support = evidence["projection_support_satisfied"]
    conservative = evidence["conservative_update_seconds"]
    conservative_validation = evidence["conservative_validation_seconds"]
    projected_remaining_validation = evidence["projected_remaining_validation_seconds"]
    projected = evidence["projected_minimum_training_seconds"]
    projected_envelope = evidence["projected_minimum_envelope_seconds"]
    feasible = evidence["minimum_update_feasible"]
    limit_reached = evidence["training_limit_reached"]
    execution_status = evidence["execution_status"]
    limit_reason = evidence["limit_hit_reason"]
    required_validations = (
        DEFAULT_CAMPAIGN.minimum_updates // DEFAULT_CAMPAIGN.validation_interval_updates
    )
    remaining_validations = max(0, required_validations - validation_count)
    expected_support = (
        type(completed) is int
        and type(validation_count) is int
        and completed >= DEFAULT_CAMPAIGN.minimum_timing_support_updates
        and (validation_count > 0 or remaining_validations == 0)
    )
    if (
        isinstance(completed, bool)
        or not isinstance(completed, int)
        or not 0 <= completed <= DEFAULT_CAMPAIGN.maximum_updates
        or type(architecture_attempt_index) is not int
        or architecture_attempt_index not in {1, 2}
        or architecture_choice not in {"base", "widen_perception", "widen_relation"}
        or (architecture_attempt_index == 1) is not (architecture_choice == "base")
        or type(prior_attempt_cumulative_seconds) is not float
        or not math.isfinite(prior_attempt_cumulative_seconds)
        or prior_attempt_cumulative_seconds < 0.0
        or (architecture_attempt_index == 1 and prior_attempt_cumulative_seconds != 0.0)
        or (architecture_attempt_index == 1 and resolved_config_sha256 != base_config_sha256)
        or (architecture_attempt_index == 2 and resolved_config_sha256 == base_config_sha256)
        or type(cumulative) is not float
        or not math.isfinite(cumulative)
        or cumulative < minimum_cumulative_training_seconds
        or type(screen_wall) is not float
        or not math.isfinite(screen_wall)
        or screen_wall < 0.0
        or type(timing_count) is not int
        or timing_count != completed
        or type(discarded) is not float
        or not math.isfinite(discarded)
        or discarded < 0.0
        or type(validation_count) is not int
        or not 0 <= validation_count <= completed // DEFAULT_CAMPAIGN.validation_interval_updates
        or type(cumulative_validation) is not float
        or not math.isfinite(cumulative_validation)
        or cumulative_validation < 0.0
        or cumulative
        < math.fsum(
            (
                prior_attempt_cumulative_seconds,
                screen_wall,
                discarded,
                cumulative_validation,
            )
        )
        or (validation_count == 0) is (validation_timing_sha256 != _ZERO_SHA256)
        or type(envelope_limit) is not float
        or envelope_limit != DEFAULT_CAMPAIGN.maximum_training_hours * 3600.0
        or type(mutation_limit) is not float
        or mutation_limit != DEFAULT_CAMPAIGN.training_mutation_seconds
        or type(reserve) is not float
        or reserve != DEFAULT_CAMPAIGN.reserved_audit_seconds
        or type(support) is not bool
        or support is not expected_support
        or (
            conservative is not None
            and (
                type(conservative) is not float
                or not math.isfinite(conservative)
                or conservative < 0.0
            )
        )
        or (
            conservative_validation is not None
            and (
                type(conservative_validation) is not float
                or not math.isfinite(conservative_validation)
                or conservative_validation < 0.0
            )
        )
        or (
            projected_remaining_validation is not None
            and (
                type(projected_remaining_validation) is not float
                or not math.isfinite(projected_remaining_validation)
                or projected_remaining_validation < 0.0
            )
        )
        or (
            projected is not None
            and (type(projected) is not float or not math.isfinite(projected) or projected < 0.0)
        )
        or (
            projected_envelope is not None
            and (
                type(projected_envelope) is not float
                or not math.isfinite(projected_envelope)
                or projected_envelope < 0.0
            )
        )
        or type(feasible) not in {type(None), bool}
        or (
            support
            and (
                conservative is None
                or conservative_validation is None
                or projected_remaining_validation is None
                or projected is None
                or projected_envelope is None
                or feasible is None
            )
        )
        or (
            not support
            and (
                conservative is not None
                or conservative_validation is not None
                or projected_remaining_validation is not None
                or projected is not None
                or projected_envelope is not None
                or feasible is not None
            )
        )
        or (
            projected_remaining_validation is not None
            and projected_remaining_validation
            != float(conservative_validation * remaining_validations)
        )
        or (
            projected is not None
            and (
                projected
                != float(
                    cumulative
                    + conservative * max(0, DEFAULT_CAMPAIGN.minimum_updates - completed)
                    + projected_remaining_validation
                )
                or projected_envelope != float(projected + DEFAULT_CAMPAIGN.reserved_audit_seconds)
            )
        )
        or (
            projected_envelope is not None
            and feasible is not (projected_envelope <= envelope_limit)
        )
        or type(limit_reached) is not bool
        or type(execution_status) is not str
        or execution_status != ("limit_hit" if limit_reached else "continue")
        or type(limit_reason) is not str
        or limit_reason
        not in {
            "none",
            "training_reserve_boundary",
            "minimum_update_projection_infeasible",
        }
        or limit_reached is not (limit_reason != "none")
        or (
            limit_reason == "training_reserve_boundary"
            and cumulative < DEFAULT_CAMPAIGN.training_mutation_seconds
        )
        or (
            limit_reason == "minimum_update_projection_infeasible"
            and (
                cumulative >= DEFAULT_CAMPAIGN.training_mutation_seconds
                or not support
                or feasible is not False
                or (completed >= DEFAULT_CAMPAIGN.minimum_updates and remaining_validations == 0)
            )
        )
        or (limit_reason == "none" and cumulative >= DEFAULT_CAMPAIGN.training_mutation_seconds)
    ):
        raise ValueError("stored execution timing/projection evidence differs")
    if cumulative >= DEFAULT_CAMPAIGN.training_mutation_seconds:
        expected_reason: LimitHitReason = "training_reserve_boundary"
    elif (
        support
        and feasible is False
        and (completed < DEFAULT_CAMPAIGN.minimum_updates or remaining_validations > 0)
    ):
        expected_reason = "minimum_update_projection_infeasible"
    else:
        expected_reason = "none"
    if limit_reason != expected_reason:
        raise ValueError("stored execution limit reason differs from its projection")
    validated_sha256(evidence["timing_evidence_sha256"], label="stored timing evidence")
    validated_sha256(validation_timing_sha256, label="stored validation timing evidence")
    validated_sha256(base_config_sha256, label="stored base config evidence")
    validated_sha256(resolved_config_sha256, label="stored resolved config evidence")
    if (
        validated_sha256(
            architecture_attempt_sha256,
            label="stored architecture attempt evidence",
        )
        == _ZERO_SHA256
    ):
        raise ValueError("stored architecture attempt digest cannot be zero")


def _validated_finish_execution_receipt(
    path: str | Path,
    *,
    protocol: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    qualification_root: Path,
) -> _ExecutionProgressReceipt:
    """Require terminal progress to have reconciled every sealed validation."""

    minimum_seconds = (
        0.0 if not candidates else float(candidates[-1]["cumulative_training_seconds"])
    )
    return _validated_execution_progress_receipt(
        path,
        protocol=protocol,
        lineage_candidates=candidates,
        qualification_root=qualification_root,
        minimum_cumulative_training_seconds=minimum_seconds,
    )


def _no_incumbent_limit_hit_reason(
    *,
    completed_updates: int,
    training_limit_reached: bool,
    limit_hit_reason: LimitHitReason,
) -> str:
    """Reconstruct the sole supported no-incumbent budget-stop reason."""

    if (
        isinstance(completed_updates, bool)
        or not isinstance(completed_updates, int)
        or completed_updates < 0
        or type(training_limit_reached) is not bool
        or limit_hit_reason
        not in {
            "none",
            "training_reserve_boundary",
            "minimum_update_projection_infeasible",
        }
    ):
        raise ValueError("no-incumbent limit evidence is malformed")
    if training_limit_reached:
        if limit_hit_reason == "minimum_update_projection_infeasible":
            return (
                "measured update timing cannot reach the supported minimum while "
                "preserving the frozen audit reserve"
            )
        if limit_hit_reason == "training_reserve_boundary":
            return (
                "the training mutation budget ended at the "
                f"{DEFAULT_CAMPAIGN.training_mutation_hours:g}-hour audit-reserve "
                "boundary before a supported incumbent"
            )
        raise ValueError("a reached training limit lacks its exact reason")
    if limit_hit_reason != "none":
        raise ValueError("a non-limited execution carries a limit reason")
    if completed_updates == DEFAULT_CAMPAIGN.maximum_updates:
        return (
            f"the hard {DEFAULT_CAMPAIGN.maximum_updates:,}-update cap was reached "
            "without a selection-safe supported incumbent"
        )
    raise ValueError("campaign cannot finish without a supported incumbent before a hard limit")


def _gate_schema_payload() -> dict[str, Any]:
    return {
        "physical_cell_count": len(PHYSICAL_CELLS),
        "physical_cells": [asdict(cell) for cell in PHYSICAL_CELLS],
        "planning_cardinalities": list(range(1, 7)),
        "planning_candidate_counts": [8, 32],
        "planning_distributions": ["in_distribution", "compositional_ood"],
        "selection_score_weights": dict(SELECTION_SCORE_WEIGHTS),
        "required_gate_functions": [
            "physical_cell_gate_failures",
            "planning_gate_failures",
            "promotion_gate_failures",
        ],
        "paired_score_schema": PAIRED_SCORE_SCHEMA,
        "screen_objective_regret": {
            "schema": SCREEN_OBJECTIVE_REGRET_SCHEMA,
            "optimization_objective": "exact_signed_dynamic_set_objective",
            "lower_bound": dynamic_set_objective_lower_bound(),
            "acceptance_metric": "objective_minus_lower_bound",
            "minimum_reduction_fraction": 0.80,
        },
    }


def _canonical_base_config_binding() -> tuple[str, str, OrpheusConfig]:
    """Return the raw-byte and resolved-payload bindings for attempt one."""

    contents = _stable_regular_bytes(_CANONICAL_CONFIG_PATH)
    config = load_config(_CANONICAL_CONFIG_PATH)
    config.validate()
    if (
        config.model.rgbd.observation_mode != "set"
        or config.model.rgbd.set_feature_dim != 32
        or config.model.dynamics.hidden_dim != 16
        or config.model.dynamics.relation_hidden_dim is not None
    ):
        raise ValueError("canonical dynamic-set profile differs from attempt-one widths")
    return sha256_bytes(contents), canonical_sha256(config.to_dict()), config


def _resolved_architecture_config(choice: str) -> OrpheusConfig:
    _raw_sha256, _payload_sha256, base = _canonical_base_config_binding()
    if choice == "base":
        config = base
    elif choice == "widen_perception":
        config = replace(
            base,
            model=replace(
                base.model,
                rgbd=replace(base.model.rgbd, set_feature_dim=64),
            ),
        )
    elif choice == "widen_relation":
        config = replace(
            base,
            model=replace(
                base.model,
                dynamics=replace(
                    base.model.dynamics,
                    relation_hidden_dim=64,
                ),
            ),
        )
    else:
        raise ValueError(f"unsupported architecture choice {choice!r}")
    config.validate()
    return config


def _repository_evaluator_binding(
    *,
    source_sha256: str,
    base_config_sha256: str,
    base_config_payload_sha256: str,
) -> dict[str, Any]:
    body = {
        "schema": _REPOSITORY_EVALUATOR_SCHEMA,
        "module": "world_model.training.dynamic_set_qualification",
        "screen_entrypoint": "evaluate_repository_disposable_screen",
        "split_entrypoint": "evaluate_repository_dynamic_set_split",
        "source_sha256": validated_sha256(source_sha256, label="evaluator source"),
        "base_config_sha256": validated_sha256(
            base_config_sha256,
            label="evaluator base config",
        ),
        "base_config_payload_sha256": validated_sha256(
            base_config_payload_sha256,
            label="evaluator base config payload",
        ),
    }
    return {**body, "evaluator_sha256": canonical_sha256(body)}


def _build_dynamic_set_protocol_binding(
    *,
    foundation: Mapping[str, Any],
    source: DynamicSetSourceFreeze,
    config_sha256: str,
    training_protocol_sha256: str | None,
    evaluation_authority: str,
) -> dict[str, Any]:
    if evaluation_authority not in {
        _FORMAL_EVALUATION_AUTHORITY,
        _TEST_EVALUATION_AUTHORITY,
    }:
        raise ValueError("dynamic-set evaluation authority is invalid")
    canonical_config_sha256, canonical_payload_sha256, _config = _canonical_base_config_binding()
    if evaluation_authority == _FORMAL_EVALUATION_AUTHORITY:
        if config_sha256 != canonical_config_sha256:
            raise ValueError(
                "formal qualification requires the exact checked-in dynamic-set CPU profile"
            )
        base_payload_sha256 = canonical_payload_sha256
        evaluator = _repository_evaluator_binding(
            source_sha256=source.source_sha256,
            base_config_sha256=canonical_config_sha256,
            base_config_payload_sha256=canonical_payload_sha256,
        )
    else:
        # Test-only protocols are deliberately and visibly non-formal.  Their
        # arbitrary config digest cannot authorize repository execution or a
        # protected formal split.
        base_payload_sha256 = config_sha256
        evaluator = None

    foundation_value = _validate_record(
        foundation,
        schema="dynamic_set_known_action_foundation_v1",
    )
    if foundation_value.get("qualified") is not True:
        raise PermissionError("known-action foundation did not qualify")
    source.validate()
    validated_sha256(config_sha256, label="dynamic-set config")
    expected_training = dynamic_set_training_protocol_sha256()
    training = (
        expected_training
        if training_protocol_sha256 is None
        else validated_sha256(
            training_protocol_sha256,
            label="dynamic-set training protocol",
        )
    )
    if training != expected_training:
        raise ValueError(
            "training protocol differs from the source-frozen specification 1.61 engine"
        )
    validate_frozen_manifests()
    gate_schema = _gate_schema_payload()
    body = {
        "schema": "dynamic_set_qualification_protocol_v2",
        "specification_version": SPECIFICATION_VERSION,
        "simulator_version": SIMULATOR_VERSION,
        "known_action_foundation_sha256": foundation_value["record_sha256"],
        "source": asdict(source),
        "config_sha256": config_sha256,
        "base_config_payload_sha256": base_payload_sha256,
        "evaluation_authority": evaluation_authority,
        "repository_evaluator": evaluator,
        "training_protocol_sha256": training,
        "physical_manifest_sha256": dict(FROZEN_PHYSICAL_MANIFEST_SHA256),
        "physical_split_sizes": dict(PHYSICAL_SPLIT_SIZES),
        "planning_manifest_sha256": dict(FROZEN_PLANNING_MANIFEST_SHA256),
        "planning_split_sizes": dict(PLANNING_SPLIT_SIZES),
        "screen_manifest_sha256": SCREEN_MANIFEST_SHA256,
        "screen_example_count": len(SCREEN_ROWS),
        "gate_schema": gate_schema,
        "gate_schema_sha256": canonical_sha256(gate_schema),
    }
    return {**body, "protocol_sha256": canonical_sha256(body)}


def build_dynamic_set_protocol_binding(
    *,
    foundation: Mapping[str, Any],
    source: DynamicSetSourceFreeze,
    config_sha256: str,
    training_protocol_sha256: str | None = None,
) -> dict[str, Any]:
    """Build the sole hash binding source, config, manifests, training, and gates."""

    return _build_dynamic_set_protocol_binding(
        foundation=foundation,
        source=source,
        config_sha256=config_sha256,
        training_protocol_sha256=training_protocol_sha256,
        evaluation_authority=_FORMAL_EVALUATION_AUTHORITY,
    )


def _build_test_dynamic_set_protocol_binding(
    *,
    foundation: Mapping[str, Any],
    source: DynamicSetSourceFreeze,
    config_sha256: str,
    training_protocol_sha256: str | None = None,
) -> dict[str, Any]:
    """Build an explicitly non-formal protocol for callback-driven unit harnesses."""

    return _build_dynamic_set_protocol_binding(
        foundation=foundation,
        source=source,
        config_sha256=config_sha256,
        training_protocol_sha256=training_protocol_sha256,
        evaluation_authority=_TEST_EVALUATION_AUTHORITY,
    )


def _architecture_attempt_record(
    *,
    protocol: Mapping[str, Any],
    index: int,
    choice: str,
    resolved_config_sha256: str,
    screen_result: Mapping[str, Any],
    start_cumulative_seconds: float,
    previous_attempt_sha256: str = _ZERO_SHA256,
) -> dict[str, Any]:
    diagnosis_sha256 = _screen_record_diagnosis_sha256(screen_result)
    end_cumulative_seconds = float(
        math.fsum((start_cumulative_seconds, screen_result["screen_wall_seconds"]))
    )
    body = {
        "schema": "dynamic_set_architecture_attempt_v1",
        "index": index,
        "choice": choice,
        "base_config_sha256": protocol["base_config_payload_sha256"],
        "resolved_config_sha256": validated_sha256(
            resolved_config_sha256,
            label="architecture resolved config",
        ),
        "source_sha256": canonical_sha256(protocol["source"]),
        "protocol_sha256": protocol["protocol_sha256"],
        "screen_result_sha256": screen_result["record_sha256"],
        "start_cumulative_seconds": start_cumulative_seconds,
        "end_cumulative_seconds": end_cumulative_seconds,
        "diagnosis_sha256": diagnosis_sha256,
        "feasibility_projection_sha256": _ZERO_SHA256,
        "status": "passed" if screen_result["passed"] else "failed_terminal",
        "previous_attempt_sha256": validated_sha256(
            previous_attempt_sha256,
            label="previous architecture attempt",
        ),
    }
    if body["status"] == "passed" and screen_result["passed"] is not True:
        raise ValueError("a passed architecture attempt requires a passed screen")
    return {**body, "attempt_sha256": canonical_sha256(body)}


def _validate_architecture_attempts(
    value: object,
    *,
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    if type(value) is not list:
        raise TypeError("architecture attempt ledger must be one exact list")
    if len(value) > 1:
        raise ValueError(
            "architecture attempt ledger lacks sealed second-attempt admission evidence"
        )
    expected_keys = {
        "schema",
        "index",
        "choice",
        "base_config_sha256",
        "resolved_config_sha256",
        "source_sha256",
        "protocol_sha256",
        "screen_result_sha256",
        "start_cumulative_seconds",
        "end_cumulative_seconds",
        "diagnosis_sha256",
        "feasibility_projection_sha256",
        "status",
        "previous_attempt_sha256",
        "attempt_sha256",
    }
    previous_sha256 = _ZERO_SHA256
    previous_end = 0.0
    resolved: list[dict[str, Any]] = []
    for offset, record in enumerate(value):
        if type(record) is not dict or set(record) != expected_keys:
            raise ValueError("architecture attempt record schema differs")
        body = {key: item for key, item in record.items() if key != "attempt_sha256"}
        for name in (
            "base_config_sha256",
            "resolved_config_sha256",
            "source_sha256",
            "protocol_sha256",
            "screen_result_sha256",
            "diagnosis_sha256",
            "feasibility_projection_sha256",
            "previous_attempt_sha256",
            "attempt_sha256",
        ):
            validated_sha256(record[name], label=f"architecture attempt {name}")
        if (
            record["schema"] != "dynamic_set_architecture_attempt_v1"
            or record["index"] != offset + 1
            or record["choice"] != "base"
            or record["resolved_config_sha256"] != protocol["base_config_payload_sha256"]
            or record["base_config_sha256"] != protocol["base_config_payload_sha256"]
            or record["source_sha256"] != canonical_sha256(protocol["source"])
            or record["protocol_sha256"] != protocol["protocol_sha256"]
            or record["feasibility_projection_sha256"] != _ZERO_SHA256
            or type(record["start_cumulative_seconds"]) is not float
            or type(record["end_cumulative_seconds"]) is not float
            or not math.isfinite(record["start_cumulative_seconds"])
            or not math.isfinite(record["end_cumulative_seconds"])
            or record["start_cumulative_seconds"] != previous_end
            or record["end_cumulative_seconds"] < record["start_cumulative_seconds"]
            or record["status"] not in {"passed", "failed_terminal"}
            or record["previous_attempt_sha256"] != previous_sha256
            or canonical_sha256(body) != record["attempt_sha256"]
        ):
            raise ValueError("architecture attempt transition binding differs")
        previous_sha256 = record["attempt_sha256"]
        previous_end = record["end_cumulative_seconds"]
        resolved.append(record)
    return tuple(resolved)


def _architecture_execution_fields_from_attempt(
    attempt: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "architecture_attempt_index": attempt["index"],
        "architecture_choice": attempt["choice"],
        "base_config_sha256": attempt["base_config_sha256"],
        "resolved_config_sha256": attempt["resolved_config_sha256"],
        "prior_attempt_cumulative_seconds": attempt["start_cumulative_seconds"],
        "architecture_attempt_sha256": attempt["attempt_sha256"],
    }


def _validate_protocol(value: object) -> dict[str, Any]:
    protocol = _exact_dict(value, label="dynamic-set protocol")
    expected_keys = {
        "schema",
        "specification_version",
        "simulator_version",
        "known_action_foundation_sha256",
        "source",
        "config_sha256",
        "base_config_payload_sha256",
        "evaluation_authority",
        "repository_evaluator",
        "training_protocol_sha256",
        "physical_manifest_sha256",
        "physical_split_sizes",
        "planning_manifest_sha256",
        "planning_split_sizes",
        "screen_manifest_sha256",
        "screen_example_count",
        "gate_schema",
        "gate_schema_sha256",
        "protocol_sha256",
    }
    if set(protocol) != expected_keys:
        raise ValueError("dynamic-set qualification protocol schema differs")
    supplied = validated_sha256(protocol.get("protocol_sha256"), label="dynamic-set protocol")
    body = {key: item for key, item in protocol.items() if key != "protocol_sha256"}
    if (
        body.get("schema") != "dynamic_set_qualification_protocol_v2"
        or body.get("specification_version") != SPECIFICATION_VERSION
        or body.get("simulator_version") != SIMULATOR_VERSION
        or body.get("physical_manifest_sha256") != dict(FROZEN_PHYSICAL_MANIFEST_SHA256)
        or body.get("physical_split_sizes") != dict(PHYSICAL_SPLIT_SIZES)
        or body.get("planning_manifest_sha256") != dict(FROZEN_PLANNING_MANIFEST_SHA256)
        or body.get("planning_split_sizes") != dict(PLANNING_SPLIT_SIZES)
        or body.get("screen_manifest_sha256") != SCREEN_MANIFEST_SHA256
        or body.get("screen_example_count") != 64
        or body.get("training_protocol_sha256") != dynamic_set_training_protocol_sha256()
        or body.get("gate_schema") != _gate_schema_payload()
        or body.get("gate_schema_sha256") != canonical_sha256(_gate_schema_payload())
        or canonical_sha256(body) != supplied
    ):
        raise ValueError("dynamic-set qualification protocol differs from its source freeze")
    DynamicSetSourceFreeze(**body["source"]).validate()
    validated_sha256(body.get("config_sha256"), label="dynamic-set config")
    validated_sha256(
        body.get("base_config_payload_sha256"),
        label="dynamic-set base config payload",
    )
    validated_sha256(body.get("known_action_foundation_sha256"), label="known-action foundation")
    authority = body.get("evaluation_authority")
    if authority == _FORMAL_EVALUATION_AUTHORITY:
        canonical_config_sha256, canonical_payload_sha256, _config = (
            _canonical_base_config_binding()
        )
        expected_evaluator = _repository_evaluator_binding(
            source_sha256=body["source"]["source_sha256"],
            base_config_sha256=canonical_config_sha256,
            base_config_payload_sha256=canonical_payload_sha256,
        )
        if (
            body["config_sha256"] != canonical_config_sha256
            or body["base_config_payload_sha256"] != canonical_payload_sha256
            or body["repository_evaluator"] != expected_evaluator
        ):
            raise ValueError("formal repository evaluator/config binding differs")
    elif authority == _TEST_EVALUATION_AUTHORITY:
        if (
            body["repository_evaluator"] is not None
            or body["base_config_payload_sha256"] != body["config_sha256"]
        ):
            raise ValueError("test-only evaluator binding differs")
    else:
        raise ValueError("dynamic-set evaluation authority differs")
    return protocol


@dataclass(frozen=True, slots=True)
class PhysicalCellEvaluation:
    cell: PhysicalCell
    metrics: PhysicalCellMetrics


@dataclass(frozen=True, slots=True)
class DynamicSetPromotionEvidence:
    """Raw paired evidence from which the orchestrator rebuilds promotion metrics."""

    candidate_physical: tuple[DynamicSetExampleScoreEvidence, ...]
    baseline_physical: tuple[DynamicSetExampleScoreEvidence, ...]
    candidate_planning: tuple[PlanningTaskScoreEvidence, ...]
    baseline_planning: tuple[PlanningTaskScoreEvidence, ...]
    candidate_aggregate_score: float
    baseline_aggregate_score: float
    critical_comparisons: tuple[AggregateScoreComparison, ...]
    accepted_like_comparisons: tuple[AggregateScoreComparison, ...]
    truth_leakage_count: int
    fabricated_target_count: int
    nonfinite_state_count: int
    rejected_optimizer_mutation_count: int
    minimum_complete_gradient_retention: float
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 161_061


@dataclass(frozen=True, slots=True)
class SplitExecutionContext:
    split: str
    permit: SplitPermit
    protocol_sha256: str
    checkpoint_sha256: str
    model_state_sha256: str
    completed_updates: int
    physical_rows: tuple[PhysicalManifestRow, ...]
    planning_rows: tuple[PlanningManifestRow, ...]
    physical_manifest_sha256: str
    planning_manifest_sha256: str
    fresh_resources_required: bool = False
    resource_config_sha256: str | None = None
    resource_source_sha256: str | None = None
    development_cache_namespace_sha256: str | None = None
    development_cache_evidence_sha256: str | None = None
    development_evaluation_cache: DynamicSetDevelopmentEvaluationCache | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    protected_ledger: OrderedSplitLedger | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    @property
    def callback_binding_sha256(self) -> str:
        return canonical_sha256(
            {
                "split": self.split,
                "permit": asdict(self.permit),
                "protocol_sha256": self.protocol_sha256,
                "checkpoint_sha256": self.checkpoint_sha256,
                "model_state_sha256": self.model_state_sha256,
                "completed_updates": self.completed_updates,
                "physical_manifest_sha256": self.physical_manifest_sha256,
                "physical_row_count": len(self.physical_rows),
                "planning_manifest_sha256": self.planning_manifest_sha256,
                "planning_task_count": len(self.planning_rows),
                "fresh_resources_required": self.fresh_resources_required,
                "resource_config_sha256": self.resource_config_sha256,
                "resource_source_sha256": self.resource_source_sha256,
                "development_cache_namespace_sha256": (self.development_cache_namespace_sha256),
                "development_cache_evidence_sha256": (self.development_cache_evidence_sha256),
            }
        )


@dataclass(frozen=True, slots=True)
class ScreenExecutionContext:
    permit: SplitPermit
    protocol_sha256: str
    rows: tuple[PhysicalManifestRow, ...]
    manifest_sha256: str

    @property
    def callback_binding_sha256(self) -> str:
        return canonical_sha256(
            {
                "split": "disposable_screen",
                "permit": asdict(self.permit),
                "protocol_sha256": self.protocol_sha256,
                "manifest_sha256": self.manifest_sha256,
                "example_count": len(self.rows),
            }
        )


@dataclass(frozen=True, slots=True)
class ScreenExecutionResult:
    metrics: DisposableScreenMetrics
    callback_binding_sha256: str
    architecture_diagnostics: ScreenArchitectureDiagnosticEvidence | None = None
    cache_evidence: ScreenCacheEvidence | None = None


@dataclass(frozen=True, slots=True)
class ScreenArchitectureDiagnosticEvidence:
    """Raw, digest-bound screen diagnostics used only for attempt routing.

    The formal repository evaluator will populate this only after its future
    truth-state diagnostic passes complete.  Until then formal attempt-two
    routing is deliberately unavailable rather than inferred from end-to-end
    proposal/collision summaries.
    """

    protocol_sha256: str
    source_sha256: str
    resolved_config_sha256: str
    screen_manifest_sha256: str
    row_count: int
    perception_failure_count: int
    oracle_state_dynamics_failure_count: int
    truth_state_contact_failure_count: int
    truth_state_noncontact_failure_count: int
    truth_state_contact_error: float
    truth_state_noncontact_error: float
    raw_evidence_sha256: str
    evidence_sha256: str

    @classmethod
    def create(
        cls,
        *,
        protocol_sha256: str,
        source_sha256: str,
        resolved_config_sha256: str,
        perception_failure_count: int,
        oracle_state_dynamics_failure_count: int,
        truth_state_contact_failure_count: int,
        truth_state_noncontact_failure_count: int,
        truth_state_contact_error: float,
        truth_state_noncontact_error: float,
        raw_evidence_sha256: str,
    ) -> ScreenArchitectureDiagnosticEvidence:
        body = {
            "protocol_sha256": protocol_sha256,
            "source_sha256": source_sha256,
            "resolved_config_sha256": resolved_config_sha256,
            "screen_manifest_sha256": SCREEN_MANIFEST_SHA256,
            "row_count": len(SCREEN_ROWS),
            "perception_failure_count": perception_failure_count,
            "oracle_state_dynamics_failure_count": (oracle_state_dynamics_failure_count),
            "truth_state_contact_failure_count": truth_state_contact_failure_count,
            "truth_state_noncontact_failure_count": (truth_state_noncontact_failure_count),
            "truth_state_contact_error": float(truth_state_contact_error),
            "truth_state_noncontact_error": float(truth_state_noncontact_error),
            "raw_evidence_sha256": raw_evidence_sha256,
        }
        result = cls(**body, evidence_sha256=canonical_sha256(body))
        return result.validate(
            protocol_sha256=protocol_sha256,
            source_sha256=source_sha256,
            resolved_config_sha256=resolved_config_sha256,
        )

    @property
    def perception_gates_passed(self) -> bool:
        return self.perception_failure_count == 0

    @property
    def oracle_state_dynamics_passed(self) -> bool:
        return self.oracle_state_dynamics_failure_count == 0

    @property
    def truth_state_contact_rollout_owns_error(self) -> bool:
        total = self.truth_state_contact_error + self.truth_state_noncontact_error
        return bool(
            self.perception_gates_passed
            and self.truth_state_contact_failure_count > 0
            and self.truth_state_noncontact_failure_count == 0
            and total > 0.0
            and self.truth_state_contact_error / total >= 0.80
        )

    def validate(
        self,
        *,
        protocol_sha256: str,
        source_sha256: str,
        resolved_config_sha256: str,
    ) -> ScreenArchitectureDiagnosticEvidence:
        if type(self) is not ScreenArchitectureDiagnosticEvidence:
            raise TypeError("screen architecture diagnostics must use the exact evidence type")
        for name in (
            "protocol_sha256",
            "source_sha256",
            "resolved_config_sha256",
            "screen_manifest_sha256",
            "raw_evidence_sha256",
            "evidence_sha256",
        ):
            validated_sha256(getattr(self, name), label=f"screen diagnostic {name}")
        for name in (
            "row_count",
            "perception_failure_count",
            "oracle_state_dynamics_failure_count",
            "truth_state_contact_failure_count",
            "truth_state_noncontact_failure_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"screen diagnostic {name} must be nonnegative")
        for name in ("truth_state_contact_error", "truth_state_noncontact_error"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"screen diagnostic {name} must be finite and nonnegative")
        unsigned = {
            name: value for name, value in asdict(self).items() if name != "evidence_sha256"
        }
        contact_row_count = sum(PHYSICAL_CELLS[row.cell_index].contact for row in SCREEN_ROWS)
        noncontact_row_count = len(SCREEN_ROWS) - contact_row_count
        if (
            self.protocol_sha256 != protocol_sha256
            or self.source_sha256 != source_sha256
            or self.resolved_config_sha256 != resolved_config_sha256
            or self.screen_manifest_sha256 != SCREEN_MANIFEST_SHA256
            or self.row_count != len(SCREEN_ROWS)
            or self.perception_failure_count > len(SCREEN_ROWS)
            or self.truth_state_contact_failure_count > contact_row_count
            or self.truth_state_noncontact_failure_count > noncontact_row_count
            or self.oracle_state_dynamics_failure_count
            != self.truth_state_contact_failure_count + self.truth_state_noncontact_failure_count
            or self.raw_evidence_sha256 == _ZERO_SHA256
            or canonical_sha256(unsigned) != self.evidence_sha256
        ):
            raise ValueError("screen architecture diagnostic binding differs")
        return self


@dataclass(frozen=True, slots=True)
class ScreenCacheEvidence:
    namespace_sha256: str
    cold_materializations: int
    warm_hits: int
    bytes_written: int
    cold_materialization_seconds: float
    warm_materialization_seconds: float
    evidence_sha256: str

    @classmethod
    def create(cls, cache: DynamicSetTrainingCache) -> ScreenCacheEvidence:
        body = {
            "namespace_sha256": cache.binding.namespace_sha256,
            "cold_materializations": cache.cold_materializations,
            "warm_hits": cache.warm_hits,
            "bytes_written": cache.bytes_written,
            "cold_materialization_seconds": float(cache.cold_materialization_seconds),
            "warm_materialization_seconds": float(cache.warm_materialization_seconds),
        }
        return cls(**body, evidence_sha256=canonical_sha256(body)).validate()

    def validate(self) -> ScreenCacheEvidence:
        validated_sha256(self.namespace_sha256, label="screen cache namespace")
        validated_sha256(self.evidence_sha256, label="screen cache evidence")
        for name in ("cold_materializations", "warm_hits", "bytes_written"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"screen cache {name} must be nonnegative")
        for name in ("cold_materialization_seconds", "warm_materialization_seconds"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"screen cache {name} must be finite and nonnegative")
        body = {name: value for name, value in asdict(self).items() if name != "evidence_sha256"}
        if canonical_sha256(body) != self.evidence_sha256:
            raise ValueError("screen cache evidence digest differs")
        return self


_SCREEN_RECORD_KEYS = frozenset(
    {
        "schema",
        "protocol_sha256",
        "permit",
        "callback_binding_sha256",
        "manifest_sha256",
        "screen_wall_seconds",
        "architecture_diagnostics",
        "cache_evidence",
        "metrics",
        "failures",
        "passed",
        "record_sha256",
    }
)
_SCREEN_DIAGNOSTIC_DERIVED_KEYS = frozenset(
    {
        "perception_gates_passed",
        "oracle_state_dynamics_passed",
        "truth_state_contact_rollout_owns_error",
    }
)


def _validated_screen_record(
    value: object,
    *,
    protocol: Mapping[str, Any],
    resolved_config_sha256: str,
    expected_passed: bool | None = None,
) -> tuple[
    dict[str, Any],
    DisposableScreenMetrics | None,
    ScreenArchitectureDiagnosticEvidence | None,
    ScreenCacheEvidence | None,
]:
    """Rebuild one screen record instead of trusting its sealed booleans."""

    record = _validate_record(value, schema="dynamic_set_screen_result_v1")
    if set(record) != _SCREEN_RECORD_KEYS:
        raise ValueError("screen result schema differs")
    protocol_sha256 = validated_sha256(protocol["protocol_sha256"], label="screen protocol")
    resolved_sha256 = validated_sha256(
        resolved_config_sha256,
        label="screen resolved configuration",
    )
    permit_payload = _exact_dict(record["permit"], label="screen permit")
    if set(permit_payload) != {"split", "index", "nonce", "protocol_sha256"}:
        raise ValueError("screen permit schema differs")
    permit = SplitPermit(**permit_payload)
    validated_sha256(permit.nonce, label="screen permit nonce")
    if (
        permit.split != "disposable_screen"
        or type(permit.index) is not int
        or permit.index != 0
        or permit.protocol_sha256 != protocol_sha256
    ):
        raise ValueError("screen permit binding differs")
    context = ScreenExecutionContext(
        permit=permit,
        protocol_sha256=protocol_sha256,
        rows=SCREEN_ROWS,
        manifest_sha256=SCREEN_MANIFEST_SHA256,
    )
    wall_seconds = record["screen_wall_seconds"]
    if (
        record["protocol_sha256"] != protocol_sha256
        or record["manifest_sha256"] != SCREEN_MANIFEST_SHA256
        or record["callback_binding_sha256"] != context.callback_binding_sha256
        or type(wall_seconds) is not float
        or not math.isfinite(wall_seconds)
        or wall_seconds < 0.0
        or type(record["passed"]) is not bool
        or (expected_passed is not None and record["passed"] is not expected_passed)
    ):
        raise ValueError("screen result binding differs")

    metrics_payload = record["metrics"]
    diagnostic_payload = record["architecture_diagnostics"]
    cache_payload = record["cache_evidence"]
    failures_payload = record["failures"]
    if type(failures_payload) is not list or any(
        type(item) is not str or not item for item in failures_payload
    ):
        raise ValueError("screen failure evidence differs")

    if metrics_payload is None:
        if (
            diagnostic_payload is not None
            or cache_payload is not None
            or record["passed"] is not False
            or len(failures_payload) != 1
            or not failures_payload[0].startswith("callback_error:")
        ):
            raise ValueError("screen callback-error evidence differs")
        return record, None, None, None

    raw_metrics = _exact_dict(metrics_payload, label="screen metrics")
    if set(raw_metrics) != {item.name for item in fields(DisposableScreenMetrics)}:
        raise ValueError("screen metrics schema differs")
    try:
        metrics = DisposableScreenMetrics(**raw_metrics)
        expected_failures = list(disposable_screen_failures(metrics))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("screen metrics are invalid") from error

    diagnostics: ScreenArchitectureDiagnosticEvidence | None = None
    if diagnostic_payload is not None:
        raw_diagnostics = _exact_dict(
            diagnostic_payload,
            label="screen architecture diagnostics",
        )
        expected_diagnostic_keys = {
            item.name for item in fields(ScreenArchitectureDiagnosticEvidence)
        } | _SCREEN_DIAGNOSTIC_DERIVED_KEYS
        if set(raw_diagnostics) != expected_diagnostic_keys:
            raise ValueError("screen architecture diagnostic schema differs")
        diagnostics = ScreenArchitectureDiagnosticEvidence(
            **{
                name: item
                for name, item in raw_diagnostics.items()
                if name not in _SCREEN_DIAGNOSTIC_DERIVED_KEYS
            }
        ).validate(
            protocol_sha256=protocol_sha256,
            source_sha256=protocol["source"]["source_sha256"],
            resolved_config_sha256=resolved_sha256,
        )
        if any(
            raw_diagnostics[name] is not getattr(diagnostics, name)
            for name in _SCREEN_DIAGNOSTIC_DERIVED_KEYS
        ):
            raise ValueError("screen architecture derived diagnostics differ")

    cache: ScreenCacheEvidence | None = None
    if cache_payload is not None:
        raw_cache = _exact_dict(cache_payload, label="screen cache evidence")
        if set(raw_cache) != {item.name for item in fields(ScreenCacheEvidence)}:
            raise ValueError("screen cache evidence schema differs")
        cache = ScreenCacheEvidence(**raw_cache).validate()
        expected_cache_binding = DynamicSetTrainingCacheBinding(
            source_sha256=canonical_sha256(protocol["source"]),
            config_sha256=protocol["config_sha256"],
            training_manifest_sha256=FROZEN_PHYSICAL_MANIFEST_SHA256["training"],
        )
        if cache.namespace_sha256 != expected_cache_binding.namespace_sha256:
            raise ValueError("screen cache namespace differs from the protocol")

    if protocol["evaluation_authority"] == _FORMAL_EVALUATION_AUTHORITY:
        if diagnostics is None:
            expected_failures.append("architecture_diagnostics:unavailable")
        if cache is None:
            expected_failures.append("screen_cache_evidence:unavailable")
    if failures_payload != expected_failures or record["passed"] is not (not expected_failures):
        raise ValueError("screen pass/failure result differs from recomputation")
    return record, metrics, diagnostics, cache


def _screen_record_diagnosis_sha256(screen_record: Mapping[str, Any]) -> str:
    diagnostics = screen_record["architecture_diagnostics"]
    if diagnostics is not None:
        return validated_sha256(
            diagnostics["evidence_sha256"],
            label="screen architecture diagnosis",
        )
    return canonical_sha256(
        {
            "schema": "dynamic_set_architecture_diagnosis_unavailable_v1",
            "reason": "truth_state_diagnostic_passes_unavailable",
            "screen_result_sha256": screen_record["record_sha256"],
        }
    )


@dataclass(frozen=True, slots=True)
class DynamicSetSplitEvidence:
    split: str
    protocol_sha256: str
    checkpoint_sha256: str
    model_state_sha256: str
    completed_updates: int
    physical_manifest_sha256: str
    physical_row_count: int
    planning_manifest_sha256: str
    planning_task_count: int
    physical_cells: tuple[PhysicalCellEvaluation, ...]
    planning_slices: tuple[PlanningSliceMetrics, ...]
    planning_invariants: PlanningInvariantMetrics
    promotion_evidence: DynamicSetPromotionEvidence
    promotion_integrity: PromotionIntegrityMetrics
    resources: ResourceMetrics
    score_components: Mapping[str, float]
    training_support_passed: bool
    callback_binding_sha256: str
    planning_pair_provenance: tuple[PlanningPairProvenance, ...] = ()
    fresh_resource_evidence: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class DynamicSetIntegrityEvidence:
    """Run-level facts not reconstructible from detached evaluation outputs."""

    truth_leakage_count: int
    fabricated_target_count: int
    nonfinite_state_count: int
    rejected_optimizer_mutation_count: int
    minimum_complete_gradient_retention: float
    training_support_passed: bool
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 161_061


@dataclass(frozen=True, slots=True)
class _RepositoryTrainingIntegrity:
    """Checkpoint-only integrity facts available before scored evaluation."""

    rejected_optimizer_mutation_count: int
    minimum_complete_gradient_retention: float
    training_support_passed: bool

    def validate(self) -> _RepositoryTrainingIntegrity:
        if (
            isinstance(self.rejected_optimizer_mutation_count, bool)
            or not isinstance(self.rejected_optimizer_mutation_count, int)
            or self.rejected_optimizer_mutation_count < 0
        ):
            raise ValueError("rejected optimizer mutation count must be nonnegative")
        if (
            isinstance(self.minimum_complete_gradient_retention, bool)
            or not isinstance(self.minimum_complete_gradient_retention, (int, float))
            or not math.isfinite(float(self.minimum_complete_gradient_retention))
            or not 0.0 <= float(self.minimum_complete_gradient_retention) <= 1.0
        ):
            raise ValueError("minimum complete gradient retention must lie in [0,1]")
        if type(self.training_support_passed) is not bool:
            raise TypeError("training support must be one exact boolean")
        return self


def _physical_result_records(
    result: DynamicSetEvaluationResult,
    context: SplitExecutionContext,
) -> tuple[DynamicSetExampleScoreEvidence, ...]:
    if type(result) is not DynamicSetEvaluationResult:
        raise TypeError("physical evaluator must return exact DynamicSetEvaluationResult")
    records = tuple(result.per_example_score_evidence)
    if result.episode_count != len(context.physical_rows) or len(records) != len(
        context.physical_rows
    ):
        raise ValueError("physical evaluation does not cover the exact split population")
    for row, record in zip(context.physical_rows, records, strict=True):
        record.validate()
        if (
            record.split != row.split
            or record.ordinal != row.ordinal
            or record.seed != row.seed
            or record.cell_index != row.cell_index
            or record.row_sha256 != canonical_sha256(asdict(row))
            or record.additive.episode_count != 1
        ):
            raise ValueError("physical per-example evidence differs from frozen row order")
    if set(result.evidence_by_cell) != set(PHYSICAL_CELLS) or set(result.by_cell) != set(
        PHYSICAL_CELLS
    ):
        raise ValueError("physical evaluation lacks one or more of the 22 cells")
    for cell in PHYSICAL_CELLS:
        if result.evidence_by_cell[cell].physical_metrics(cell) != result.by_cell[cell]:
            raise ValueError("physical cell metrics differ from their additive evidence")
    rebuilt_score = pooled_selection_score_evidence(records)
    rebuilt_components = dict(rebuilt_score.components)
    result_components = dict(result.score.components)
    if set(rebuilt_components) != set(result_components) or any(
        not math.isclose(
            rebuilt_components[name],
            result_components[name],
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        for name in rebuilt_components
    ):
        raise ValueError("physical score components differ from per-example evidence")
    binding = result.population_binding
    if binding is None or binding.split != context.split or binding.row_count != len(records):
        raise ValueError("physical evaluator population binding differs")
    if context.split in PROTECTED_SPLIT_ORDER and (
        binding.protocol_sha256 != context.protocol_sha256
        or binding.manifest_sha256 != context.physical_manifest_sha256
        or binding.permit_index != context.permit.index
    ):
        raise PermissionError("protected physical evaluator permit binding differs")
    return records


def _planning_result_outcomes(
    result: PlanningPopulationEvaluationResult,
    context: SplitExecutionContext,
) -> tuple[Any, ...]:
    if type(result) is not PlanningPopulationEvaluationResult:
        raise TypeError("planning evaluator must return exact PlanningPopulationEvaluationResult")
    validate_planning_population_evaluation_result(result)
    outcomes = tuple(result.outcomes)
    if len(outcomes) != len(context.planning_rows):
        raise ValueError("planning evaluation does not cover the exact task population")
    if tuple(outcome.row for outcome in outcomes) != context.planning_rows:
        raise ValueError("planning outcomes differ from frozen task row order")
    rebuilt = reduce_planning_task_outcomes(outcomes)
    if rebuilt != result.reduction:
        raise ValueError("planning reduction differs from its per-task outcomes")
    rebuilt_error = planning_error(outcomes)
    if not math.isclose(rebuilt_error, result.planning_error, rel_tol=0.0, abs_tol=1.0e-15):
        raise ValueError("planning score differs from its per-task outcomes")
    binding = result.population_binding
    if binding.split != context.split or binding.row_count != len(outcomes):
        raise ValueError("planning evaluator population binding differs")
    if context.split in PROTECTED_SPLIT_ORDER and (
        binding.protocol_sha256 != context.protocol_sha256
        or binding.manifest_sha256 != context.planning_manifest_sha256
        or binding.permit_index != context.permit.index
    ):
        raise PermissionError("protected planning evaluator permit binding differs")
    return outcomes


def _paired_subset_scores(
    baseline_components: Mapping[str, float],
    candidate_components: Mapping[str, float],
) -> tuple[float, float]:
    allowed = set(SELECTION_SCORE_WEIGHTS) - {"planning_error"}
    names = set(baseline_components) | set(candidate_components)
    if not names or not names <= allowed:
        raise ValueError("physical paired score has an unsupported component schema")
    denominator = sum(SELECTION_SCORE_WEIGHTS[name] for name in names)

    def score(components: Mapping[str, float]) -> float:
        # A component missing on only one side represents support collapse and
        # receives the unit normalized-error penalty. Components inapplicable
        # to both sides are absent from ``names`` and therefore do not dilute
        # the row/cell comparison.
        return (
            sum(SELECTION_SCORE_WEIGHTS[name] * float(components.get(name, 1.0)) for name in names)
            / denominator
        )

    return score(baseline_components), score(candidate_components)


def _pooled_physical_comparison(
    *,
    name: str,
    candidate: Sequence[DynamicSetExampleScoreEvidence],
    baseline: Sequence[DynamicSetExampleScoreEvidence],
) -> AggregateScoreComparison:
    candidate_components = pooled_selection_score_evidence(candidate).components
    baseline_components = pooled_selection_score_evidence(baseline).components
    baseline_score, candidate_score = _paired_subset_scores(
        baseline_components,
        candidate_components,
    )
    return AggregateScoreComparison(name, baseline_score, candidate_score).validate()


def _is_accepted_like_physical_row(
    row: PhysicalManifestRow,
    *,
    object_count: int,
) -> bool:
    """Identify the exact legacy-like static, action-free RGB-D population."""

    expected_distribution = (
        "compositional_ood" if row.split == "compositional_ood" else "in_distribution"
    )
    return bool(
        row.object_count == object_count
        and not row.contact
        and not row.dynamic_membership
        and row.lifecycle_schedule == "none"
        and not row.known_action
        and row.contact_origin == "none"
        and row.action_target_rank is None
        and row.action_time_stratum is None
        and row.contact_geometry == "none"
        and row.distribution == expected_distribution
    )


def _accepted_like_population(
    rows: Sequence[PhysicalManifestRow],
    *,
    object_count: int,
) -> tuple[PhysicalManifestRow, ...]:
    """Return and authenticate one frozen accepted-like N=1/N=2 subset."""

    population = tuple(
        row for row in rows if _is_accepted_like_physical_row(row, object_count=object_count)
    )
    splits = {row.split for row in rows}
    if len(splits) != 1:
        raise ValueError("accepted-like reconstruction requires one complete physical split")
    split = next(iter(splits))
    try:
        expected_sha256 = ACCEPTED_LIKE_POPULATION_SHA256[split][object_count]
        expected_count = PHYSICAL_SPLIT_SIZES[split] // len(PHYSICAL_CELLS) // 2
    except KeyError as error:
        raise ValueError("accepted-like population split/cardinality is unsupported") from error
    if (
        len(population) != expected_count
        or canonical_sha256([asdict(row) for row in population]) != expected_sha256
    ):
        raise ValueError("accepted-like population differs from its frozen row binding")
    return population


def _physical_regression_comparisons(
    rows: Sequence[PhysicalManifestRow],
    candidate: Sequence[DynamicSetExampleScoreEvidence],
    baseline: Sequence[DynamicSetExampleScoreEvidence],
) -> tuple[tuple[AggregateScoreComparison, ...], tuple[AggregateScoreComparison, ...]]:
    """Reconstruct every critical and accepted-like comparison from raw rows."""

    rows_value = tuple(rows)
    candidate_value = tuple(candidate)
    baseline_value = tuple(baseline)
    expected_keys = tuple(
        (row.split, row.ordinal, row.seed, row.cell_index, canonical_sha256(asdict(row)))
        for row in rows_value
    )
    if (
        not rows_value
        or len(candidate_value) != len(rows_value)
        or len(baseline_value) != len(rows_value)
        or tuple(record.pair_key for record in candidate_value) != expected_keys
        or tuple(record.pair_key for record in baseline_value) != expected_keys
    ):
        raise ValueError("physical regression evidence differs from its exact row population")
    triples = tuple(zip(rows_value, candidate_value, baseline_value, strict=True))
    critical = tuple(
        _pooled_physical_comparison(
            name=(
                f"N{cell.object_count}/contact={int(cell.contact)}/"
                f"dynamic={int(cell.dynamic_membership)}"
            ),
            candidate=tuple(
                candidate_record
                for row, candidate_record, _baseline_record in triples
                if row.cell_index == cell_index
            ),
            baseline=tuple(
                baseline_record
                for row, _candidate_record, baseline_record in triples
                if row.cell_index == cell_index
            ),
        )
        for cell_index, cell in enumerate(PHYSICAL_CELLS)
    )
    accepted_like: list[AggregateScoreComparison] = []
    for object_count in (1, 2):
        population = _accepted_like_population(rows_value, object_count=object_count)
        ordinals = {row.ordinal for row in population}
        accepted_like.append(
            _pooled_physical_comparison(
                name=f"accepted_like/{ACCEPTED_LIKE_SCHEMA}/N{object_count}",
                candidate=tuple(
                    candidate_record
                    for row, candidate_record, _baseline_record in triples
                    if row.ordinal in ordinals
                ),
                baseline=tuple(
                    baseline_record
                    for row, _candidate_record, baseline_record in triples
                    if row.ordinal in ordinals
                ),
            )
        )
    return critical, tuple(accepted_like)


def _rebuild_promotion_integrity(
    evidence: DynamicSetPromotionEvidence,
) -> PromotionIntegrityMetrics:
    """Rebuild governed promotion metrics from raw paired populations."""

    if type(evidence) is not DynamicSetPromotionEvidence:
        raise TypeError("promotion evidence must be exact DynamicSetPromotionEvidence")
    paired = pooled_paired_improvement_evidence(
        evidence.candidate_physical,
        evidence.baseline_physical,
        evidence.candidate_planning,
        evidence.baseline_planning,
        bootstrap_samples=evidence.bootstrap_samples,
        bootstrap_seed=evidence.bootstrap_seed,
    )
    if (
        paired.candidate_aggregate_score != evidence.candidate_aggregate_score
        or paired.baseline_aggregate_score != evidence.baseline_aggregate_score
    ):
        raise ValueError("paired point estimate differs from aggregate selection scores")
    return build_promotion_integrity_metrics_from_paired_evidence(
        paired,
        critical_comparisons=evidence.critical_comparisons,
        accepted_like_comparisons=evidence.accepted_like_comparisons,
        truth_leakage_count=evidence.truth_leakage_count,
        fabricated_target_count=evidence.fabricated_target_count,
        nonfinite_state_count=evidence.nonfinite_state_count,
        rejected_optimizer_mutation_count=evidence.rejected_optimizer_mutation_count,
        minimum_complete_gradient_retention=evidence.minimum_complete_gradient_retention,
    )


def _nonfinite_numeric_count(value: object) -> int:
    """Count nonfinite numeric leaves in detached scored evidence."""

    if isinstance(value, bool) or value is None or isinstance(value, (str, bytes, int)):
        return 0
    if isinstance(value, float):
        return int(not math.isfinite(value))
    if isinstance(value, Mapping):
        return sum(
            _nonfinite_numeric_count(key) + _nonfinite_numeric_count(item)
            for key, item in value.items()
        )
    if isinstance(value, (tuple, list)):
        return sum(_nonfinite_numeric_count(item) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return sum(_nonfinite_numeric_count(getattr(value, item.name)) for item in fields(value))
    raise TypeError(f"unsupported detached integrity-evidence type {type(value).__name__}")


def _derived_scored_integrity_counts(
    physical: Sequence[DynamicSetExampleScoreEvidence],
    planning: Sequence[Any],
) -> tuple[int, int, int]:
    """Reconstruct leakage/fabrication/finite-state counts from scored rows.

    Truth leakage is summed from each row's digest-bound public-boundary
    evidence. Planning fabrication is an observable schema fact: an unresolved
    handle may not carry an evaluation, while a resolved handle must carry one
    and no failure reason. All numeric leaves are then exhaustively scanned.
    """

    physical_rows = tuple(physical)
    planning_rows = tuple(planning)
    if not physical_rows or not planning_rows:
        raise ValueError("integrity derivation requires complete physical and planning evidence")
    truth_leakage = dynamic_set_truth_leakage_count(physical_rows)
    fabricated = 0
    for outcome in planning_rows:
        resolved = bool(getattr(outcome, "handle_resolved", False))
        evaluation = getattr(outcome, "evaluation", None)
        failure_reason = getattr(outcome, "failure_reason", None)
        fabricated += int(
            (resolved and (evaluation is None or failure_reason is not None))
            or (not resolved and (evaluation is not None or not failure_reason))
        )
    nonfinite = _nonfinite_numeric_count((physical_rows, planning_rows))
    return truth_leakage, fabricated, nonfinite


def build_dynamic_set_split_evidence(
    context: SplitExecutionContext,
    *,
    candidate_physical: DynamicSetEvaluationResult,
    baseline_physical: DynamicSetEvaluationResult,
    candidate_planning: PlanningPopulationEvaluationResult,
    baseline_planning: PlanningPopulationEvaluationResult,
    integrity: DynamicSetIntegrityEvidence,
    planning_pair_provenance: Sequence[PlanningPairProvenance] = (),
    fresh_resource_evidence: FreshWorkerResourceEvidence | None = None,
) -> DynamicSetSplitEvidence:
    """Bind concrete physical/planning evaluators into qualification evidence.

    The paired bootstrap uses every exact physical row and planning task.  It
    rebuilds nonlinear physical components from pooled additive evidence and
    independently resamples the planning population under the frozen
    85%/15% physical/planning split.
    """

    if type(context) is not SplitExecutionContext:
        raise TypeError("concrete evidence construction requires SplitExecutionContext")
    if type(integrity) is not DynamicSetIntegrityEvidence:
        raise TypeError("integrity facts must use exact DynamicSetIntegrityEvidence")
    candidate_records = _physical_result_records(candidate_physical, context)
    baseline_records = _physical_result_records(baseline_physical, context)
    candidate_outcomes = _planning_result_outcomes(candidate_planning, context)
    baseline_outcomes = _planning_result_outcomes(baseline_planning, context)
    derived_integrity = _derived_scored_integrity_counts(
        candidate_records,
        candidate_outcomes,
    )
    supplied_integrity = (
        integrity.truth_leakage_count,
        integrity.fabricated_target_count,
        integrity.nonfinite_state_count,
    )
    if supplied_integrity != derived_integrity:
        raise ValueError("run-level integrity claims differ from scored-evidence derivation")
    planning_provenance = tuple(planning_pair_provenance)
    if planning_provenance:
        if len(planning_provenance) != len(context.planning_rows):
            raise ValueError("planning pair provenance does not cover the exact task population")
        for row, provenance, candidate_outcome, baseline_outcome in zip(
            context.planning_rows,
            planning_provenance,
            candidate_outcomes,
            baseline_outcomes,
            strict=True,
        ):
            if type(provenance) is not PlanningPairProvenance:
                raise TypeError("planning pair provenance must use its exact evidence type")
            provenance.validate(row)
            for outcome in (candidate_outcome, baseline_outcome):
                if (
                    outcome.evaluation is not None
                    and outcome.evaluation.private_oracle_sha256 != provenance.private_oracle_sha256
                ):
                    raise ValueError("planning score outcome differs from paired oracle provenance")
    score_components = {
        **dict(candidate_physical.score.components),
        "planning_error": candidate_planning.planning_error,
    }
    baseline_score_components = {
        **dict(baseline_physical.score.components),
        "planning_error": baseline_planning.planning_error,
    }
    candidate_aggregate_score = selection_score(score_components)
    baseline_aggregate_score = selection_score(baseline_score_components)

    critical, accepted_like = _physical_regression_comparisons(
        context.physical_rows,
        candidate_records,
        baseline_records,
    )
    promotion = DynamicSetPromotionEvidence(
        candidate_physical=candidate_records,
        baseline_physical=baseline_records,
        candidate_planning=tuple(
            PlanningTaskScoreEvidence.create(outcome) for outcome in candidate_outcomes
        ),
        baseline_planning=tuple(
            PlanningTaskScoreEvidence.create(outcome) for outcome in baseline_outcomes
        ),
        candidate_aggregate_score=candidate_aggregate_score,
        baseline_aggregate_score=baseline_aggregate_score,
        critical_comparisons=critical,
        accepted_like_comparisons=accepted_like,
        truth_leakage_count=integrity.truth_leakage_count,
        fabricated_target_count=integrity.fabricated_target_count,
        nonfinite_state_count=integrity.nonfinite_state_count,
        rejected_optimizer_mutation_count=integrity.rejected_optimizer_mutation_count,
        minimum_complete_gradient_retention=integrity.minimum_complete_gradient_retention,
        bootstrap_samples=integrity.bootstrap_samples,
        bootstrap_seed=integrity.bootstrap_seed,
    )
    rebuilt_integrity = _rebuild_promotion_integrity(promotion)
    resources = candidate_physical.resources
    fresh_resource_payload: Mapping[str, Any] | None = None
    if fresh_resource_evidence is not None:
        if type(fresh_resource_evidence) is not FreshWorkerResourceEvidence:
            raise TypeError("fresh resource evidence must use its exact repository type")
        fresh_resource_evidence.validate()
        resources = fresh_resource_evidence.resource_metrics
        fresh_resource_payload = fresh_resource_evidence.to_mapping()
    return DynamicSetSplitEvidence(
        split=context.split,
        protocol_sha256=context.protocol_sha256,
        checkpoint_sha256=context.checkpoint_sha256,
        model_state_sha256=context.model_state_sha256,
        completed_updates=context.completed_updates,
        physical_manifest_sha256=context.physical_manifest_sha256,
        physical_row_count=len(candidate_records),
        planning_manifest_sha256=context.planning_manifest_sha256,
        planning_task_count=len(candidate_outcomes),
        physical_cells=tuple(
            PhysicalCellEvaluation(cell, candidate_physical.by_cell[cell])
            for cell in PHYSICAL_CELLS
        ),
        planning_slices=tuple(candidate_planning.reduction.slices),
        planning_invariants=candidate_planning.invariants,
        promotion_evidence=promotion,
        promotion_integrity=rebuilt_integrity,
        resources=resources,
        score_components=score_components,
        training_support_passed=integrity.training_support_passed,
        callback_binding_sha256=context.callback_binding_sha256,
        planning_pair_provenance=planning_provenance,
        fresh_resource_evidence=fresh_resource_payload,
    )


def _validated_repository_config(config: OrpheusConfig) -> OrpheusConfig:
    if not isinstance(config, OrpheusConfig):
        raise TypeError("repository evaluation requires one OrpheusConfig")
    config.validate()
    canonical_sha256(config.to_dict())
    return config


def _trainer_checkpoint_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Remove executor-owned timing after qualification authenticates it."""

    value = dict(payload)
    if "execution_timing" not in value:
        raise ValueError("qualification checkpoint lacks executor timing evidence")
    del value["execution_timing"]
    return value


def _load_repository_candidate(
    context: SplitExecutionContext,
    *,
    checkpoint_path: str | Path,
    config: OrpheusConfig,
    protocol: Mapping[str, Any],
) -> tuple[OnlineWorldModel, _RepositoryTrainingIntegrity]:
    """Safely load and fully validate one exact-resume trainer checkpoint."""

    contents = _stable_regular_bytes(checkpoint_path)
    if sha256_bytes(contents) != context.checkpoint_sha256:
        raise ValueError("repository callback checkpoint bytes differ from split context")
    try:
        payload = torch.load(io.BytesIO(contents), map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("dynamic-set checkpoint is not a safe tensor payload") from error
    if type(payload) is not dict or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("repository callback requires a dynamic-set trainer checkpoint")
    model_state = payload.get("model_state")
    if not isinstance(model_state, Mapping):
        raise TypeError("dynamic-set checkpoint lacks a model-state mapping")
    model_state_sha256 = dynamic_set_model_state_sha256(model_state)
    if (
        validated_sha256(payload.get("model_state_sha256"), label="checkpoint model state")
        != model_state_sha256
        or model_state_sha256 != context.model_state_sha256
    ):
        raise ValueError("checkpoint model-state digest differs from split context")
    next_sample_state = payload.get("next_sample_state")
    if not isinstance(next_sample_state, Mapping):
        raise ValueError("dynamic-set checkpoint lacks next-sample state")
    schedule_seed = next_sample_state.get("schedule_seed")
    if isinstance(schedule_seed, bool) or not isinstance(schedule_seed, int):
        raise ValueError("dynamic-set checkpoint schedule seed is invalid")
    _checkpoint_timing_progress_fields(
        payload,
        completed_updates=context.completed_updates,
    )

    model = OnlineWorldModel.from_config(config, device="cpu")
    trainer = DynamicSetTrainer.from_online_world_model(
        model=model,
        training_rows=physical_manifest("training"),
        objective_adapter=DynamicSetEpisodeObjectiveAdapter(),
        resolved_config=config,
        source_provenance=_exact_dict(protocol["source"], label="protocol source"),
        schedule_seed=schedule_seed,
    )
    trainer.load_checkpoint_payload(
        _trainer_checkpoint_payload(payload),
        restore_rng=False,
    )
    if trainer.completed_updates != context.completed_updates:
        raise ValueError("checkpoint update count differs from split context")
    if trainer.bindings.protocol_sha256 != dynamic_set_training_protocol_sha256():
        raise ValueError("checkpoint training protocol differs from qualification")
    if trainer.bindings.manifest_sha256 != FROZEN_PHYSICAL_MANIFEST_SHA256["training"]:
        raise ValueError("checkpoint training manifest differs from the 66,000-row freeze")
    trainer_state = _exact_dict(payload.get("trainer_state"), label="checkpoint trainer state")
    minimum_retention = trainer_state.get("minimum_complete_gradient_retention")
    rejected_mutations = trainer_state.get("rejected_optimizer_mutation_count")
    if (
        minimum_retention != trainer.observed_minimum_complete_gradient_retention
        or rejected_mutations != trainer.rejected_optimizer_mutation_count
    ):
        raise ValueError("checkpoint training-integrity aggregates differ after exact restore")
    training_support_passed = bool(
        context.completed_updates >= DEFAULT_CAMPAIGN.minimum_updates
        and context.completed_updates % DEFAULT_CAMPAIGN.validation_interval_updates == 0
    )
    training_integrity = _RepositoryTrainingIntegrity(
        rejected_optimizer_mutation_count=trainer.rejected_optimizer_mutation_count,
        minimum_complete_gradient_retention=(trainer.observed_minimum_complete_gradient_retention),
        training_support_passed=training_support_passed,
    ).validate()
    model.eval()
    return model, training_integrity


def _zero_residual_repository_baseline(config: OrpheusConfig) -> OnlineWorldModel:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(161_061)
        model = OnlineWorldModel.from_config(config, device="cpu")
    try:
        proposer = model.observation_modules["rgbd"].set_proposer
        relation_output = model.dynamics.interactions.edge_network.output
        zero_layers = (
            proposer.mask_residual_projection,
            proposer.existence_residual_head,
            proposer.appearance_residual_head,
            proposer.log_variance_residual_head,
            relation_output,
        )
    except (AttributeError, KeyError) as error:
        raise ValueError("configured baseline lacks the declared residual output heads") from error
    for layer in zero_layers:
        if not isinstance(layer, nn.Linear) or any(
            bool(torch.count_nonzero(parameter.detach())) for parameter in layer.parameters()
        ):
            raise ValueError("fresh structured baseline is not exactly zero-residual")
    model.eval()
    return model


def _development_evaluation_cache_binding(
    *,
    config: OrpheusConfig,
    protocol: Mapping[str, Any],
) -> DynamicSetDevelopmentEvaluationCacheBinding:
    evaluator = _exact_dict(
        protocol.get("repository_evaluator"),
        label="development cache repository evaluator",
    )
    return DynamicSetDevelopmentEvaluationCacheBinding(
        source_sha256=canonical_sha256(protocol["source"]),
        config_sha256=canonical_sha256(config.to_dict()),
        physical_manifest_sha256=FROZEN_PHYSICAL_MANIFEST_SHA256["development"],
        planning_manifest_sha256=FROZEN_PLANNING_MANIFEST_SHA256["development"],
        evaluator_sha256=validated_sha256(
            evaluator["evaluator_sha256"],
            label="development cache evaluator",
        ),
    ).validate()


def _evaluate_repository_physical(
    model: OnlineWorldModel,
    context: SplitExecutionContext,
) -> DynamicSetEvaluationResult:
    if context.split == "development":
        if context.development_evaluation_cache is None:
            raise PermissionError("formal development physical evaluation lacks its bound cache")
        materializations = (
            context.development_evaluation_cache.physical(row) for row in context.physical_rows
        )
        return evaluate_dynamic_set_materializations(
            model,
            materializations,
            allowed_split="development",
            require_all_cells=True,
        )
    if context.protected_ledger is None:
        raise PermissionError("protected physical evaluation lacks its durable ledger")
    return evaluate_authorized_dynamic_set_rows(
        model,
        ledger=context.protected_ledger,
        permit=context.permit,
        expected_protocol_sha256=context.protocol_sha256,
        expected_manifest_sha256=context.physical_manifest_sha256,
        expected_rows=context.physical_rows,
        require_all_cells=True,
    )


def _evaluate_repository_physical_pair(
    candidate: OnlineWorldModel,
    baseline: OnlineWorldModel,
    context: SplitExecutionContext,
) -> DynamicSetPairedEvaluationResult:
    """Materialize each bound physical row once for both public models."""

    if context.split == "development":
        if context.development_evaluation_cache is None:
            raise PermissionError("formal development physical evaluation lacks its bound cache")
        materializations = (
            context.development_evaluation_cache.physical(row) for row in context.physical_rows
        )
        return evaluate_paired_dynamic_set_materializations(
            candidate,
            baseline,
            materializations,
            allowed_split="development",
            require_all_cells=True,
        )
    if context.protected_ledger is None:
        raise PermissionError("protected physical evaluation lacks its durable ledger")
    return evaluate_authorized_paired_dynamic_set_rows(
        candidate,
        baseline,
        ledger=context.protected_ledger,
        permit=context.permit,
        expected_protocol_sha256=context.protocol_sha256,
        expected_manifest_sha256=context.physical_manifest_sha256,
        expected_rows=context.physical_rows,
        require_all_cells=True,
    )


def _evaluate_repository_planning_pair(
    candidate: OnlineWorldModel,
    baseline: OnlineWorldModel,
    context: SplitExecutionContext,
) -> PlanningPairedPopulationEvaluationResult:
    """Materialize each task once and defer one oracle opening until both plans finish."""

    if context.split == "development":
        if context.development_evaluation_cache is None:
            raise PermissionError("formal development planning evaluation lacks its bound cache")
        materializations = (
            context.development_evaluation_cache.planning(row) for row in context.planning_rows
        )
        result = evaluate_paired_development_planning_materializations(
            candidate,
            baseline,
            materializations,
        )
    else:
        if context.protected_ledger is None:
            raise PermissionError("protected planning evaluation lacks its durable ledger")
        result = evaluate_authorized_paired_planning_rows(
            candidate,
            baseline,
            ledger=context.protected_ledger,
            permit=context.permit,
            expected_protocol_sha256=context.protocol_sha256,
            expected_manifest_sha256=context.planning_manifest_sha256,
            expected_rows=context.planning_rows,
        )
    return validate_paired_planning_population_evaluation_result(result)


def _complete_repository_integrity(
    training: _RepositoryTrainingIntegrity,
    candidate_physical: DynamicSetEvaluationResult,
    candidate_planning: PlanningPopulationEvaluationResult,
) -> DynamicSetIntegrityEvidence:
    """Combine checkpoint facts with integrity derived after public scoring."""

    if type(training) is not _RepositoryTrainingIntegrity:
        raise TypeError("repository training integrity must use its exact evidence type")
    training.validate()
    if type(candidate_physical) is not DynamicSetEvaluationResult:
        raise TypeError("repository physical integrity requires an exact evaluation result")
    if type(candidate_planning) is not PlanningPopulationEvaluationResult:
        raise TypeError("repository planning integrity requires an exact evaluation result")
    truth_leakage, fabricated_targets, nonfinite_state = _derived_scored_integrity_counts(
        candidate_physical.per_example_score_evidence,
        candidate_planning.outcomes,
    )
    return DynamicSetIntegrityEvidence(
        truth_leakage_count=truth_leakage,
        fabricated_target_count=fabricated_targets,
        nonfinite_state_count=nonfinite_state,
        rejected_optimizer_mutation_count=training.rejected_optimizer_mutation_count,
        minimum_complete_gradient_retention=training.minimum_complete_gradient_retention,
        training_support_passed=training.training_support_passed,
    )


def evaluate_repository_dynamic_set_split(
    context: SplitExecutionContext,
    *,
    checkpoint_path: str | Path,
    resolved_config: OrpheusConfig,
    qualification_protocol: Mapping[str, Any],
) -> DynamicSetSplitEvidence:
    """Run concrete physical, planning, baseline, and paired split evaluation.

    ``PlanningManifestRow`` materialization is repository-owned by
    :mod:`dynamic_set_planning_materializer`; it is not an injected callback.
    Protected evaluator calls receive the exact same ledger permit and never
    open or advance a split themselves.
    """

    if type(context) is not SplitExecutionContext:
        raise TypeError("repository split evaluation requires SplitExecutionContext")
    config = _validated_repository_config(resolved_config)
    protocol = _validate_protocol(qualification_protocol)
    if protocol["protocol_sha256"] != context.protocol_sha256:
        raise ValueError("repository callback protocol differs from split context")
    candidate, training_integrity = _load_repository_candidate(
        context,
        checkpoint_path=checkpoint_path,
        config=config,
        protocol=protocol,
    )
    fresh_resources = measure_fresh_worker_resources(
        checkpoint_path=checkpoint_path,
        resolved_config=config,
        expected_checkpoint_sha256=context.checkpoint_sha256,
        expected_model_state_sha256=context.model_state_sha256,
        expected_config_sha256=canonical_sha256(config.to_dict()),
        expected_source_sha256=canonical_sha256(protocol["source"]),
    )
    baseline = _zero_residual_repository_baseline(config)
    paired_physical = _evaluate_repository_physical_pair(candidate, baseline, context)
    paired_planning = _evaluate_repository_planning_pair(candidate, baseline, context)
    integrity = _complete_repository_integrity(
        training_integrity,
        paired_physical.candidate,
        paired_planning.candidate,
    )
    return build_dynamic_set_split_evidence(
        context,
        candidate_physical=paired_physical.candidate,
        baseline_physical=paired_physical.reference,
        candidate_planning=paired_planning.candidate,
        baseline_planning=paired_planning.reference,
        integrity=integrity,
        planning_pair_provenance=paired_planning.provenance,
        fresh_resource_evidence=fresh_resources,
    )


def _screen_population_objective(
    model: OnlineWorldModel,
    rows: Sequence[PhysicalManifestRow],
    adapter: DynamicSetEpisodeObjectiveAdapter,
    *,
    materializer: Callable[[PhysicalManifestRow], object] = materialize_dynamic_set_episode,
) -> float:
    if len(rows) != 64 or tuple(rows) != SCREEN_ROWS:
        raise ValueError("screen objective requires the exact frozen 64 rows")
    if not callable(materializer):
        raise TypeError("screen materializer must be callable")
    values: list[float] = []
    for microbatch in _screen_training_microbatches(rows, materializer=materializer):
        inputs = adapter.build_objective_inputs(model, microbatch)
        losses = dynamic_set_objective(inputs.perception, inputs.dynamics)
        value = float(losses.total.detach())
        if not math.isfinite(value):
            raise FloatingPointError("screen optimization objective is nonfinite")
        values.append(value)
    return math.fsum(values) / len(values)


def _screen_training_microbatches(
    rows: Sequence[PhysicalManifestRow],
    *,
    materializer: Callable[[PhysicalManifestRow], object],
) -> tuple[DynamicSetTrainingMicrobatch, ...]:
    """Materialize the frozen screen with the same frame schedule as training."""

    if len(rows) != 64 or tuple(rows) != SCREEN_ROWS:
        raise ValueError("screen microbatches require the exact frozen 64 rows")
    if not callable(materializer):
        raise TypeError("screen materializer must be callable")
    contextual = getattr(materializer, "materialize_for_training", None)
    batches: list[DynamicSetTrainingMicrobatch] = []
    for offset in range(0, len(rows), MICROBATCH_SIZE):
        batch_rows = tuple(rows[offset : offset + MICROBATCH_SIZE])
        batch_index = offset // MICROBATCH_SIZE
        update_index = batch_index // 6
        microbatch_index = batch_index % 6
        if callable(contextual):
            perception_frame_index = dynamic_set_perception_frame_index(
                update_index,
                microbatch_index,
            )
            materializations = tuple(
                contextual(row, perception_frame_index=perception_frame_index) for row in batch_rows
            )
        else:
            materializations = tuple(materializer(row) for row in batch_rows)
        batches.append(
            DynamicSetTrainingMicrobatch(
                update_index=update_index,
                microbatch_index=microbatch_index,
                dataset_indices=tuple(range(offset, offset + MICROBATCH_SIZE)),
                rows=batch_rows,
                materializations=materializations,
            )
        )
    return tuple(batches)


def _screen_architecture_diagnostic(
    model: OnlineWorldModel,
    evaluation: DynamicSetEvaluationResult,
    adapter: DynamicSetEpisodeObjectiveAdapter,
    materializer: Callable[[PhysicalManifestRow], object],
    *,
    protocol_sha256: str,
    source_sha256: str,
    resolved_config_sha256: str,
) -> ScreenArchitectureDiagnosticEvidence:
    """Run the frozen perception and truth-state dynamics ownership probes."""

    records = tuple(evaluation.per_example_score_evidence)
    if len(records) != len(SCREEN_ROWS):
        raise ValueError("screen diagnostic requires every per-row physical record")
    perception_by_ordinal: dict[int, bool] = {}
    raw_rows: list[dict[str, Any]] = []
    perception_names = {
        "proposal_precision",
        "proposal_recall",
        "proposal_f1",
        "exact_count_accuracy",
        "current_position_rmse_m",
    }
    for row, record in zip(SCREEN_ROWS, records, strict=True):
        if record.ordinal != row.ordinal or record.row_sha256 != canonical_sha256(asdict(row)):
            raise ValueError("screen diagnostic physical row order differs")
        metrics = record.additive.to_accumulator().physical_metrics(PHYSICAL_CELLS[row.cell_index])
        failures = physical_cell_gate_failures(PHYSICAL_CELLS[row.cell_index], metrics)
        perception_failed = any(
            failure.split(":", 1)[0] in perception_names for failure in failures
        )
        perception_by_ordinal[row.ordinal] = perception_failed

    oracle_failures = 0
    contact_failures = 0
    noncontact_failures = 0
    contact_error = 0.0
    noncontact_error = 0.0
    identity = torch.eye(6, dtype=torch.bool)
    for microbatch in _screen_training_microbatches(SCREEN_ROWS, materializer=materializer):
        dynamics = adapter.build_objective_inputs(model, microbatch).dynamics
        predicted = dynamics.predicted_state.detach().to(device="cpu", dtype=torch.float64)
        target = dynamics.target_state.detach().to(device="cpu", dtype=torch.float64)
        stable = (
            (
                dynamics.contact_window_mask
                | dynamics.known_action_predictable_mask
                | dynamics.unaffected_object_mask
            )
            .detach()
            .cpu()
        )
        logits = dynamics.collision_logits.detach().cpu()
        collision_target = dynamics.collision_target.detach().cpu()
        collision_support = dynamics.collision_support.detach().cpu()
        for index, row in enumerate(microbatch.rows):
            support = stable[index]
            if bool(support.any()):
                position_error = (
                    predicted[index, :, :, :3][support] - target[index, :, :, :3][support]
                )
                velocity_error = (
                    predicted[index, :, :, 3:6][support] - target[index, :, :, 3:6][support]
                )
                position_rmse = float(position_error.square().mean().sqrt())
                velocity_rmse = float(velocity_error.square().mean().sqrt())
            else:
                position_rmse = 1.0
                velocity_rmse = 1.0
            upper = collision_support[index].bool() & torch.triu(~identity, diagonal=1)
            predicted_collision = logits[index] >= 0.0
            expected_collision = collision_target[index].bool()
            true_positive = int((predicted_collision & expected_collision & upper).sum())
            false_positive = int((predicted_collision & ~expected_collision & upper).sum())
            false_negative = int((~predicted_collision & expected_collision & upper).sum())
            f1_support = 2 * true_positive + false_positive + false_negative
            collision_error = (
                float(false_positive > 0)
                if f1_support == 0
                else 1.0 - 2.0 * true_positive / f1_support
            )
            velocity_limit = 0.050 if row.contact or row.dynamic_membership else 0.020
            normalized_error = (
                position_rmse / 0.050 + velocity_rmse / velocity_limit + collision_error
            )
            oracle_failed = bool(
                position_rmse > 0.050 or velocity_rmse > velocity_limit or collision_error > 0.05
            )
            oracle_failures += int(oracle_failed)
            if row.contact:
                contact_failures += int(oracle_failed)
                contact_error += normalized_error
            else:
                noncontact_failures += int(oracle_failed)
                noncontact_error += normalized_error
            raw_rows.append(
                {
                    "row_sha256": canonical_sha256(asdict(row)),
                    "ordinal": row.ordinal,
                    "cell_index": row.cell_index,
                    "contact": row.contact,
                    "perception_failed": perception_by_ordinal[row.ordinal],
                    "position_rmse_m": position_rmse,
                    "velocity_rmse_mps": velocity_rmse,
                    "collision_error": collision_error,
                    "oracle_state_dynamics_failed": oracle_failed,
                    "normalized_truth_state_error": normalized_error,
                }
            )
    if len(raw_rows) != len(SCREEN_ROWS) or [item["ordinal"] for item in raw_rows] != [
        row.ordinal for row in SCREEN_ROWS
    ]:
        raise RuntimeError("screen architecture diagnostics lost frozen row order")
    return ScreenArchitectureDiagnosticEvidence.create(
        protocol_sha256=protocol_sha256,
        source_sha256=source_sha256,
        resolved_config_sha256=resolved_config_sha256,
        perception_failure_count=sum(perception_by_ordinal.values()),
        oracle_state_dynamics_failure_count=oracle_failures,
        truth_state_contact_failure_count=contact_failures,
        truth_state_noncontact_failure_count=noncontact_failures,
        truth_state_contact_error=contact_error,
        truth_state_noncontact_error=noncontact_error,
        raw_evidence_sha256=canonical_sha256(raw_rows),
    )


def _validated_screen_work_directory(
    path: str | Path,
    *,
    qualification_root: str | Path,
) -> Path:
    """Create one dedicated cache-compatible directory outside sealed evidence."""

    requested = Path(path).absolute()
    if requested.is_symlink():
        raise OSError("screen work path must not be a symbolic link")
    qualification = Path(qualification_root).resolve(strict=True)
    if requested.exists() and not requested.is_dir():
        raise OSError("screen work path must be one real directory")
    requested.mkdir(parents=True, exist_ok=True, mode=0o700)
    resolved = requested.resolve(strict=True)
    if resolved != requested:
        raise OSError("screen work path may not traverse symbolic links")
    if (
        resolved == qualification
        or resolved in qualification.parents
        or qualification in resolved.parents
    ):
        raise ValueError("screen work directory must not overlap sealed qualification artifacts")
    for item in resolved.iterdir():
        if item.name != TRAINING_CACHE_DIRECTORY_NAME:
            raise OSError(f"screen work directory contains unsupported artifact: {item.name}")
        metadata = os.lstat(item)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise OSError("screen training cache must be one real directory")
        validate_training_cache_directory(item)
    return resolved


def evaluate_repository_disposable_screen(
    context: ScreenExecutionContext,
    *,
    resolved_config: OrpheusConfig,
    qualification_protocol: Mapping[str, Any],
    work_directory: str | Path,
    qualification_root: str | Path,
    updates: int = 512,
) -> ScreenExecutionResult:
    """Train/evaluate the exact frozen 64-row disposable screen once."""

    if type(context) is not ScreenExecutionContext:
        raise TypeError("repository screen evaluation requires ScreenExecutionContext")
    config = _validated_repository_config(resolved_config)
    protocol = _validate_protocol(qualification_protocol)
    if (
        protocol["protocol_sha256"] != context.protocol_sha256
        or context.rows != SCREEN_ROWS
        or context.manifest_sha256 != SCREEN_MANIFEST_SHA256
    ):
        raise ValueError("repository screen context differs from its protocol freeze")
    work = _validated_screen_work_directory(
        work_directory,
        qualification_root=qualification_root,
    )
    source = _exact_dict(protocol["source"], label="protocol source")
    training_cache = DynamicSetTrainingCache(
        work / TRAINING_CACHE_DIRECTORY_NAME,
        binding=DynamicSetTrainingCacheBinding(
            source_sha256=canonical_sha256(source),
            config_sha256=protocol["config_sha256"],
            training_manifest_sha256=FROZEN_PHYSICAL_MANIFEST_SHA256["training"],
        ),
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(161_061)
        model = OnlineWorldModel.from_config(config, device="cpu")
        adapter = DynamicSetEpisodeObjectiveAdapter()
        trainer = DynamicSetTrainer.from_online_world_model(
            model=model,
            training_rows=SCREEN_ROWS,
            objective_adapter=adapter,
            resolved_config=config,
            source_provenance=source,
            schedule_seed=161_061,
            materializer=training_cache,
            screen_only=True,
        )
        screen_evaluations: list[DynamicSetEvaluationResult] = []

        def snapshot(current: nn.Module, _step: int) -> Any:
            if not isinstance(current, OnlineWorldModel):
                raise TypeError("screen trainer changed its model type")
            result = evaluate_dynamic_set_materializations(
                current,
                (training_cache(row) for row in SCREEN_ROWS),
                allowed_split="training",
                require_all_cells=True,
            )
            screen_evaluations.append(result)
            objective = _screen_population_objective(
                current,
                SCREEN_ROWS,
                adapter,
                materializer=training_cache,
            )
            return screen_snapshot_from_result(
                result,
                optimization_objective=objective,
            )

        report = trainer.run_disposable_screen(
            updates=updates,
            evaluation_hook=snapshot,
        )
        if len(screen_evaluations) != 2:
            raise RuntimeError("disposable screen must produce exact before/after evaluations")
        architecture_diagnostics = _screen_architecture_diagnostic(
            model,
            screen_evaluations[-1],
            adapter,
            training_cache,
            protocol_sha256=protocol["protocol_sha256"],
            source_sha256=protocol["source"]["source_sha256"],
            resolved_config_sha256=canonical_sha256(config.to_dict()),
        )
    return ScreenExecutionResult(
        metrics=report.metrics,
        callback_binding_sha256=context.callback_binding_sha256,
        architecture_diagnostics=architecture_diagnostics,
        cache_evidence=ScreenCacheEvidence.create(training_cache),
    )


def _supported_payload(value: SupportedScalar | None) -> dict[str, Any] | None:
    return None if value is None else {"value": float(value.value), "support": value.support}


def _physical_metrics_payload(value: PhysicalCellMetrics) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for item in fields(value):
        current = getattr(value, item.name)
        if item.name == "horizon_position_rmse_m":
            payload[item.name] = {
                f"{float(horizon):.2f}": _supported_payload(metric)
                for horizon, metric in sorted(current.items())
            }
        else:
            payload[item.name] = _supported_payload(current)
    return payload


def _raw_promotion_payload(value: DynamicSetPromotionEvidence) -> dict[str, Any]:
    return {
        "candidate_physical": [asdict(item) for item in value.candidate_physical],
        "baseline_physical": [asdict(item) for item in value.baseline_physical],
        "candidate_planning": [asdict(item) for item in value.candidate_planning],
        "baseline_planning": [asdict(item) for item in value.baseline_planning],
        "candidate_aggregate_score": value.candidate_aggregate_score,
        "baseline_aggregate_score": value.baseline_aggregate_score,
        "critical_comparisons": [asdict(item) for item in value.critical_comparisons],
        "accepted_like_comparisons": [asdict(item) for item in value.accepted_like_comparisons],
        "truth_leakage_count": value.truth_leakage_count,
        "fabricated_target_count": value.fabricated_target_count,
        "nonfinite_state_count": value.nonfinite_state_count,
        "rejected_optimizer_mutation_count": value.rejected_optimizer_mutation_count,
        "minimum_complete_gradient_retention": value.minimum_complete_gradient_retention,
        "bootstrap_samples": value.bootstrap_samples,
        "bootstrap_seed": value.bootstrap_seed,
    }


def _evidence_payload(value: DynamicSetSplitEvidence) -> dict[str, Any]:
    raw_promotion = _raw_promotion_payload(value.promotion_evidence)
    return {
        "split": value.split,
        "protocol_sha256": value.protocol_sha256,
        "checkpoint_sha256": value.checkpoint_sha256,
        "model_state_sha256": value.model_state_sha256,
        "completed_updates": value.completed_updates,
        "physical_manifest_sha256": value.physical_manifest_sha256,
        "physical_row_count": value.physical_row_count,
        "planning_manifest_sha256": value.planning_manifest_sha256,
        "planning_task_count": value.planning_task_count,
        "physical_cells": [
            {"cell": asdict(entry.cell), "metrics": _physical_metrics_payload(entry.metrics)}
            for entry in value.physical_cells
        ],
        "planning_slices": [
            {
                **{
                    name: getattr(item, name)
                    for name in ("object_count", "candidate_count", "distribution")
                },
                **{
                    name: _supported_payload(getattr(item, name))
                    for name in (
                        "handle_resolution",
                        "oracle_winner_accuracy",
                        "normalized_regret_median",
                        "normalized_regret_p95",
                        "successful_oracle_goal_success",
                    )
                },
            }
            for item in value.planning_slices
        ],
        "planning_invariants": asdict(value.planning_invariants),
        "promotion_evidence": {
            "physical_pair_count": len(value.promotion_evidence.candidate_physical),
            "planning_pair_count": len(value.promotion_evidence.candidate_planning),
            "critical_comparison_count": len(value.promotion_evidence.critical_comparisons),
            "accepted_like_comparison_count": len(
                value.promotion_evidence.accepted_like_comparisons
            ),
            "raw_evidence_sha256": canonical_sha256(raw_promotion),
            "planning_pair_provenance_count": len(value.planning_pair_provenance),
            "planning_pair_provenance_sha256": canonical_sha256(
                [asdict(item) for item in value.planning_pair_provenance]
            ),
            "bootstrap_samples": value.promotion_evidence.bootstrap_samples,
            "bootstrap_seed": value.promotion_evidence.bootstrap_seed,
        },
        "promotion_integrity": asdict(value.promotion_integrity),
        "resources": asdict(value.resources),
        "fresh_resource_evidence": (
            None if value.fresh_resource_evidence is None else dict(value.fresh_resource_evidence)
        ),
        "score_components": dict(value.score_components),
        "training_support_passed": value.training_support_passed,
        "callback_binding_sha256": value.callback_binding_sha256,
    }


def validate_split_evidence(
    value: DynamicSetSplitEvidence,
    context: SplitExecutionContext,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Recompute every split gate and exact population/checkpoint binding."""

    if type(value) is not DynamicSetSplitEvidence:
        raise TypeError("evaluation callback must return exact DynamicSetSplitEvidence")
    if type(context) is not SplitExecutionContext:
        raise TypeError("split validation requires its exact execution context")
    if type(context.fresh_resources_required) is not bool:
        raise TypeError("fresh-resource requirement must be one exact bool")
    if context.fresh_resources_required:
        validated_sha256(context.resource_config_sha256, label="resource context config")
        validated_sha256(context.resource_source_sha256, label="resource context source")
    elif context.resource_config_sha256 is not None or context.resource_source_sha256 is not None:
        raise ValueError("optional resource context must not carry formal bindings")
    expected_scalars = {
        "split": context.split,
        "protocol_sha256": context.protocol_sha256,
        "checkpoint_sha256": context.checkpoint_sha256,
        "model_state_sha256": context.model_state_sha256,
        "completed_updates": context.completed_updates,
        "physical_manifest_sha256": context.physical_manifest_sha256,
        "physical_row_count": len(context.physical_rows),
        "planning_manifest_sha256": context.planning_manifest_sha256,
        "planning_task_count": len(context.planning_rows),
        "callback_binding_sha256": context.callback_binding_sha256,
    }
    if any(getattr(value, name) != expected for name, expected in expected_scalars.items()):
        raise ValueError("evaluation callback result differs from its invocation binding")
    if value.fresh_resource_evidence is None and context.fresh_resources_required:
        raise PermissionError("formal evaluation lacks fresh-worker resource evidence")
    if value.fresh_resource_evidence is not None:
        fresh = FreshWorkerResourceEvidence.from_mapping(
            _exact_dict(
                value.fresh_resource_evidence,
                label="fresh worker resource evidence",
            )
        )
        if (
            fresh.checkpoint_sha256 != context.checkpoint_sha256
            or fresh.model_state_sha256 != context.model_state_sha256
            or fresh.resource_metrics != value.resources
            or fresh.workload_sha256 != FRESH_RESOURCE_WORKLOAD_SHA256
            or (
                context.resource_config_sha256 is not None
                and fresh.config_sha256 != context.resource_config_sha256
            )
            or (
                context.resource_source_sha256 is not None
                and fresh.source_sha256 != context.resource_source_sha256
            )
        ):
            raise ValueError("fresh worker resource evidence differs from split/checkpoint")
    expected_physical_size = PHYSICAL_SPLIT_SIZES[context.split]
    expected_planning_size = PLANNING_SPLIT_SIZES[context.split]
    if (
        len(context.physical_rows) != expected_physical_size
        or [row.ordinal for row in context.physical_rows] != list(range(expected_physical_size))
        or any(row.split != context.split for row in context.physical_rows)
        or canonical_sha256([asdict(row) for row in context.physical_rows])
        != context.physical_manifest_sha256
        or context.physical_manifest_sha256 != FROZEN_PHYSICAL_MANIFEST_SHA256[context.split]
    ):
        raise ValueError("physical population order/hash/size differs from the source freeze")
    if (
        len(context.planning_rows) != expected_planning_size
        or [row.ordinal for row in context.planning_rows] != list(range(expected_planning_size))
        or any(row.split != context.split for row in context.planning_rows)
        or canonical_sha256([asdict(row) for row in context.planning_rows])
        != context.planning_manifest_sha256
        or context.planning_manifest_sha256 != FROZEN_PLANNING_MANIFEST_SHA256[context.split]
    ):
        raise ValueError("planning population order/hash/size differs from the source freeze")
    if type(value.training_support_passed) is not bool:
        raise TypeError("training_support_passed must be one exact bool")
    entries = tuple(value.physical_cells)
    actual_cells = tuple(entry.cell for entry in entries)
    if actual_cells != PHYSICAL_CELLS:
        raise ValueError("evaluation must contain each of the 22 physical cells in frozen order")
    by_cell = {entry.cell: entry.metrics for entry in entries}
    failures: list[str] = []
    for cell in PHYSICAL_CELLS:
        failures.extend(
            f"physical/N{cell.object_count}/contact={int(cell.contact)}/dynamic={int(cell.dynamic_membership)}/{failure}"
            for failure in physical_cell_gate_failures(cell, by_cell[cell])
        )
    raw_promotion = value.promotion_evidence
    if type(raw_promotion) is not DynamicSetPromotionEvidence:
        raise TypeError("promotion evidence must be exact DynamicSetPromotionEvidence")
    for name, aggregate_score in (
        ("candidate", raw_promotion.candidate_aggregate_score),
        ("baseline", raw_promotion.baseline_aggregate_score),
    ):
        if (
            isinstance(aggregate_score, bool)
            or not isinstance(aggregate_score, (int, float))
            or not math.isfinite(float(aggregate_score))
            or float(aggregate_score) < 0.0
        ):
            raise ValueError(f"paired {name} aggregate score must be finite and nonnegative")
    if raw_promotion.baseline_aggregate_score <= 0.0:
        raise ValueError("paired baseline aggregate score must be positive")
    expected_physical_keys = tuple(
        (
            row.split,
            row.ordinal,
            row.seed,
            row.cell_index,
            canonical_sha256(asdict(row)),
        )
        for row in context.physical_rows
    )
    if (
        len(raw_promotion.candidate_physical) != len(expected_physical_keys)
        or len(raw_promotion.baseline_physical) != len(expected_physical_keys)
        or any(
            type(record) is not DynamicSetExampleScoreEvidence
            for record in (
                *raw_promotion.candidate_physical,
                *raw_promotion.baseline_physical,
            )
        )
        or tuple(record.pair_key for record in raw_promotion.candidate_physical)
        != expected_physical_keys
        or tuple(record.pair_key for record in raw_promotion.baseline_physical)
        != expected_physical_keys
    ):
        raise ValueError("paired improvement evidence does not cover the exact physical row order")
    if (
        len(raw_promotion.candidate_planning) != len(context.planning_rows)
        or len(raw_promotion.baseline_planning) != len(context.planning_rows)
        or any(
            type(record) is not PlanningTaskScoreEvidence
            for record in (
                *raw_promotion.candidate_planning,
                *raw_promotion.baseline_planning,
            )
        )
        or tuple(record.row for record in raw_promotion.candidate_planning) != context.planning_rows
        or tuple(record.row for record in raw_promotion.baseline_planning) != context.planning_rows
    ):
        raise ValueError("paired improvement evidence does not cover the exact planning task order")
    candidate_planning_reduction = reduce_planning_task_outcomes(
        tuple(record.validate().outcome for record in raw_promotion.candidate_planning)
    )
    if tuple(value.planning_slices) != candidate_planning_reduction.slices:
        raise ValueError("planning slice diagnostics differ from raw task evidence")
    if (
        (
            value.planning_invariants.serial_vectorized_winner_parity
            and not candidate_planning_reduction.serial_vectorized_winner_parity
        )
        or (
            value.planning_invariants.maximum_cost_difference
            < candidate_planning_reduction.maximum_cost_difference
        )
        or (
            value.planning_invariants.action_target_isolation
            and not candidate_planning_reduction.active_set_frozen
        )
        or (
            value.planning_invariants.source_belief_unchanged
            and not candidate_planning_reduction.source_belief_unchanged
        )
    ):
        raise ValueError("planning invariants contradict raw task evidence")
    distribution = (
        "compositional_ood" if context.split == "compositional_ood" else "in_distribution"
    )
    failures.extend(
        f"planning/{failure}"
        for failure in planning_gate_failures(
            candidate_planning_reduction.gate_metrics,
            value.planning_invariants,
            expected_distributions=(distribution,),
        )
    )
    provenance = tuple(value.planning_pair_provenance)
    if provenance:
        if len(provenance) != len(context.planning_rows):
            raise ValueError("planning pair provenance does not cover the exact task order")
        for row, item, candidate_record, baseline_record in zip(
            context.planning_rows,
            provenance,
            raw_promotion.candidate_planning,
            raw_promotion.baseline_planning,
            strict=True,
        ):
            if type(item) is not PlanningPairProvenance:
                raise TypeError("planning pair provenance must use its exact evidence type")
            item.validate(row)
            for record in (candidate_record, baseline_record):
                evaluation = record.outcome.evaluation
                if (
                    evaluation is not None
                    and evaluation.private_oracle_sha256 != item.private_oracle_sha256
                ):
                    raise ValueError(
                        "serialized planning score differs from paired oracle provenance"
                    )
    expected_critical, expected_accepted_like = _physical_regression_comparisons(
        context.physical_rows,
        raw_promotion.candidate_physical,
        raw_promotion.baseline_physical,
    )
    if raw_promotion.critical_comparisons != expected_critical:
        raise ValueError("critical comparisons differ from the exact 22 raw cell populations")
    if raw_promotion.accepted_like_comparisons != expected_accepted_like:
        raise ValueError(
            "accepted-like comparisons differ from the frozen legacy-like raw populations"
        )
    rebuilt_integrity = _rebuild_promotion_integrity(raw_promotion)
    if rebuilt_integrity != value.promotion_integrity:
        raise ValueError("promotion integrity differs from rebuilt paired evidence")
    failures.extend(
        f"promotion/{failure}"
        for failure in promotion_gate_failures(value.promotion_integrity, value.resources)
    )
    checkpoint_score = DynamicSetCheckpointScore(
        completed_updates=value.completed_updates,
        model_state_sha256=value.model_state_sha256,
        components=value.score_components,
        training_support_passed=value.training_support_passed,
        selection_guardrails_passed=not selection_guardrail_failures(failures),
    ).validate()
    if checkpoint_score.score != raw_promotion.candidate_aggregate_score:
        raise ValueError("checkpoint score differs from raw paired candidate evidence")
    if not value.training_support_passed:
        failures.append("training_support_passed:false")
    payload = _evidence_payload(value)
    payload["selection_score"] = checkpoint_score.score
    payload["gate_failures"] = failures
    payload["passed"] = not failures
    payload["evidence_sha256"] = canonical_sha256(payload)
    return tuple(failures), payload


def _json_list(value: object, *, label: str, length: int | None = None) -> list[Any]:
    if type(value) is not list or (length is not None and len(value) != length):
        qualifier = "" if length is None else f" with length {length}"
        raise ValueError(f"{label} must be one exact JSON list{qualifier}")
    return value


def _exact_int(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _finite_float(value: object, *, label: str, minimum: float | None = None) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (minimum is not None and float(value) < minimum)
    ):
        qualifier = "finite" if minimum is None else f"finite and >= {minimum}"
        raise ValueError(f"{label} must be {qualifier}")
    return float(value)


def _supported_from_payload(value: object, *, label: str) -> SupportedScalar | None:
    if value is None:
        return None
    payload = _exact_dict(value, label=label)
    if set(payload) != {"value", "support"}:
        raise ValueError(f"{label} schema differs")
    return SupportedScalar(
        value=_finite_float(payload["value"], label=f"{label}.value"),
        support=_exact_int(payload["support"], label=f"{label}.support"),
    ).validate(name=label)


def _physical_metrics_from_payload(value: object) -> PhysicalCellMetrics:
    payload = _exact_dict(value, label="physical metrics")
    expected = {item.name for item in fields(PhysicalCellMetrics)}
    if set(payload) != expected:
        raise ValueError("physical metrics schema differs")
    horizons = _exact_dict(
        payload["horizon_position_rmse_m"],
        label="physical horizon metrics",
    )
    expected_horizons = {f"{float(horizon):.2f}" for horizon in HORIZON_POSITION_LIMITS_M}
    if set(horizons) != expected_horizons:
        raise ValueError("physical horizon metric schema differs")
    values = {
        name: _supported_from_payload(item, label=f"physical.{name}")
        for name, item in payload.items()
        if name != "horizon_position_rmse_m"
    }
    values["horizon_position_rmse_m"] = {
        horizon: _supported_from_payload(
            horizons[f"{float(horizon):.2f}"],
            label=f"physical.horizon_{horizon:.2f}",
        )
        for horizon in HORIZON_POSITION_LIMITS_M
    }
    return PhysicalCellMetrics(**values)


def _additive_snapshot_from_payload(value: object) -> DynamicSetAdditiveSnapshot:
    payload = _exact_dict(value, label="physical additive evidence")
    if set(payload) != {item.name for item in fields(DynamicSetAdditiveSnapshot)}:
        raise ValueError("physical additive evidence schema differs")

    def ints(name: str, length: int) -> tuple[int, ...]:
        return tuple(
            _exact_int(item, label=f"physical additive {name}")
            for item in _json_list(payload[name], label=name, length=length)
        )

    def total_count(name: str) -> tuple[float, int]:
        items = _json_list(payload[name], label=name, length=2)
        return (
            _finite_float(items[0], label=f"physical additive {name} total", minimum=0.0),
            _exact_int(items[1], label=f"physical additive {name} count"),
        )

    def horizons(name: str) -> tuple[tuple[float, float, int], ...]:
        items = _json_list(payload[name], label=name, length=len(HORIZON_POSITION_LIMITS_M))
        result = tuple(
            (
                _finite_float(row[0], label=f"{name} horizon", minimum=0.0),
                _finite_float(row[1], label=f"{name} squared error", minimum=0.0),
                _exact_int(row[2], label=f"{name} support"),
            )
            for row in (_json_list(item, label=f"{name} row", length=3) for item in items)
        )
        if tuple(item[0] for item in result) != tuple(HORIZON_POSITION_LIMITS_M):
            raise ValueError(f"{name} horizon order differs")
        return result

    return DynamicSetAdditiveSnapshot(
        episode_count=_exact_int(payload["episode_count"], label="episode_count"),
        proposal=ints("proposal", 3),
        exact_count=ints("exact_count", 2),
        current_position=total_count("current_position"),
        mature_velocity=total_count("mature_velocity"),
        post_event_velocity=total_count("post_event_velocity"),
        horizon_position=horizons("horizon_position"),
        horizon_velocity=horizons("horizon_velocity"),
        collision=ints("collision", 3),
        collision_timing=total_count("collision_timing"),
        persistent_identity=ints("persistent_identity", 2),
        identity_switch=ints("identity_switch", 2),
        birth=ints("birth", 3),
        birth_latency=total_count("birth_latency"),
        removal=ints("removal", 3),
        removal_latency=total_count("removal_latency"),
        uncertainty_90=ints("uncertainty_90", 2),
    )


def _public_boundary_from_payload(value: object) -> DynamicSetPublicBoundaryEvidence:
    payload = _exact_dict(value, label="physical public-boundary evidence")
    if set(payload) != {item.name for item in fields(DynamicSetPublicBoundaryEvidence)}:
        raise ValueError("physical public-boundary evidence schema differs")
    return DynamicSetPublicBoundaryEvidence(**payload).require_clean(require_truth_binding=True)


def _physical_score_record_from_payload(value: object) -> DynamicSetExampleScoreEvidence:
    payload = _exact_dict(value, label="physical row evidence")
    expected = {item.name for item in fields(DynamicSetExampleScoreEvidence)}
    if set(payload) != expected:
        raise ValueError("physical row evidence schema differs")
    supported = tuple(
        (
            item[0],
            _finite_float(item[1], label="physical supported component", minimum=0.0),
        )
        for item in (
            _json_list(entry, label="physical supported component", length=2)
            for entry in _json_list(
                payload["supported_components"],
                label="physical supported components",
            )
        )
    )
    if any(type(name) is not str or not name for name, _score in supported):
        raise ValueError("physical supported component name differs")
    return DynamicSetExampleScoreEvidence(
        split=payload["split"],
        ordinal=_exact_int(payload["ordinal"], label="physical ordinal"),
        seed=_exact_int(payload["seed"], label="physical seed"),
        cell_index=_exact_int(payload["cell_index"], label="physical cell index"),
        row_sha256=payload["row_sha256"],
        additive=_additive_snapshot_from_payload(payload["additive"]),
        supported_components=supported,
        public_boundary_evidence=_public_boundary_from_payload(payload["public_boundary_evidence"]),
        evidence_sha256=payload["evidence_sha256"],
    ).validate()


def _planning_row_from_payload(value: object) -> PlanningManifestRow:
    payload = _exact_dict(value, label="planning manifest row")
    if set(payload) != {item.name for item in fields(PlanningManifestRow)}:
        raise ValueError("planning manifest row schema differs")
    return PlanningManifestRow(**payload)


def _planning_score_record_from_payload(value: object) -> PlanningTaskScoreEvidence:
    payload = _exact_dict(value, label="planning task evidence")
    if set(payload) != {item.name for item in fields(PlanningTaskScoreEvidence)}:
        raise ValueError("planning task evidence schema differs")
    row = _planning_row_from_payload(payload["row"])
    outcome_payload = _exact_dict(payload["outcome"], label="planning outcome")
    if set(outcome_payload) != {item.name for item in fields(PlanningTaskOutcome)}:
        raise ValueError("planning outcome schema differs")
    outcome_row = _planning_row_from_payload(outcome_payload["row"])
    evaluation_payload = outcome_payload["evaluation"]
    evaluation = None
    if evaluation_payload is not None:
        evaluation_fields = _exact_dict(
            evaluation_payload,
            label="planning task evaluation",
        )
        if set(evaluation_fields) != {item.name for item in fields(PlanningTaskEvaluation)}:
            raise ValueError("planning task evaluation schema differs")
        evaluation_row = _planning_row_from_payload(evaluation_fields.pop("row"))
        evaluation = PlanningTaskEvaluation(row=evaluation_row, **evaluation_fields)
    outcome = PlanningTaskOutcome(
        row=outcome_row,
        handle_resolved=outcome_payload["handle_resolved"],
        evaluation=evaluation,
        failure_reason=outcome_payload["failure_reason"],
    )
    return PlanningTaskScoreEvidence(
        row=row,
        row_sha256=payload["row_sha256"],
        outcome=outcome,
        task_error=_finite_float(
            payload["task_error"],
            label="planning task error",
            minimum=0.0,
        ),
        evidence_sha256=payload["evidence_sha256"],
    ).validate()


def _promotion_from_payload(value: object) -> DynamicSetPromotionEvidence:
    payload = _exact_dict(value, label="raw promotion evidence")
    if set(payload) != {item.name for item in fields(DynamicSetPromotionEvidence)}:
        raise ValueError("raw promotion evidence schema differs")
    return DynamicSetPromotionEvidence(
        candidate_physical=tuple(
            _physical_score_record_from_payload(item)
            for item in _json_list(
                payload["candidate_physical"],
                label="candidate physical evidence",
            )
        ),
        baseline_physical=tuple(
            _physical_score_record_from_payload(item)
            for item in _json_list(
                payload["baseline_physical"],
                label="baseline physical evidence",
            )
        ),
        candidate_planning=tuple(
            _planning_score_record_from_payload(item)
            for item in _json_list(
                payload["candidate_planning"],
                label="candidate planning evidence",
            )
        ),
        baseline_planning=tuple(
            _planning_score_record_from_payload(item)
            for item in _json_list(
                payload["baseline_planning"],
                label="baseline planning evidence",
            )
        ),
        candidate_aggregate_score=_finite_float(
            payload["candidate_aggregate_score"],
            label="candidate aggregate score",
            minimum=0.0,
        ),
        baseline_aggregate_score=_finite_float(
            payload["baseline_aggregate_score"],
            label="baseline aggregate score",
            minimum=0.0,
        ),
        critical_comparisons=tuple(
            AggregateScoreComparison(**_exact_dict(item, label="critical comparison"))
            for item in _json_list(
                payload["critical_comparisons"],
                label="critical comparisons",
            )
        ),
        accepted_like_comparisons=tuple(
            AggregateScoreComparison(**_exact_dict(item, label="accepted-like comparison"))
            for item in _json_list(
                payload["accepted_like_comparisons"],
                label="accepted-like comparisons",
            )
        ),
        truth_leakage_count=_exact_int(payload["truth_leakage_count"], label="truth leakage count"),
        fabricated_target_count=_exact_int(
            payload["fabricated_target_count"], label="fabricated target count"
        ),
        nonfinite_state_count=_exact_int(
            payload["nonfinite_state_count"], label="nonfinite state count"
        ),
        rejected_optimizer_mutation_count=_exact_int(
            payload["rejected_optimizer_mutation_count"],
            label="rejected optimizer mutation count",
        ),
        minimum_complete_gradient_retention=_finite_float(
            payload["minimum_complete_gradient_retention"],
            label="minimum complete gradient retention",
            minimum=0.0,
        ),
        bootstrap_samples=_exact_int(
            payload["bootstrap_samples"],
            label="bootstrap samples",
            minimum=1_000,
        ),
        bootstrap_seed=_exact_int(payload["bootstrap_seed"], label="bootstrap seed"),
    )


def _planning_provenance_from_payload(
    value: object,
    rows: Sequence[PlanningManifestRow],
) -> tuple[PlanningPairProvenance, ...]:
    payload = _json_list(value, label="planning pair provenance")
    if not payload:
        return ()
    if len(payload) != len(rows):
        raise ValueError("planning pair provenance population differs")
    result: list[PlanningPairProvenance] = []
    for item, row in zip(payload, rows, strict=True):
        fields_payload = _exact_dict(item, label="planning pair provenance item")
        if set(fields_payload) != {entry.name for entry in fields(PlanningPairProvenance)}:
            raise ValueError("planning pair provenance schema differs")
        result.append(PlanningPairProvenance(**fields_payload).validate(row))
    return tuple(result)


def _split_evidence_from_scored_record(
    record: Mapping[str, Any],
    context: SplitExecutionContext,
) -> tuple[DynamicSetSplitEvidence, tuple[str, ...], dict[str, Any]]:
    value = _validate_record(record, schema=SCORED_EVIDENCE_SCHEMA)
    if set(value) != {
        "schema",
        "summary",
        "raw_promotion_evidence",
        "planning_pair_provenance",
        "record_sha256",
    }:
        raise ValueError("scored sufficient-statistics artifact schema differs")
    summary = _exact_dict(value["summary"], label="scored evidence summary")
    physical_entries: list[PhysicalCellEvaluation] = []
    for item in _json_list(summary.get("physical_cells"), label="physical cells"):
        entry = _exact_dict(item, label="physical cell entry")
        if set(entry) != {"cell", "metrics"}:
            raise ValueError("physical cell entry schema differs")
        cell_fields = _exact_dict(entry["cell"], label="physical cell")
        if set(cell_fields) != {item.name for item in fields(PhysicalCell)}:
            raise ValueError("physical cell schema differs")
        cell = PhysicalCell(**cell_fields).validate()
        physical_entries.append(
            PhysicalCellEvaluation(cell, _physical_metrics_from_payload(entry["metrics"]))
        )
    planning_slices: list[PlanningSliceMetrics] = []
    for item in _json_list(summary.get("planning_slices"), label="planning slices"):
        entry = _exact_dict(item, label="planning slice")
        if set(entry) != {item.name for item in fields(PlanningSliceMetrics)}:
            raise ValueError("planning slice schema differs")
        planning_slices.append(
            PlanningSliceMetrics(
                object_count=_exact_int(entry["object_count"], label="planning object count"),
                candidate_count=_exact_int(
                    entry["candidate_count"], label="planning candidate count"
                ),
                distribution=entry["distribution"],
                handle_resolution=_supported_from_payload(
                    entry["handle_resolution"], label="planning handle resolution"
                ),
                oracle_winner_accuracy=_supported_from_payload(
                    entry["oracle_winner_accuracy"], label="planning winner accuracy"
                ),
                normalized_regret_median=_supported_from_payload(
                    entry["normalized_regret_median"], label="planning median regret"
                ),
                normalized_regret_p95=_supported_from_payload(
                    entry["normalized_regret_p95"], label="planning p95 regret"
                ),
                successful_oracle_goal_success=_supported_from_payload(
                    entry["successful_oracle_goal_success"],
                    label="planning goal success",
                ),
            )
        )
    invariants_payload = _exact_dict(
        summary.get("planning_invariants"),
        label="planning invariants",
    )
    resources_payload = _exact_dict(summary.get("resources"), label="resource evidence")
    raw_fresh_resources = summary.get("fresh_resource_evidence")
    fresh_resources_payload = (
        None
        if raw_fresh_resources is None
        else _exact_dict(raw_fresh_resources, label="fresh worker resource evidence")
    )
    integrity_payload = _exact_dict(
        summary.get("promotion_integrity"),
        label="promotion integrity",
    )
    promotion = _promotion_from_payload(value["raw_promotion_evidence"])
    provenance = _planning_provenance_from_payload(
        value["planning_pair_provenance"],
        context.planning_rows,
    )
    split_evidence = DynamicSetSplitEvidence(
        split=summary.get("split"),
        protocol_sha256=summary.get("protocol_sha256"),
        checkpoint_sha256=summary.get("checkpoint_sha256"),
        model_state_sha256=summary.get("model_state_sha256"),
        completed_updates=summary.get("completed_updates"),
        physical_manifest_sha256=summary.get("physical_manifest_sha256"),
        physical_row_count=summary.get("physical_row_count"),
        planning_manifest_sha256=summary.get("planning_manifest_sha256"),
        planning_task_count=summary.get("planning_task_count"),
        physical_cells=tuple(physical_entries),
        planning_slices=tuple(planning_slices),
        planning_invariants=PlanningInvariantMetrics(**invariants_payload),
        promotion_evidence=promotion,
        promotion_integrity=PromotionIntegrityMetrics(**integrity_payload),
        resources=ResourceMetrics(**resources_payload),
        score_components=_exact_dict(
            summary.get("score_components"),
            label="score components",
        ),
        training_support_passed=summary.get("training_support_passed"),
        callback_binding_sha256=summary.get("callback_binding_sha256"),
        planning_pair_provenance=provenance,
        fresh_resource_evidence=fresh_resources_payload,
    )
    failures, rebuilt_summary = validate_split_evidence(split_evidence, context)
    if rebuilt_summary != summary:
        raise ValueError("scored evidence summary differs from independent recomputation")
    return split_evidence, failures, rebuilt_summary


def _scored_evidence_artifact_name(*, split: str, completed_updates: int) -> str:
    if split == "development":
        if completed_updates not in range(512, 32_768 + 1, 512):
            raise ValueError("development evidence update boundary differs")
        return f"development_scored_evidence_{completed_updates:06d}.json"
    if split not in PROTECTED_SPLIT_ORDER:
        raise ValueError("scored evidence split differs")
    return f"{split}_scored_evidence.json"


def _write_scored_evidence_artifact(
    artifacts: QualificationArtifactDirectory,
    *,
    evidence: DynamicSetSplitEvidence,
    evidence_payload: Mapping[str, Any],
) -> dict[str, Any]:
    expected = _evidence_payload(evidence)
    expected["selection_score"] = selection_score(evidence.score_components)
    expected["gate_failures"] = list(evidence_payload["gate_failures"])
    expected["passed"] = not expected["gate_failures"]
    expected["evidence_sha256"] = canonical_sha256(expected)
    if dict(evidence_payload) != expected:
        raise ValueError("validated evidence payload changed before persistence")
    record = _record_with_digest(
        {
            "schema": SCORED_EVIDENCE_SCHEMA,
            "summary": dict(evidence_payload),
            "raw_promotion_evidence": _raw_promotion_payload(evidence.promotion_evidence),
            "planning_pair_provenance": [
                asdict(item) for item in evidence.planning_pair_provenance
            ],
        }
    )
    contents = canonical_json_bytes(record) + b"\n"
    name = _scored_evidence_artifact_name(
        split=evidence.split,
        completed_updates=evidence.completed_updates,
    )
    artifact_sha256 = artifacts.write_fresh_bytes(name, contents)
    return {
        "scored_evidence_artifact_name": name,
        "scored_evidence_artifact_sha256": artifact_sha256,
        "scored_evidence_artifact_bytes": len(contents),
        "scored_evidence_record_sha256": record["record_sha256"],
    }


def _read_scored_evidence_artifact(
    artifacts: QualificationArtifactDirectory,
    binding: Mapping[str, Any],
    context: SplitExecutionContext,
    *,
    recompute: bool,
) -> tuple[dict[str, Any], tuple[str, ...] | None, dict[str, Any] | None]:
    expected_keys = {
        "scored_evidence_artifact_name",
        "scored_evidence_artifact_sha256",
        "scored_evidence_artifact_bytes",
        "scored_evidence_record_sha256",
    }
    if set(binding) != expected_keys:
        raise ValueError("scored evidence artifact binding schema differs")
    expected_name = _scored_evidence_artifact_name(
        split=context.split,
        completed_updates=context.completed_updates,
    )
    name = binding["scored_evidence_artifact_name"]
    if type(name) is not str or name != expected_name:
        raise ValueError("scored evidence artifact name differs")
    validated_sha256(
        binding["scored_evidence_artifact_sha256"],
        label="scored evidence artifact",
    )
    validated_sha256(
        binding["scored_evidence_record_sha256"],
        label="scored evidence record",
    )
    expected_bytes = _exact_int(
        binding["scored_evidence_artifact_bytes"],
        label="scored evidence artifact bytes",
        minimum=1,
    )
    contents = artifacts.read_bytes(name)
    if (
        len(contents) != expected_bytes
        or sha256_bytes(contents) != binding["scored_evidence_artifact_sha256"]
    ):
        raise OSError("scored evidence artifact bytes differ from their sealed binding")
    try:
        payload = json.loads(contents)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("scored evidence artifact is not strict JSON") from error
    record = _validate_record(payload, schema=SCORED_EVIDENCE_SCHEMA)
    if record["record_sha256"] != binding["scored_evidence_record_sha256"]:
        raise ValueError("scored evidence record differs from its sealed binding")
    if not recompute:
        return record, None, None
    _evidence, failures, summary = _split_evidence_from_scored_record(record, context)
    return record, failures, summary


def _active_permit(ledger: OrderedSplitLedger, *, split: str) -> SplitPermit:
    record = ledger.load()
    transitions = record["transitions"]
    active = next(
        (
            transition
            for transition in reversed(transitions)
            if transition["event"] in {"begin", "complete"}
        ),
        None,
    )
    if active is None or active["event"] != "begin":
        raise RuntimeError(f"{split!r} has not been begun")
    if active["split"] != split:
        raise RuntimeError(f"active ledger split is {active['split']!r}, not {split!r}")
    return SplitPermit(
        split=active["split"],
        index=active["index"],
        nonce=active["nonce"],
        protocol_sha256=ledger.protocol_sha256,
    )


class DynamicSetQualification:
    """Durable compact coordinator for development and protected qualification."""

    def __init__(self, artifacts: QualificationArtifactDirectory) -> None:
        self.artifacts = artifacts
        self.protocol = _validate_protocol(artifacts.read_json("protocol.json"))
        foundation = _validate_record(
            artifacts.read_json("known_action_foundation.json"),
            schema="dynamic_set_known_action_foundation_v1",
        )
        if (
            foundation.get("qualified") is not True
            or foundation.get("specification_version") != "1.60"
            or foundation["record_sha256"] != self.protocol["known_action_foundation_sha256"]
        ):
            raise PermissionError("known-action foundation differs from protocol")

    @classmethod
    def create_fresh(
        cls,
        root: str | Path,
        *,
        known_action_directory: str | Path,
        source: DynamicSetSourceFreeze,
        config_sha256: str,
        training_protocol_sha256: str | None = None,
    ) -> DynamicSetQualification:
        return cls._create_fresh_with_authority(
            root,
            known_action_directory=known_action_directory,
            source=source,
            config_sha256=config_sha256,
            training_protocol_sha256=training_protocol_sha256,
            test_only=False,
        )

    @classmethod
    def create_fresh_test_only(
        cls,
        root: str | Path,
        *,
        known_action_directory: str | Path,
        source: DynamicSetSourceFreeze,
        config_sha256: str,
        training_protocol_sha256: str | None = None,
    ) -> DynamicSetQualification:
        """Create a visibly non-formal callback harness for unit tests only."""

        return cls._create_fresh_with_authority(
            root,
            known_action_directory=known_action_directory,
            source=source,
            config_sha256=config_sha256,
            training_protocol_sha256=training_protocol_sha256,
            test_only=True,
        )

    @classmethod
    def _create_fresh_with_authority(
        cls,
        root: str | Path,
        *,
        known_action_directory: str | Path,
        source: DynamicSetSourceFreeze,
        config_sha256: str,
        training_protocol_sha256: str | None,
        test_only: bool,
    ) -> DynamicSetQualification:
        authenticate_dynamic_set_source_freeze(source)
        foundation = validate_known_action_foundation(known_action_directory)
        builder = (
            _build_test_dynamic_set_protocol_binding
            if test_only
            else build_dynamic_set_protocol_binding
        )
        protocol = builder(
            foundation=foundation,
            source=source,
            config_sha256=config_sha256,
            training_protocol_sha256=training_protocol_sha256,
        )
        authenticate_dynamic_set_source_freeze(source)
        artifacts = QualificationArtifactDirectory.create_fresh(
            Path(root).absolute(),
            allowed_names=ARTIFACT_NAMES,
            maximum_file_bytes=MAXIMUM_ARTIFACT_BYTES,
        )
        artifacts.write_fresh_json("known_action_foundation.json", foundation)
        artifacts.write_fresh_json("protocol.json", protocol)
        development = OrderedSplitLedger(
            artifacts,
            artifact_name="development_ledger.json",
            protocol_sha256=protocol["protocol_sha256"],
            split_order=("disposable_screen", "campaign"),
        )
        development.create_fresh()
        artifacts.write_fresh_json(
            "campaign_state.json",
            _record_with_digest(
                {
                    "schema": "dynamic_set_campaign_state_v1",
                    "protocol_sha256": protocol["protocol_sha256"],
                    "state": "awaiting_screen",
                    "screen_result_sha256": None,
                    "architecture_attempts": [],
                    "campaign_permit_sha256": None,
                    "development_cache_evidence": None,
                    "validation_candidates": [],
                    "incumbent": None,
                    "status": None,
                    "reason": None,
                    "elapsed_training_hours": 0.0,
                    "terminal_execution_receipt": None,
                }
            ),
        )
        return cls(artifacts)

    @classmethod
    def attach(cls, root: str | Path) -> DynamicSetQualification:
        artifacts = QualificationArtifactDirectory.attach(
            Path(root).absolute(),
            allowed_names=ARTIFACT_NAMES,
            maximum_file_bytes=MAXIMUM_ARTIFACT_BYTES,
        )
        return cls(artifacts)

    @property
    def protocol_sha256(self) -> str:
        return self.protocol["protocol_sha256"]

    def authenticate_current_source(self) -> DynamicSetSourceFreeze:
        source = DynamicSetSourceFreeze(**self.protocol["source"])
        return authenticate_dynamic_set_source_freeze(source)

    def screen_wall_seconds(self) -> float:
        """Return the sealed screen duration for the main-envelope handoff."""

        campaign = self._campaign()
        if campaign["state"] == "awaiting_screen":
            raise RuntimeError("screen wall timing is unavailable before screen completion")
        screen = _validate_record(
            self.artifacts.read_json("screen_result.json"),
            schema="dynamic_set_screen_result_v1",
        )
        value = screen.get("screen_wall_seconds")
        attempts = campaign["architecture_attempts"]
        if (
            screen["record_sha256"] != campaign["screen_result_sha256"]
            or type(value) is not float
            or not math.isfinite(value)
            or value < 0.0
            or not attempts
            or attempts[-1]["screen_result_sha256"] != screen["record_sha256"]
        ):
            raise ValueError("sealed screen wall timing differs from campaign state")
        total = attempts[-1]["end_cumulative_seconds"]
        if type(total) is not float or not math.isfinite(total) or total < value:
            raise ValueError("architecture-attempt cumulative screen timing differs")
        return value

    def architecture_execution_binding(self) -> dict[str, Any]:
        """Return the exact passed-attempt binding consumed by the executor."""

        self._require_evaluation_authority(_FORMAL_EVALUATION_AUTHORITY)
        return self._current_architecture_execution_binding()

    def _current_architecture_execution_binding(self) -> dict[str, Any]:
        campaign = self._campaign()
        attempts = campaign["architecture_attempts"]
        if not attempts or attempts[-1]["status"] != "passed":
            raise RuntimeError("campaign has no passed architecture attempt")
        return _architecture_execution_fields_from_attempt(attempts[-1])

    def architecture_resolved_config(self) -> OrpheusConfig:
        """Reconstruct the sole ledger-authorized model config without caller input."""

        binding = self.architecture_execution_binding()
        config = _resolved_architecture_config(binding["architecture_choice"])
        if canonical_sha256(config.to_dict()) != binding["resolved_config_sha256"]:
            raise ValueError("reconstructed architecture config differs from its ledger")
        return config

    def _development_ledger(self) -> OrderedSplitLedger:
        return OrderedSplitLedger(
            self.artifacts,
            artifact_name="development_ledger.json",
            protocol_sha256=self.protocol_sha256,
            split_order=("disposable_screen", "campaign"),
        )

    def _campaign(self) -> dict[str, Any]:
        value = _validate_record(
            self.artifacts.read_json("campaign_state.json"),
            schema="dynamic_set_campaign_state_v1",
        )
        if value["protocol_sha256"] != self.protocol_sha256:
            raise ValueError("campaign protocol binding differs")
        expected_keys = {
            "schema",
            "protocol_sha256",
            "state",
            "screen_result_sha256",
            "architecture_attempts",
            "campaign_permit_sha256",
            "development_cache_evidence",
            "validation_candidates",
            "incumbent",
            "status",
            "reason",
            "elapsed_training_hours",
            "terminal_execution_receipt",
            "record_sha256",
        }
        if set(value) != expected_keys:
            raise ValueError("campaign state schema differs")
        if value["state"] not in {"awaiting_screen", "screen_passed", "training", "terminal"}:
            raise ValueError("campaign state is invalid")
        if (
            isinstance(value["elapsed_training_hours"], bool)
            or not isinstance(value["elapsed_training_hours"], (int, float))
            or not math.isfinite(float(value["elapsed_training_hours"]))
            or value["elapsed_training_hours"] < 0.0
        ):
            raise ValueError("campaign elapsed training time is invalid")
        architecture_attempts = _validate_architecture_attempts(
            value["architecture_attempts"],
            protocol=self.protocol,
        )
        active_architecture = (
            None
            if not architecture_attempts
            else _architecture_execution_fields_from_attempt(architecture_attempts[-1])
        )
        candidates = value["validation_candidates"]
        if type(candidates) is not list:
            raise ValueError("campaign validation history must be one exact list")
        development_cache_evidence = value["development_cache_evidence"]
        if development_cache_evidence is not None:
            cache_keys = {
                "schema",
                "namespace_sha256",
                "physical_entry_count",
                "planning_entry_count",
                "physical_merkle_sha256",
                "planning_merkle_sha256",
                "population_merkle_sha256",
                "evidence_sha256",
            }
            if (
                type(development_cache_evidence) is not dict
                or set(development_cache_evidence) != cache_keys
                or development_cache_evidence["schema"]
                != "dynamic_set_development_evaluation_cache_evidence_v1"
                or development_cache_evidence["physical_entry_count"]
                != PHYSICAL_SPLIT_SIZES["development"]
                or development_cache_evidence["planning_entry_count"]
                != PLANNING_SPLIT_SIZES["development"]
            ):
                raise ValueError("campaign development cache evidence schema differs")
            for name in (
                "namespace_sha256",
                "physical_merkle_sha256",
                "planning_merkle_sha256",
                "population_merkle_sha256",
                "evidence_sha256",
            ):
                validated_sha256(
                    development_cache_evidence[name],
                    label=f"campaign development cache {name}",
                )
            if (
                self.protocol["evaluation_authority"] != _FORMAL_EVALUATION_AUTHORITY
                or active_architecture is None
            ):
                raise ValueError("development cache evidence lacks a formal architecture")
            cache_config = _resolved_architecture_config(active_architecture["architecture_choice"])
            if (
                canonical_sha256(cache_config.to_dict())
                != active_architecture["resolved_config_sha256"]
            ):
                raise ValueError("development cache architecture config differs")
            cache_binding = _development_evaluation_cache_binding(
                config=cache_config,
                protocol=self.protocol,
            )
            DynamicSetDevelopmentEvaluationCacheEvidence.from_mapping(
                development_cache_evidence,
                binding=cache_binding,
            ).validate(
                binding=cache_binding,
                physical_entry_count=PHYSICAL_SPLIT_SIZES["development"],
                planning_entry_count=PLANNING_SPLIT_SIZES["development"],
            )
        incumbent: dict[str, Any] | None = None
        incumbent_score: DynamicSetCheckpointScore | None = None
        formal_evaluation = self.protocol["evaluation_authority"] == _FORMAL_EVALUATION_AUTHORITY
        previous = "0" * 64
        candidate_keys = {
            "sequence",
            "completed_updates",
            "checkpoint_path",
            "checkpoint_sha256",
            "checkpoint_bytes",
            "model_state_sha256",
            "score_components",
            "selection_score",
            "training_support_passed",
            "selection_guardrail_failures",
            "selection_guardrails_passed",
            "gate_failures",
            "all_gates_passed",
            "evidence_sha256",
            "scored_evidence_artifact_name",
            "scored_evidence_artifact_sha256",
            "scored_evidence_artifact_bytes",
            "scored_evidence_record_sha256",
            "callback_binding_sha256",
            "development_cache_namespace_sha256",
            "development_cache_input_evidence_sha256",
            "development_cache_evidence_sha256",
            "development_cache_population_sha256",
            "development_cache_cold_materializations",
            "development_cache_recertified_materializations",
            "development_cache_warm_hits",
            "accepted",
            "selection_reason",
            "previous_sha256",
            "validation_wall_seconds",
            "validation_timing_count",
            "cumulative_validation_wall_seconds",
            "previous_validation_timing_sha256",
            "validation_timing_sha256",
            "candidate_sha256",
            *_EXECUTION_RECEIPT_EVIDENCE_KEYS,
        }
        previous_training_seconds = 0.0
        validation_durations: list[float] = []
        previous_validation_timing_sha256 = _ZERO_SHA256
        for index, candidate in enumerate(candidates):
            if type(candidate) is not dict or set(candidate) != candidate_keys:
                raise ValueError("campaign candidate schema differs")
            body = {key: item for key, item in candidate.items() if key != "candidate_sha256"}
            if (
                candidate["sequence"] != index
                or candidate["completed_updates"] != (index + 1) * 512
                or candidate["previous_sha256"] != previous
                or canonical_sha256(body) != candidate["candidate_sha256"]
            ):
                raise ValueError("campaign candidate transition chain differs")
            previous = validated_sha256(candidate["candidate_sha256"], label="campaign candidate")
            for name in (
                "checkpoint_sha256",
                "model_state_sha256",
                "evidence_sha256",
                "scored_evidence_artifact_sha256",
                "scored_evidence_record_sha256",
                "callback_binding_sha256",
                "development_cache_namespace_sha256",
                "development_cache_input_evidence_sha256",
                "development_cache_evidence_sha256",
                "development_cache_population_sha256",
                "execution_progress_sha256",
                "execution_progress_record_sha256",
                "execution_protocol_sha256",
                "execution_config_sha256",
                "execution_source_sha256",
                "execution_checkpoint_sha256",
                "execution_model_state_sha256",
                "timing_evidence_sha256",
                "execution_validation_timing_sha256",
                "execution_validated_candidate_sha256",
                "execution_validated_checkpoint_sha256",
                "execution_validated_model_state_sha256",
                "previous_validation_timing_sha256",
                "validation_timing_sha256",
            ):
                validated_sha256(candidate[name], label=f"candidate {name}")
            validation_wall_seconds = candidate["validation_wall_seconds"]
            if (
                type(validation_wall_seconds) is not float
                or not math.isfinite(validation_wall_seconds)
                or validation_wall_seconds < 0.0
            ):
                raise ValueError("campaign validation wall timing differs")
            receipt_validation_cumulative = float(math.fsum(validation_durations))
            if (
                candidate["execution_validation_timing_count"] != index
                or candidate["execution_cumulative_validation_seconds"]
                != receipt_validation_cumulative
                or candidate["execution_validation_timing_sha256"]
                != previous_validation_timing_sha256
            ):
                raise ValueError(
                    "candidate execution receipt does not precede its validation timing link"
                )
            validation_durations.append(validation_wall_seconds)
            cumulative_validation_wall_seconds = float(math.fsum(validation_durations))
            validation_timing_body = {
                "schema": VALIDATION_TIMING_EVIDENCE_SCHEMA,
                "sequence": index,
                "completed_updates": candidate["completed_updates"],
                "checkpoint_sha256": candidate["checkpoint_sha256"],
                "model_state_sha256": candidate["model_state_sha256"],
                "execution_progress_record_sha256": candidate["execution_progress_record_sha256"],
                "callback_binding_sha256": candidate["callback_binding_sha256"],
                "validation_wall_seconds": validation_wall_seconds,
                "validation_timing_count": candidate["validation_timing_count"],
                "cumulative_validation_wall_seconds": candidate[
                    "cumulative_validation_wall_seconds"
                ],
                "previous_validation_timing_sha256": candidate["previous_validation_timing_sha256"],
            }
            if (
                type(candidate["validation_timing_count"]) is not int
                or candidate["validation_timing_count"] != index + 1
                or type(candidate["cumulative_validation_wall_seconds"]) is not float
                or not math.isfinite(candidate["cumulative_validation_wall_seconds"])
                or candidate["cumulative_validation_wall_seconds"]
                != cumulative_validation_wall_seconds
                or candidate["previous_validation_timing_sha256"]
                != previous_validation_timing_sha256
                or candidate["validation_timing_sha256"] != canonical_sha256(validation_timing_body)
            ):
                raise ValueError("campaign validation timing chain differs")
            previous_validation_timing_sha256 = candidate["validation_timing_sha256"]
            expected_lineage = _expected_execution_lineage(candidates[:index])
            _validate_stored_execution_timing(
                candidate,
                minimum_cumulative_training_seconds=previous_training_seconds,
            )
            if (
                type(candidate["checkpoint_path"]) is not str
                or not candidate["checkpoint_path"]
                or not Path(candidate["checkpoint_path"]).is_absolute()
                or isinstance(candidate["checkpoint_bytes"], bool)
                or not isinstance(candidate["checkpoint_bytes"], int)
                or candidate["checkpoint_bytes"] <= 0
                or type(candidate["gate_failures"]) is not list
                or any(type(failure) is not str for failure in candidate["gate_failures"])
                or candidate["selection_guardrail_failures"]
                != list(selection_guardrail_failures(candidate["gate_failures"]))
                or type(candidate["selection_guardrails_passed"]) is not bool
                or candidate["selection_guardrails_passed"]
                is not (not candidate["selection_guardrail_failures"])
                or candidate["all_gates_passed"] is not (not candidate["gate_failures"])
                or candidate["scored_evidence_artifact_name"]
                != _scored_evidence_artifact_name(
                    split="development",
                    completed_updates=candidate["completed_updates"],
                )
                or isinstance(candidate["scored_evidence_artifact_bytes"], bool)
                or not isinstance(candidate["scored_evidence_artifact_bytes"], int)
                or not 0 < candidate["scored_evidence_artifact_bytes"] <= MAXIMUM_ARTIFACT_BYTES
                or type(candidate["accepted"]) is not bool
                or any(
                    type(candidate[name]) is not int or candidate[name] < 0
                    for name in (
                        "development_cache_cold_materializations",
                        "development_cache_recertified_materializations",
                        "development_cache_warm_hits",
                    )
                )
                or (
                    formal_evaluation
                    and (
                        development_cache_evidence is None
                        or candidate["development_cache_namespace_sha256"]
                        != development_cache_evidence["namespace_sha256"]
                        or candidate["development_cache_input_evidence_sha256"]
                        != (
                            _ZERO_SHA256
                            if index == 0
                            else development_cache_evidence["evidence_sha256"]
                        )
                        or candidate["development_cache_evidence_sha256"]
                        != development_cache_evidence["evidence_sha256"]
                        or candidate["development_cache_population_sha256"]
                        != development_cache_evidence["population_merkle_sha256"]
                        or candidate["development_cache_cold_materializations"]
                        + candidate["development_cache_recertified_materializations"]
                        + candidate["development_cache_warm_hits"]
                        != PHYSICAL_SPLIT_SIZES["development"] + PLANNING_SPLIT_SIZES["development"]
                    )
                )
                or (
                    not formal_evaluation
                    and (
                        development_cache_evidence is not None
                        or candidate["development_cache_namespace_sha256"] != _ZERO_SHA256
                        or candidate["development_cache_input_evidence_sha256"] != _ZERO_SHA256
                        or candidate["development_cache_evidence_sha256"] != _ZERO_SHA256
                        or candidate["development_cache_population_sha256"] != _ZERO_SHA256
                        or candidate["development_cache_cold_materializations"] != 0
                        or candidate["development_cache_recertified_materializations"] != 0
                        or candidate["development_cache_warm_hits"] != 0
                    )
                )
                or type(candidate["execution_progress_path"]) is not str
                or not Path(candidate["execution_progress_path"]).is_absolute()
                or Path(candidate["execution_progress_path"]).name != "progress.json"
                or isinstance(candidate["execution_progress_bytes"], bool)
                or not isinstance(candidate["execution_progress_bytes"], int)
                or not 0 < candidate["execution_progress_bytes"] <= MAXIMUM_ARTIFACT_BYTES
                or candidate["execution_active_resume_name"] not in {"resume_a.pt", "resume_b.pt"}
                or candidate["execution_protocol_sha256"] != self.protocol_sha256
                or candidate["execution_config_sha256"] != self.protocol["config_sha256"]
                or candidate["execution_source_sha256"] != canonical_sha256(self.protocol["source"])
                or candidate["execution_schedule_seed"] != EXECUTION_SCHEDULE_SEED
                or active_architecture is None
                or any(
                    candidate[name] != expected for name, expected in active_architecture.items()
                )
                or candidate["execution_completed_updates"] != candidate["completed_updates"]
                or candidate["execution_checkpoint_sha256"] != candidate["checkpoint_sha256"]
                or candidate["execution_model_state_sha256"] != candidate["model_state_sha256"]
                or isinstance(candidate["cumulative_training_seconds"], bool)
                or not isinstance(candidate["cumulative_training_seconds"], (int, float))
                or not math.isfinite(float(candidate["cumulative_training_seconds"]))
                or float(candidate["cumulative_training_seconds"]) < previous_training_seconds
                or isinstance(candidate["rejected_update_count"], bool)
                or not isinstance(candidate["rejected_update_count"], int)
                or candidate["rejected_update_count"] < 0
                or candidate["training_limit_reached"] is not False
                or (
                    candidate["execution_validated_candidate_count"],
                    candidate["execution_validated_boundary_updates"],
                    candidate["execution_validated_candidate_sha256"],
                    candidate["execution_validated_checkpoint_sha256"],
                    candidate["execution_validated_model_state_sha256"],
                )
                != expected_lineage
            ):
                raise ValueError("campaign candidate evidence fields differ")
            previous_training_seconds = float(candidate["cumulative_training_seconds"])
            score = DynamicSetCheckpointScore(
                completed_updates=candidate["completed_updates"],
                model_state_sha256=candidate["model_state_sha256"],
                components=candidate["score_components"],
                training_support_passed=candidate["training_support_passed"],
                selection_guardrails_passed=candidate["selection_guardrails_passed"],
            ).validate()
            if candidate["selection_score"] != score.score:
                raise ValueError("campaign candidate score differs from its frozen schema")
            decision = select_dynamic_set_incumbent(incumbent_score, score)
            if (
                candidate["accepted"] != decision.accepted
                or candidate["selection_reason"] != decision.reason
            ):
                raise ValueError("campaign candidate selection differs from strict replay")
            if decision.accepted:
                incumbent_score = score
                incumbent = candidate
        if (development_cache_evidence is not None and not candidates) or (
            development_cache_evidence is not None and not formal_evaluation
        ):
            raise ValueError("campaign development cache evidence has no formal candidate")
        if value["incumbent"] != incumbent:
            raise ValueError("campaign incumbent differs from strict lower-score replay")
        terminal_receipt = value["terminal_execution_receipt"]
        if terminal_receipt is not None:
            if type(terminal_receipt) is not dict or set(terminal_receipt) != set(
                _EXECUTION_RECEIPT_EVIDENCE_KEYS
            ):
                raise ValueError("campaign terminal execution receipt schema differs")
            for name in (
                "execution_progress_sha256",
                "execution_progress_record_sha256",
                "execution_protocol_sha256",
                "execution_config_sha256",
                "execution_source_sha256",
                "execution_checkpoint_sha256",
                "execution_model_state_sha256",
                "timing_evidence_sha256",
                "execution_validation_timing_sha256",
                "execution_validated_candidate_sha256",
                "execution_validated_checkpoint_sha256",
                "execution_validated_model_state_sha256",
            ):
                validated_sha256(terminal_receipt[name], label=f"terminal {name}")
            _validate_stored_execution_timing(
                terminal_receipt,
                minimum_cumulative_training_seconds=0.0,
            )
            if (
                type(terminal_receipt["execution_progress_path"]) is not str
                or not Path(terminal_receipt["execution_progress_path"]).is_absolute()
                or Path(terminal_receipt["execution_progress_path"]).name != "progress.json"
                or isinstance(terminal_receipt["execution_progress_bytes"], bool)
                or not isinstance(terminal_receipt["execution_progress_bytes"], int)
                or not 0 < terminal_receipt["execution_progress_bytes"] <= MAXIMUM_ARTIFACT_BYTES
                or terminal_receipt["execution_active_resume_name"]
                not in {"resume_a.pt", "resume_b.pt"}
                or terminal_receipt["execution_protocol_sha256"] != self.protocol_sha256
                or terminal_receipt["execution_config_sha256"] != self.protocol["config_sha256"]
                or terminal_receipt["execution_source_sha256"]
                != canonical_sha256(self.protocol["source"])
                or terminal_receipt["execution_schedule_seed"] != EXECUTION_SCHEDULE_SEED
                or active_architecture is None
                or any(
                    terminal_receipt[name] != expected
                    for name, expected in active_architecture.items()
                )
                or isinstance(terminal_receipt["execution_completed_updates"], bool)
                or not isinstance(terminal_receipt["execution_completed_updates"], int)
                or terminal_receipt["execution_completed_updates"] < 0
                or isinstance(terminal_receipt["cumulative_training_seconds"], bool)
                or not isinstance(terminal_receipt["cumulative_training_seconds"], (int, float))
                or not math.isfinite(float(terminal_receipt["cumulative_training_seconds"]))
                or terminal_receipt["cumulative_training_seconds"] < 0.0
                or isinstance(terminal_receipt["rejected_update_count"], bool)
                or not isinstance(terminal_receipt["rejected_update_count"], int)
                or terminal_receipt["rejected_update_count"] < 0
                or type(terminal_receipt["training_limit_reached"]) is not bool
            ):
                raise ValueError("campaign terminal execution receipt evidence differs")
            expected_terminal_lineage = _expected_execution_lineage(candidates)
            terminal_lineage = (
                terminal_receipt["execution_validated_candidate_count"],
                terminal_receipt["execution_validated_boundary_updates"],
                terminal_receipt["execution_validated_candidate_sha256"],
                terminal_receipt["execution_validated_checkpoint_sha256"],
                terminal_receipt["execution_validated_model_state_sha256"],
            )
            if (
                terminal_lineage != expected_terminal_lineage
                or terminal_receipt["execution_validation_timing_count"] != len(candidates)
                or terminal_receipt["execution_cumulative_validation_seconds"]
                != float(math.fsum(validation_durations))
                or terminal_receipt["execution_validation_timing_sha256"]
                != previous_validation_timing_sha256
            ):
                raise ValueError("campaign terminal execution ancestry differs")
            lineage_boundary = expected_terminal_lineage[1]
            if (
                not lineage_boundary
                <= terminal_receipt["execution_completed_updates"]
                <= min(
                    lineage_boundary + DEFAULT_CAMPAIGN.validation_interval_updates,
                    DEFAULT_CAMPAIGN.maximum_updates,
                )
            ):
                raise ValueError("campaign terminal execution cursor differs")
            if candidates and terminal_receipt["cumulative_training_seconds"] < float(
                candidates[-1]["cumulative_training_seconds"]
            ):
                raise ValueError("campaign terminal training time moved backwards")
        screen_present = "screen_result.json" in self.artifacts.inventory()
        if value["state"] == "awaiting_screen":
            if (
                screen_present
                or value["screen_result_sha256"] is not None
                or architecture_attempts
                or candidates
                or terminal_receipt is not None
            ):
                raise ValueError("unopened campaign already contains screen/training evidence")
        else:
            if not architecture_attempts:
                raise ValueError("campaign lacks its architecture attempt")
            screen, _screen_metrics, _screen_diagnostics, _screen_cache = _validated_screen_record(
                self.artifacts.read_json("screen_result.json"),
                protocol=self.protocol,
                resolved_config_sha256=architecture_attempts[-1]["resolved_config_sha256"],
            )
            if screen["record_sha256"] != value["screen_result_sha256"]:
                raise ValueError("campaign screen binding differs")
            if (
                not architecture_attempts
                or architecture_attempts[-1]["screen_result_sha256"] != screen["record_sha256"]
                or (
                    value["state"] in {"screen_passed", "training"}
                    and architecture_attempts[-1]["status"] != "passed"
                )
            ):
                raise ValueError("campaign architecture attempt differs from its screen")
            first_attempt_name = "screen_attempt_1_result.json"
            if first_attempt_name in self.artifacts.inventory():
                raise ValueError("campaign contains an unsupported second-attempt screen artifact")
            if (
                architecture_attempts[0]["start_cumulative_seconds"] != 0.0
                or architecture_attempts[0]["end_cumulative_seconds"]
                != screen["screen_wall_seconds"]
                or architecture_attempts[0]["diagnosis_sha256"]
                != _screen_record_diagnosis_sha256(screen)
            ):
                raise ValueError("single architecture attempt timing/diagnosis differs")
            expected_final_status = "passed" if screen["passed"] else "failed_terminal"
            if architecture_attempts[-1]["status"] != expected_final_status:
                raise ValueError("final architecture attempt status differs from its screen")
            cache_payload = screen["cache_evidence"]
            if cache_payload is not None:
                if type(cache_payload) is not dict:
                    raise ValueError("campaign screen cache evidence differs")
                ScreenCacheEvidence(**cache_payload).validate()
            diagnostics_payload = screen["architecture_diagnostics"]
            if diagnostics_payload is not None:
                derived_names = {
                    "perception_gates_passed",
                    "oracle_state_dynamics_passed",
                    "truth_state_contact_rollout_owns_error",
                }
                if type(diagnostics_payload) is not dict or not derived_names < set(
                    diagnostics_payload
                ):
                    raise ValueError("campaign screen architecture diagnostics differ")
                raw_diagnostics = {
                    name: item
                    for name, item in diagnostics_payload.items()
                    if name not in derived_names
                }
                diagnostics = ScreenArchitectureDiagnosticEvidence(**raw_diagnostics).validate(
                    protocol_sha256=self.protocol_sha256,
                    source_sha256=self.protocol["source"]["source_sha256"],
                    resolved_config_sha256=architecture_attempts[-1]["resolved_config_sha256"],
                )
                if any(
                    diagnostics_payload[name] != getattr(diagnostics, name)
                    for name in derived_names
                ):
                    raise ValueError("campaign screen derived diagnostics differ")
            if (
                self.protocol["evaluation_authority"] == _FORMAL_EVALUATION_AUTHORITY
                and screen["passed"]
                and (cache_payload is None or diagnostics_payload is None)
            ):
                raise ValueError("formal passed screen lacks repository diagnostic evidence")
            if any(
                candidate["screen_wall_seconds"] != screen["screen_wall_seconds"]
                for candidate in candidates
            ) or (
                terminal_receipt is not None
                and terminal_receipt["screen_wall_seconds"] != screen["screen_wall_seconds"]
            ):
                raise ValueError("campaign execution screen timing differs from its sealed screen")
            if value["state"] in {"screen_passed", "training"} and screen["passed"] is not True:
                raise ValueError("active campaign lacks a passed disposable screen")
        if value["state"] == "screen_passed":
            if (
                value["campaign_permit_sha256"] is not None
                or candidates
                or value["status"] is not None
                or terminal_receipt is not None
            ):
                raise ValueError("screen-passed state already contains campaign results")
        elif value["state"] == "training":
            validated_sha256(value["campaign_permit_sha256"], label="campaign permit")
            if (
                value["status"] is not None
                or value["reason"] is not None
                or terminal_receipt is not None
            ):
                raise ValueError("active training campaign already has a terminal result")
        elif value["state"] == "terminal" and (
            value["status"] not in TERMINAL_STATUSES or type(value["reason"]) is not str
        ):
            raise ValueError("terminal campaign lacks one frozen status and reason")
        if (
            value["state"] == "terminal"
            and value["campaign_permit_sha256"] is not None
            and terminal_receipt is None
        ):
            raise ValueError("terminal training campaign lacks an execution receipt")
        if terminal_receipt is not None and not math.isclose(
            float(value["elapsed_training_hours"]),
            float(terminal_receipt["cumulative_training_seconds"]) / 3600.0,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise ValueError("campaign elapsed time differs from its execution receipt")
        if (
            value["state"] == "terminal"
            and value["campaign_permit_sha256"] is not None
            and value["status"] == "limit_hit"
            and incumbent is None
        ):
            if terminal_receipt is None:
                raise ValueError("no-incumbent limit hit lacks terminal execution evidence")
            expected_reason = _no_incumbent_limit_hit_reason(
                completed_updates=terminal_receipt["execution_completed_updates"],
                training_limit_reached=terminal_receipt["training_limit_reached"],
                limit_hit_reason=terminal_receipt["limit_hit_reason"],
            )
            if value["reason"] != expected_reason:
                raise ValueError("no-incumbent limit-hit reason differs from strict replay")
        return value

    def _development_report(self) -> dict[str, Any]:
        report = _validate_record(
            self.artifacts.read_json("development_report.json"),
            schema="dynamic_set_development_report_v1",
        )
        expected_keys = {
            "schema",
            "specification_version",
            "protocol_sha256",
            "evaluation_authority",
            "formal",
            "status",
            "passed",
            "reason",
            "screen_result_sha256",
            "campaign_state_sha256",
            "development_ledger_sha256",
            "selected_candidate_sha256",
            "selected_evidence_sha256",
            "selected_scored_evidence",
            "development_cache_evidence",
            "checkpoint",
            "physical_manifest_sha256",
            "physical_row_count",
            "planning_manifest_sha256",
            "planning_task_count",
            "record_sha256",
        }
        campaign = self._campaign()
        if (
            set(report) != expected_keys
            or report["specification_version"] != SPECIFICATION_VERSION
            or report["protocol_sha256"] != self.protocol_sha256
            or report["evaluation_authority"] != self.protocol["evaluation_authority"]
            or report["formal"]
            is not (self.protocol["evaluation_authority"] == _FORMAL_EVALUATION_AUTHORITY)
            or report["status"] not in TERMINAL_STATUSES
            or report["passed"] is not (report["status"] == "qualified_convergence")
            or report["status"] != campaign["status"]
            or report["campaign_state_sha256"] != campaign["record_sha256"]
            or report["screen_result_sha256"] != campaign["screen_result_sha256"]
            or report["development_ledger_sha256"]
            != sha256_bytes(self.artifacts.read_bytes("development_ledger.json"))
            or report["physical_manifest_sha256"] != FROZEN_PHYSICAL_MANIFEST_SHA256["development"]
            or report["physical_row_count"] != PHYSICAL_SPLIT_SIZES["development"]
            or report["planning_manifest_sha256"] != FROZEN_PLANNING_MANIFEST_SHA256["development"]
            or report["planning_task_count"] != PLANNING_SPLIT_SIZES["development"]
            or report["development_cache_evidence"] != campaign["development_cache_evidence"]
        ):
            raise ValueError("development report binding differs")
        incumbent = campaign["incumbent"]
        if report["status"] == "qualified_convergence":
            checkpoint = report["checkpoint"]
            if (
                incumbent is None
                or type(checkpoint) is not dict
                or checkpoint.get("artifact_name") != "development_model.pt"
                or checkpoint.get("sha256")
                != sha256_bytes(self.artifacts.read_bytes("development_model.pt"))
                or checkpoint.get("model_state_sha256") != incumbent["model_state_sha256"]
                or checkpoint.get("completed_updates") != incumbent["completed_updates"]
                or report["selected_candidate_sha256"] != incumbent["candidate_sha256"]
                or report["selected_evidence_sha256"] != incumbent["evidence_sha256"]
                or report["selected_scored_evidence"] != self._scored_evidence_binding(incumbent)
            ):
                raise ValueError("development checkpoint/incumbent binding differs")
        elif report["checkpoint"] is not None:
            raise ValueError("unqualified development report cannot carry a checkpoint")
        return report

    def _replace_campaign(self, body: Mapping[str, Any]) -> dict[str, Any]:
        record = _record_with_digest(body)
        self.artifacts.replace_json("campaign_state.json", record)
        return record

    def _require_evaluation_authority(self, expected: str) -> None:
        if self.protocol["evaluation_authority"] != expected:
            raise PermissionError(f"qualification evaluation authority is not {expected!r}")

    def _split_resource_context_fields(self) -> dict[str, Any]:
        """Bind formal split evidence to the active source/config/workload."""

        if self.protocol["evaluation_authority"] != _FORMAL_EVALUATION_AUTHORITY:
            return {
                "fresh_resources_required": False,
                "resource_config_sha256": None,
                "resource_source_sha256": None,
            }
        return {
            "fresh_resources_required": True,
            "resource_config_sha256": canonical_sha256(
                self.architecture_resolved_config().to_dict()
            ),
            "resource_source_sha256": canonical_sha256(self.protocol["source"]),
        }

    def _formal_repository_config(self, path: str | Path) -> OrpheusConfig:
        self._require_evaluation_authority(_FORMAL_EVALUATION_AUTHORITY)
        resolved = Path(path).absolute()
        contents = _stable_regular_bytes(resolved)
        if sha256_bytes(contents) != self.protocol["config_sha256"]:
            raise ValueError("repository config bytes differ from the formal protocol")
        config = load_config(resolved)
        config = _validated_repository_config(config)
        if canonical_sha256(config.to_dict()) != self.protocol["base_config_payload_sha256"]:
            raise ValueError("resolved repository config differs from the formal profile")
        return config

    def execute_screen(
        self,
        *,
        config_path: str | Path,
        work_directory: str | Path,
        updates: int = 512,
    ) -> dict[str, Any]:
        """Run the sole formal screen through the source-bound repository evaluator."""

        config = self._formal_repository_config(config_path)

        def repository_evaluator(context: ScreenExecutionContext) -> ScreenExecutionResult:
            return evaluate_repository_disposable_screen(
                context,
                resolved_config=config,
                qualification_protocol=self.protocol,
                work_directory=work_directory,
                qualification_root=self.artifacts.root,
                updates=updates,
            )

        return self._execute_screen_with_evaluator(
            repository_evaluator,
            formal_base_config=config,
        )

    def execute_screen_test_only(
        self,
        callback: Callable[[ScreenExecutionContext], ScreenExecutionResult],
    ) -> dict[str, Any]:
        """Run an injected screen only inside an explicitly non-formal harness."""

        self._require_evaluation_authority(_TEST_EVALUATION_AUTHORITY)
        return self._execute_screen_with_evaluator(callback)

    def _execute_screen_with_evaluator(
        self,
        callback: Callable[[ScreenExecutionContext], ScreenExecutionResult],
        *,
        formal_base_config: OrpheusConfig | None = None,
    ) -> dict[str, Any]:
        if not callable(callback):
            raise TypeError("screen callback must be callable")
        if formal_base_config is not None:
            if self.protocol["evaluation_authority"] != _FORMAL_EVALUATION_AUTHORITY:
                raise PermissionError("only formal evaluation may bind a screen base config")
            _validated_repository_config(formal_base_config)
            if (
                canonical_sha256(formal_base_config.to_dict())
                != self.protocol["base_config_payload_sha256"]
            ):
                raise ValueError("screen base config differs from the formal protocol")
        self.authenticate_current_source()
        campaign = self._campaign()
        if campaign["state"] != "awaiting_screen":
            raise RuntimeError("disposable screen was already consumed")
        ledger = self._development_ledger()
        permit = ledger.begin("disposable_screen")
        context = ScreenExecutionContext(
            permit=permit,
            protocol_sha256=self.protocol_sha256,
            rows=SCREEN_ROWS,
            manifest_sha256=SCREEN_MANIFEST_SHA256,
        )

        def invoke(
            evaluator: Callable[[ScreenExecutionContext], ScreenExecutionResult],
            *,
            expected_config_sha256: str,
        ) -> tuple[dict[str, Any], ScreenExecutionResult | None, BaseException | None]:
            screen_started = time.monotonic()
            try:
                result = evaluator(context)
                self.authenticate_current_source()
                if type(result) is not ScreenExecutionResult:
                    raise TypeError("screen callback must return exact ScreenExecutionResult")
                if result.callback_binding_sha256 != context.callback_binding_sha256:
                    raise ValueError("screen callback result differs from its invocation binding")
                failures = list(disposable_screen_failures(result.metrics))
                diagnostics_payload: dict[str, Any] | None = None
                cache_payload: dict[str, Any] | None = None
                if result.cache_evidence is not None:
                    result.cache_evidence.validate()
                    cache_payload = asdict(result.cache_evidence)
                if result.architecture_diagnostics is not None:
                    result.architecture_diagnostics.validate(
                        protocol_sha256=self.protocol_sha256,
                        source_sha256=self.protocol["source"]["source_sha256"],
                        resolved_config_sha256=expected_config_sha256,
                    )
                    diagnostics_payload = {
                        **asdict(result.architecture_diagnostics),
                        "perception_gates_passed": (
                            result.architecture_diagnostics.perception_gates_passed
                        ),
                        "oracle_state_dynamics_passed": (
                            result.architecture_diagnostics.oracle_state_dynamics_passed
                        ),
                        "truth_state_contact_rollout_owns_error": (
                            result.architecture_diagnostics.truth_state_contact_rollout_owns_error
                        ),
                    }
                if self.protocol["evaluation_authority"] == _FORMAL_EVALUATION_AUTHORITY:
                    if diagnostics_payload is None:
                        failures.append("architecture_diagnostics:unavailable")
                    if cache_payload is None:
                        failures.append("screen_cache_evidence:unavailable")
                self.authenticate_current_source()
                screen_wall_seconds = float(time.monotonic() - screen_started)
                if not math.isfinite(screen_wall_seconds) or screen_wall_seconds < 0.0:
                    raise ValueError("screen wall timing is invalid")
                body = {
                    "schema": "dynamic_set_screen_result_v1",
                    "protocol_sha256": self.protocol_sha256,
                    "permit": asdict(permit),
                    "callback_binding_sha256": result.callback_binding_sha256,
                    "manifest_sha256": SCREEN_MANIFEST_SHA256,
                    "screen_wall_seconds": screen_wall_seconds,
                    "architecture_diagnostics": diagnostics_payload,
                    "cache_evidence": cache_payload,
                    "metrics": asdict(result.metrics),
                    "failures": failures,
                    "passed": not failures,
                }
                return _record_with_digest(body), result, None
            except BaseException as error:
                screen_wall_seconds = float(time.monotonic() - screen_started)
                if not math.isfinite(screen_wall_seconds) or screen_wall_seconds < 0.0:
                    screen_wall_seconds = 0.0
                body = {
                    "schema": "dynamic_set_screen_result_v1",
                    "protocol_sha256": self.protocol_sha256,
                    "permit": asdict(permit),
                    "callback_binding_sha256": context.callback_binding_sha256,
                    "manifest_sha256": SCREEN_MANIFEST_SHA256,
                    "screen_wall_seconds": screen_wall_seconds,
                    "architecture_diagnostics": None,
                    "cache_evidence": None,
                    "metrics": None,
                    "failures": [f"callback_error:{type(error).__name__}:{error}"],
                    "passed": False,
                }
                return _record_with_digest(body), None, error

        first_record, first_result, first_error = invoke(
            callback,
            expected_config_sha256=self.protocol["base_config_payload_sha256"],
        )
        first_attempt = _architecture_attempt_record(
            protocol=self.protocol,
            index=1,
            choice="base",
            resolved_config_sha256=self.protocol["base_config_payload_sha256"],
            screen_result=first_record,
            start_cumulative_seconds=0.0,
        )
        attempts = [first_attempt]

        if (
            first_error is None
            and first_record["passed"] is False
            and first_result is not None
            and first_result.architecture_diagnostics is not None
            and formal_base_config is not None
            and first_result.metrics.finite_owner_gradients
            and first_result.metrics.rejected_update_count == 0
            and 0 < first_result.metrics.completed_updates <= 512
        ):
            choice, configured = configure_second_attempt(
                formal_base_config,
                first_attempt_failed_early=True,
                oracle_state_dynamics_passed=(
                    first_result.architecture_diagnostics.oracle_state_dynamics_passed
                ),
                perception_gates_passed=(
                    first_result.architecture_diagnostics.perception_gates_passed
                ),
                truth_state_contact_owns_error=(
                    first_result.architecture_diagnostics.truth_state_contact_rollout_owns_error
                ),
                admission_evidence=None,
                admission_evidence_sha256=None,
            )
            if choice != "none" or configured is not None:
                raise RuntimeError(
                    "second architecture attempt was authorized without sealed admission evidence"
                )

        self.artifacts.write_fresh_json("screen_result.json", first_record)
        ledger.complete(
            permit,
            status="passed" if first_record["passed"] else "failed",
            result_sha256=first_record["record_sha256"],
        )
        if first_record["passed"]:
            self.authenticate_current_source()
            self._replace_campaign(
                {
                    **{key: item for key, item in campaign.items() if key != "record_sha256"},
                    "state": "screen_passed",
                    "screen_result_sha256": first_record["record_sha256"],
                    "architecture_attempts": attempts,
                }
            )
        else:
            self._replace_campaign(
                {
                    **{key: item for key, item in campaign.items() if key != "record_sha256"},
                    "state": "terminal",
                    "screen_result_sha256": first_record["record_sha256"],
                    "architecture_attempts": attempts,
                    "status": "failed_to_improve",
                    "reason": (
                        "disposable screen callback failed"
                        if first_error is not None
                        else "disposable screen failed its frozen acceptance gate"
                    ),
                }
            )
            self._publish_development_report()
        if first_error is not None:
            raise first_error
        return first_record

    def begin_campaign(self) -> SplitPermit:
        self.authenticate_current_source()
        campaign = self._campaign()
        if campaign["state"] != "screen_passed":
            raise RuntimeError("campaign requires one passed disposable screen")
        ledger = self._development_ledger()
        permit = ledger.begin("campaign")
        updated = {
            **{key: item for key, item in campaign.items() if key != "record_sha256"},
            "state": "training",
            "campaign_permit_sha256": canonical_sha256(asdict(permit)),
        }
        self._replace_campaign(updated)
        return permit

    def execute_validation_candidate(
        self,
        *,
        completed_updates: int,
        checkpoint_path: str | Path,
        execution_progress_path: str | Path,
        model_state_sha256: str,
        config_path: str | Path,
    ) -> dict[str, Any]:
        """Evaluate one formal candidate with the exact repository evaluator."""

        self._formal_repository_config(config_path)
        config = self.architecture_resolved_config()

        def repository_evaluator(context: SplitExecutionContext) -> DynamicSetSplitEvidence:
            return evaluate_repository_dynamic_set_split(
                context,
                checkpoint_path=checkpoint_path,
                resolved_config=config,
                qualification_protocol=self.protocol,
            )

        return self._execute_validation_candidate_with_evaluator(
            completed_updates=completed_updates,
            checkpoint_path=checkpoint_path,
            execution_progress_path=execution_progress_path,
            model_state_sha256=model_state_sha256,
            callback=repository_evaluator,
        )

    def execute_validation_candidate_test_only(
        self,
        *,
        completed_updates: int,
        checkpoint_path: str | Path,
        execution_progress_path: str | Path,
        model_state_sha256: str,
        callback: Callable[[SplitExecutionContext], DynamicSetSplitEvidence],
    ) -> dict[str, Any]:
        """Evaluate an injected candidate only in a non-formal unit harness."""

        self._require_evaluation_authority(_TEST_EVALUATION_AUTHORITY)
        return self._execute_validation_candidate_with_evaluator(
            completed_updates=completed_updates,
            checkpoint_path=checkpoint_path,
            execution_progress_path=execution_progress_path,
            model_state_sha256=model_state_sha256,
            callback=callback,
        )

    def _execute_validation_candidate_with_evaluator(
        self,
        *,
        completed_updates: int,
        checkpoint_path: str | Path,
        execution_progress_path: str | Path,
        model_state_sha256: str,
        callback: Callable[[SplitExecutionContext], DynamicSetSplitEvidence],
    ) -> dict[str, Any]:
        if not callable(callback):
            raise TypeError("validation callback must be callable")
        self.authenticate_current_source()
        campaign = self._campaign()
        if campaign["state"] != "training":
            raise RuntimeError("validation candidates require an active campaign")
        candidates = campaign["validation_candidates"]
        if type(candidates) is not list:
            raise ValueError("campaign validation history is invalid")
        expected_update = 512 if not candidates else candidates[-1]["completed_updates"] + 512
        if (
            completed_updates != expected_update
            or completed_updates > DEFAULT_CAMPAIGN.maximum_updates
        ):
            raise ValueError("validation candidates must cover every exact 512-update boundary")
        validated_sha256(model_state_sha256, label="candidate model state")
        permit = _active_permit(self._development_ledger(), split="campaign")
        if campaign["campaign_permit_sha256"] != canonical_sha256(asdict(permit)):
            raise ValueError("campaign permit differs from durable campaign state")
        previous_seconds = (
            0.0 if not candidates else float(candidates[-1]["cumulative_training_seconds"])
        )
        receipt = _validated_execution_progress_receipt(
            execution_progress_path,
            protocol=self.protocol,
            lineage_candidates=candidates,
            qualification_root=self.artifacts.root,
            minimum_cumulative_training_seconds=previous_seconds,
        )
        expected_architecture = _architecture_execution_fields_from_attempt(
            campaign["architecture_attempts"][-1]
        )
        if any(
            receipt.evidence()[name] != expected for name, expected in expected_architecture.items()
        ):
            raise ValueError("execution receipt architecture differs from the campaign")
        if receipt.screen_wall_seconds != self.screen_wall_seconds():
            raise ValueError("execution receipt screen timing differs from the sealed screen")
        if receipt.training_limit_reached:
            raise PermissionError("a time-limited execution receipt cannot open validation")
        resolved_checkpoint_path = Path(checkpoint_path).absolute()
        if (
            receipt.completed_updates != completed_updates
            or receipt.model_state_sha256 != model_state_sha256
            or resolved_checkpoint_path.parent.resolve(strict=True) != receipt.path.parent
            or resolved_checkpoint_path.name != f"update_{completed_updates:06d}.pt"
        ):
            raise ValueError("validation arguments differ from the execution receipt")
        checkpoint_bytes = _stable_regular_bytes(resolved_checkpoint_path)
        checkpoint_sha256 = sha256_bytes(checkpoint_bytes)
        if (
            checkpoint_bytes != receipt.active_checkpoint_contents
            or checkpoint_sha256 != receipt.checkpoint_sha256
        ):
            raise ValueError("validation-boundary checkpoint differs from execution progress")
        physical_rows = physical_manifest("development")
        planning_rows = planning_manifest("development")
        development_cache: DynamicSetDevelopmentEvaluationCache | None = None
        development_cache_binding: DynamicSetDevelopmentEvaluationCacheBinding | None = None
        if self.protocol["evaluation_authority"] == _FORMAL_EVALUATION_AUTHORITY:
            expected_attempt_name = f"attempt_{receipt.architecture_attempt_index:02d}"
            if receipt.path.parent.name != expected_attempt_name:
                raise ValueError("execution receipt attempt directory differs from architecture")
            campaign_work_root = receipt.path.parent.parent.resolve(strict=True)
            resolved_config = self.architecture_resolved_config()
            development_cache_binding = _development_evaluation_cache_binding(
                config=resolved_config,
                protocol=self.protocol,
            )
            if development_cache_binding.config_sha256 != receipt.resolved_config_sha256:
                raise ValueError("development cache config differs from execution architecture")
            development_cache = DynamicSetDevelopmentEvaluationCache(
                campaign_work_root / DEVELOPMENT_EVALUATION_CACHE_DIRECTORY_NAME,
                binding=development_cache_binding,
                physical_rows=physical_rows,
                planning_rows=planning_rows,
                trusted_evidence=campaign["development_cache_evidence"],
            )
        context = SplitExecutionContext(
            split="development",
            permit=permit,
            protocol_sha256=self.protocol_sha256,
            checkpoint_sha256=checkpoint_sha256,
            model_state_sha256=model_state_sha256,
            completed_updates=completed_updates,
            physical_rows=physical_rows,
            planning_rows=planning_rows,
            physical_manifest_sha256=FROZEN_PHYSICAL_MANIFEST_SHA256["development"],
            planning_manifest_sha256=FROZEN_PLANNING_MANIFEST_SHA256["development"],
            development_cache_namespace_sha256=(
                None
                if development_cache_binding is None
                else development_cache_binding.namespace_sha256
            ),
            development_cache_evidence_sha256=(
                None
                if campaign["development_cache_evidence"] is None
                else campaign["development_cache_evidence"]["evidence_sha256"]
            ),
            development_evaluation_cache=development_cache,
            **self._split_resource_context_fields(),
        )
        validation_started = time.monotonic()
        try:
            evidence = callback(context)
            self.authenticate_current_source()
            if _stable_regular_bytes(resolved_checkpoint_path) != checkpoint_bytes:
                raise OSError("candidate checkpoint changed during evaluation")
            _assert_execution_receipt_stable(receipt)
            failures, evidence_payload = validate_split_evidence(evidence, context)
            if self.protocol["evaluation_authority"] == _FORMAL_EVALUATION_AUTHORITY and len(
                evidence.planning_pair_provenance
            ) != len(planning_rows):
                raise PermissionError(
                    "formal development evidence lacks complete paired planning provenance"
                )
            if development_cache is None:
                cache_evidence = None
                cache_evidence_mapping = None
            else:
                cache_evidence = development_cache.seal_complete()
                cache_evidence_mapping = cache_evidence.to_dict()
                total_cache_accesses = (
                    development_cache.cold_materializations
                    + development_cache.recertified_materializations
                    + development_cache.warm_hits
                )
                if total_cache_accesses != len(physical_rows) + len(planning_rows):
                    raise ValueError("development evaluator did not consume its complete cache")
                if campaign["development_cache_evidence"] is not None and (
                    development_cache.cold_materializations != 0
                    or development_cache.recertified_materializations != 0
                    or development_cache.warm_hits != total_cache_accesses
                ):
                    raise ValueError("sealed development cache was not consumed as warm evidence")
            self.authenticate_current_source()
            _assert_execution_receipt_stable(receipt)
            validation_wall_seconds = float(time.monotonic() - validation_started)
            if not math.isfinite(validation_wall_seconds) or validation_wall_seconds < 0.0:
                raise ValueError("development validation wall timing is invalid")
            scored_evidence_binding = _write_scored_evidence_artifact(
                self.artifacts,
                evidence=evidence,
                evidence_payload=evidence_payload,
            )
        except BaseException as error:
            failed_validation_wall_seconds = float(time.monotonic() - validation_started)
            if (
                not math.isfinite(failed_validation_wall_seconds)
                or failed_validation_wall_seconds < 0.0
            ):
                failed_validation_wall_seconds = 0.0
            error_result = _record_with_digest(
                {
                    "schema": "dynamic_set_validation_error_v1",
                    "protocol_sha256": self.protocol_sha256,
                    "completed_updates": completed_updates,
                    "checkpoint_sha256": checkpoint_sha256,
                    "model_state_sha256": model_state_sha256,
                    "callback_binding_sha256": context.callback_binding_sha256,
                    **receipt.evidence(),
                    "validation_wall_seconds": failed_validation_wall_seconds,
                    "error": {"type": type(error).__name__, "message": str(error)},
                }
            )
            self.artifacts.write_fresh_json("development_error.json", error_result)
            self._replace_campaign(
                {
                    **{key: item for key, item in campaign.items() if key != "record_sha256"},
                    "state": "terminal",
                    "status": "failed_to_improve",
                    "reason": "development validation callback failed closed",
                    "elapsed_training_hours": receipt.cumulative_training_seconds / 3600.0,
                    "terminal_execution_receipt": receipt.evidence(),
                }
            )
            self._development_ledger().complete(
                permit,
                status="failed",
                result_sha256=error_result["record_sha256"],
            )
            self._publish_development_report()
            raise
        current_incumbent = campaign["incumbent"]
        incumbent_score = None
        if current_incumbent is not None:
            incumbent_score = DynamicSetCheckpointScore(
                completed_updates=current_incumbent["completed_updates"],
                model_state_sha256=current_incumbent["model_state_sha256"],
                components=current_incumbent["score_components"],
                training_support_passed=True,
                selection_guardrails_passed=current_incumbent["selection_guardrails_passed"],
            )
        guardrail_failures = selection_guardrail_failures(failures)
        candidate_score = DynamicSetCheckpointScore(
            completed_updates=completed_updates,
            model_state_sha256=model_state_sha256,
            components=evidence.score_components,
            training_support_passed=evidence.training_support_passed,
            selection_guardrails_passed=not guardrail_failures,
        )
        decision = select_dynamic_set_incumbent(incumbent_score, candidate_score)
        validation_timing = _validation_timing_fields(
            candidates=candidates,
            completed_updates=completed_updates,
            checkpoint_sha256=checkpoint_sha256,
            model_state_sha256=model_state_sha256,
            execution_progress_record_sha256=receipt.record_sha256,
            callback_binding_sha256=context.callback_binding_sha256,
            validation_wall_seconds=validation_wall_seconds,
        )
        entry_body = {
            "sequence": len(candidates),
            "completed_updates": completed_updates,
            "checkpoint_path": str(resolved_checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_bytes": len(checkpoint_bytes),
            "model_state_sha256": model_state_sha256,
            "score_components": dict(evidence.score_components),
            "selection_score": candidate_score.score,
            "training_support_passed": evidence.training_support_passed,
            "selection_guardrail_failures": list(guardrail_failures),
            "selection_guardrails_passed": not guardrail_failures,
            "gate_failures": list(failures),
            "all_gates_passed": not failures,
            "evidence_sha256": evidence_payload["evidence_sha256"],
            **scored_evidence_binding,
            "callback_binding_sha256": context.callback_binding_sha256,
            "development_cache_namespace_sha256": (
                _ZERO_SHA256 if cache_evidence is None else cache_evidence.namespace_sha256
            ),
            "development_cache_input_evidence_sha256": (
                _ZERO_SHA256
                if context.development_cache_evidence_sha256 is None
                else context.development_cache_evidence_sha256
            ),
            "development_cache_evidence_sha256": (
                _ZERO_SHA256 if cache_evidence is None else cache_evidence.evidence_sha256
            ),
            "development_cache_population_sha256": (
                _ZERO_SHA256 if cache_evidence is None else cache_evidence.population_merkle_sha256
            ),
            "development_cache_cold_materializations": (
                0 if development_cache is None else development_cache.cold_materializations
            ),
            "development_cache_recertified_materializations": (
                0 if development_cache is None else development_cache.recertified_materializations
            ),
            "development_cache_warm_hits": (
                0 if development_cache is None else development_cache.warm_hits
            ),
            "accepted": decision.accepted,
            "selection_reason": decision.reason,
            "previous_sha256": candidates[-1]["candidate_sha256"] if candidates else "0" * 64,
            **receipt.evidence(),
            **validation_timing,
        }
        entry = {**entry_body, "candidate_sha256": canonical_sha256(entry_body)}
        candidates = [*candidates, entry]
        incumbent = (
            {
                **entry,
                "score_components": dict(entry["score_components"]),
            }
            if decision.accepted
            else current_incumbent
        )
        self._replace_campaign(
            {
                **{key: item for key, item in campaign.items() if key != "record_sha256"},
                "validation_candidates": candidates,
                "incumbent": incumbent,
                "development_cache_evidence": cache_evidence_mapping,
            }
        )
        return entry

    def finish_campaign(
        self,
        *,
        requested_status: QualificationStatus,
        execution_progress_path: str | Path,
    ) -> dict[str, Any]:
        if requested_status not in TERMINAL_STATUSES:
            raise ValueError("campaign status is outside the four frozen outcomes")
        self.authenticate_current_source()
        campaign = self._campaign()
        if campaign["state"] != "training":
            raise RuntimeError("no active campaign can be finished")
        candidates = campaign["validation_candidates"]
        incumbent = campaign["incumbent"]
        receipt = _validated_finish_execution_receipt(
            execution_progress_path,
            protocol=self.protocol,
            candidates=candidates,
            qualification_root=self.artifacts.root,
        )
        expected_architecture = _architecture_execution_fields_from_attempt(
            campaign["architecture_attempts"][-1]
        )
        if any(
            receipt.evidence()[name] != expected for name, expected in expected_architecture.items()
        ):
            raise ValueError("terminal execution architecture differs from the campaign")
        if receipt.screen_wall_seconds != self.screen_wall_seconds():
            raise ValueError("terminal execution screen timing differs from the sealed screen")
        elapsed_training_hours = receipt.cumulative_training_seconds / 3600.0
        if receipt.training_limit_reached and requested_status != "limit_hit":
            raise PermissionError("the authenticated terminal execution receipt requires limit_hit")
        if incumbent is None:
            if requested_status != "limit_hit":
                raise RuntimeError("campaign cannot finish without a supported incumbent")
            reason = _no_incumbent_limit_hit_reason(
                completed_updates=receipt.completed_updates,
                training_limit_reached=receipt.training_limit_reached,
                limit_hit_reason=receipt.limit_hit_reason,
            )
            self.authenticate_current_source()
            _assert_execution_receipt_stable(receipt)
            permit = _active_permit(self._development_ledger(), split="campaign")
            updated = self._replace_campaign(
                {
                    **{key: item for key, item in campaign.items() if key != "record_sha256"},
                    "state": "terminal",
                    "status": "limit_hit",
                    "reason": reason,
                    "elapsed_training_hours": elapsed_training_hours,
                    "terminal_execution_receipt": receipt.evidence(),
                }
            )
            self._development_ledger().complete(
                permit,
                status="failed",
                result_sha256=updated["record_sha256"],
            )
            return self._publish_development_report()
        if (
            not receipt.training_limit_reached
            and receipt.completed_updates != candidates[-1]["completed_updates"]
        ):
            raise ValueError("a non-limited campaign must finish on its last validation boundary")
        validation_candidates = tuple(
            ValidationCandidate(
                step=item["completed_updates"],
                score=item["selection_score"],
                accepted=item["accepted"],
                training_support_passed=item["training_support_passed"],
                model_state_hash=item["model_state_sha256"],
                checkpoint_path=item["checkpoint_path"],
                selection_guardrails_passed=item["selection_guardrails_passed"],
            )
            for item in candidates
        )
        inspection = CampaignInspection(
            run_directory=str(self.artifacts.root),
            completed_steps=receipt.completed_updates,
            protocol_hash=self.protocol_sha256,
            best_step=incumbent["completed_updates"],
            best_score=incumbent["selection_score"],
            reference_step=0,
            validation_candidates=validation_candidates,
        )
        improvement_failed = any(
            failure.startswith("promotion/paired_score_improvement_fraction")
            or failure.startswith("promotion/paired_bootstrap_lower_bound")
            for failure in incumbent["gate_failures"]
        )
        decision = decide_dynamic_set_campaign(
            inspection,
            absolute_and_promotion_gates_passed=incumbent["all_gates_passed"],
            failed_to_improve=improvement_failed,
            elapsed_training_hours=elapsed_training_hours,
            execution_limit_hit_reason=receipt.limit_hit_reason,
        )
        if decision.status != requested_status:
            raise ValueError(
                f"requested campaign status {requested_status!r} is unsupported; recomputed {decision.status!r}"
            )
        if requested_status == "qualified_convergence" and receipt.training_limit_reached:
            raise PermissionError("qualified convergence cannot be declared after the time limit")
        self.authenticate_current_source()
        _assert_execution_receipt_stable(receipt)
        permit = _active_permit(self._development_ledger(), split="campaign")
        updated = self._replace_campaign(
            {
                **{key: item for key, item in campaign.items() if key != "record_sha256"},
                "state": "terminal",
                "status": requested_status,
                "reason": decision.reason,
                "elapsed_training_hours": elapsed_training_hours,
                "terminal_execution_receipt": receipt.evidence(),
            }
        )
        self._development_ledger().complete(
            permit,
            status="passed" if requested_status == "qualified_convergence" else "failed",
            result_sha256=updated["record_sha256"],
        )
        return self._publish_development_report()

    @staticmethod
    def _scored_evidence_binding(value: Mapping[str, Any]) -> dict[str, Any]:
        return {
            name: value[name]
            for name in (
                "scored_evidence_artifact_name",
                "scored_evidence_artifact_sha256",
                "scored_evidence_artifact_bytes",
                "scored_evidence_record_sha256",
            )
        }

    def _campaign_permit(self) -> SplitPermit:
        ledger = self._development_ledger().load()
        begins = [
            item
            for item in ledger["transitions"]
            if item["event"] == "begin" and item["split"] == "campaign"
        ]
        if len(begins) != 1:
            raise ValueError("development ledger lacks one exact campaign permit")
        begin = begins[0]
        return SplitPermit(
            split="campaign",
            index=begin["index"],
            nonce=begin["nonce"],
            protocol_sha256=self.protocol_sha256,
        )

    def _development_split_context(
        self,
        candidate: Mapping[str, Any],
    ) -> SplitExecutionContext:
        physical_rows = physical_manifest("development")
        planning_rows = planning_manifest("development")
        return SplitExecutionContext(
            split="development",
            permit=self._campaign_permit(),
            protocol_sha256=self.protocol_sha256,
            checkpoint_sha256=candidate["checkpoint_sha256"],
            model_state_sha256=candidate["model_state_sha256"],
            completed_updates=candidate["completed_updates"],
            physical_rows=physical_rows,
            planning_rows=planning_rows,
            physical_manifest_sha256=FROZEN_PHYSICAL_MANIFEST_SHA256["development"],
            planning_manifest_sha256=FROZEN_PLANNING_MANIFEST_SHA256["development"],
            development_cache_namespace_sha256=(
                None
                if candidate["development_cache_namespace_sha256"] == _ZERO_SHA256
                else candidate["development_cache_namespace_sha256"]
            ),
            development_cache_evidence_sha256=(
                None
                if candidate["development_cache_input_evidence_sha256"] == _ZERO_SHA256
                else candidate["development_cache_input_evidence_sha256"]
            ),
            **self._split_resource_context_fields(),
        )

    def _publish_development_report(self) -> dict[str, Any]:
        if "development_report.json" in self.artifacts.inventory():
            raise FileExistsError("development report is already published")
        campaign = self._campaign()
        status = campaign["status"]
        if status not in TERMINAL_STATUSES or campaign["state"] != "terminal":
            raise RuntimeError("development report requires a terminal four-way campaign outcome")
        ledger_bytes = self.artifacts.read_bytes("development_ledger.json")
        incumbent = campaign["incumbent"]
        checkpoint: dict[str, Any] | None = None
        if status == "qualified_convergence":
            if incumbent is None or not incumbent["all_gates_passed"]:
                raise PermissionError("qualified convergence lacks a fully passed incumbent")
            contents = _stable_regular_bytes(incumbent["checkpoint_path"])
            digest = sha256_bytes(contents)
            if digest != incumbent["checkpoint_sha256"]:
                raise OSError("selected checkpoint bytes changed before development publication")
            self.artifacts.write_fresh_bytes("development_model.pt", contents)
            checkpoint = {
                "artifact_name": "development_model.pt",
                "sha256": digest,
                "bytes": len(contents),
                "model_state_sha256": incumbent["model_state_sha256"],
                "completed_updates": incumbent["completed_updates"],
            }
            evidence_context = self._development_split_context(incumbent)
            _read_scored_evidence_artifact(
                self.artifacts,
                self._scored_evidence_binding(incumbent),
                evidence_context,
                recompute=False,
            )
        body = {
            "schema": "dynamic_set_development_report_v1",
            "specification_version": SPECIFICATION_VERSION,
            "protocol_sha256": self.protocol_sha256,
            "evaluation_authority": self.protocol["evaluation_authority"],
            "formal": self.protocol["evaluation_authority"] == _FORMAL_EVALUATION_AUTHORITY,
            "status": status,
            "passed": status == "qualified_convergence",
            "reason": campaign["reason"],
            "screen_result_sha256": campaign["screen_result_sha256"],
            "campaign_state_sha256": campaign["record_sha256"],
            "development_ledger_sha256": sha256_bytes(ledger_bytes),
            "selected_candidate_sha256": (
                None if incumbent is None else incumbent["candidate_sha256"]
            ),
            "selected_evidence_sha256": (
                None if incumbent is None else incumbent["evidence_sha256"]
            ),
            "selected_scored_evidence": (
                None if incumbent is None else self._scored_evidence_binding(incumbent)
            ),
            "development_cache_evidence": campaign["development_cache_evidence"],
            "checkpoint": checkpoint,
            "physical_manifest_sha256": FROZEN_PHYSICAL_MANIFEST_SHA256["development"],
            "physical_row_count": PHYSICAL_SPLIT_SIZES["development"],
            "planning_manifest_sha256": FROZEN_PLANNING_MANIFEST_SHA256["development"],
            "planning_task_count": PLANNING_SPLIT_SIZES["development"],
        }
        report = _record_with_digest(body)
        self.artifacts.write_fresh_json("development_report.json", report)
        return report

    def _independent_review_recomputation(self) -> dict[str, Any]:
        report = self._development_report()
        if report["status"] != "qualified_convergence" or report["passed"] is not True:
            raise PermissionError(
                "only qualified development evidence can be reviewed for promotion"
            )
        campaign = self._campaign()
        incumbent = campaign["incumbent"]
        if type(incumbent) is not dict:
            raise ValueError("qualified development lacks its selected candidate")
        binding = self._scored_evidence_binding(incumbent)
        if report["selected_scored_evidence"] != binding:
            raise ValueError("development report scored-evidence binding differs")
        context = self._development_split_context(incumbent)
        record, failures, summary = _read_scored_evidence_artifact(
            self.artifacts,
            binding,
            context,
            recompute=True,
        )
        assert failures is not None and summary is not None
        if (
            failures
            or summary["passed"] is not True
            or summary["evidence_sha256"] != incumbent["evidence_sha256"]
            or summary["selection_score"] != incumbent["selection_score"]
            or summary["gate_failures"] != incumbent["gate_failures"]
            or record["record_sha256"] != binding["scored_evidence_record_sha256"]
        ):
            raise PermissionError("selected development evidence fails independent recomputation")
        review_body = {
            "schema": "dynamic_set_independent_recomputation_v1",
            "protocol_sha256": self.protocol_sha256,
            "candidate_sha256": incumbent["candidate_sha256"],
            "scored_evidence_artifact_sha256": binding["scored_evidence_artifact_sha256"],
            "scored_evidence_record_sha256": binding["scored_evidence_record_sha256"],
            "recomputed_evidence_sha256": summary["evidence_sha256"],
            "recomputed_selection_score": summary["selection_score"],
            "recomputed_gate_failures": list(failures),
            "recomputed_passed": not failures,
        }
        return {
            "scored_evidence_artifact_sha256": binding["scored_evidence_artifact_sha256"],
            "scored_evidence_record_sha256": binding["scored_evidence_record_sha256"],
            "review_evidence_sha256": canonical_sha256(review_body),
        }

    def record_independent_development_review(self, receipt: Mapping[str, Any]) -> dict[str, Any]:
        self.authenticate_current_source()
        if "development_review_receipt.json" in self.artifacts.inventory():
            raise FileExistsError("development review receipt already exists")
        report_bytes = self.artifacts.read_bytes("development_report.json")
        report = self._development_report()
        if report["status"] != "qualified_convergence" or report["passed"] is not True:
            raise PermissionError("only qualified development evidence can be reviewed")
        checkpoint_bytes = self.artifacts.read_bytes("development_model.pt")
        ledger_bytes = self.artifacts.read_bytes("development_ledger.json")
        value = _exact_dict(receipt, label="independent development review")
        expected_keys = {
            "schema",
            "reviewer",
            "decision",
            "protocol_sha256",
            "development_report_sha256",
            "development_checkpoint_sha256",
            "development_ledger_sha256",
            "scored_evidence_artifact_sha256",
            "scored_evidence_record_sha256",
            "review_evidence_sha256",
            "receipt_sha256",
        }
        if set(value) != expected_keys:
            raise ValueError("independent development review schema differs")
        supplied = validated_sha256(value["receipt_sha256"], label="development review receipt")
        body = {key: item for key, item in value.items() if key != "receipt_sha256"}
        expected = {
            "schema": "dynamic_set_independent_development_review_v2",
            "decision": "approved",
            "protocol_sha256": self.protocol_sha256,
            "development_report_sha256": sha256_bytes(report_bytes),
            "development_checkpoint_sha256": sha256_bytes(checkpoint_bytes),
            "development_ledger_sha256": sha256_bytes(ledger_bytes),
        }
        if any(body.get(name) != item for name, item in expected.items()):
            raise PermissionError("independent development receipt digest binding differs")
        if type(body.get("reviewer")) is not str or not body["reviewer"].strip():
            raise ValueError("independent development reviewer must be named")
        recomputed = self._independent_review_recomputation()
        if any(body.get(name) != item for name, item in recomputed.items()):
            raise PermissionError(
                "independent development receipt differs from raw evidence recomputation"
            )
        if canonical_sha256(body) != supplied:
            raise ValueError("independent development receipt self-hash differs")
        self.authenticate_current_source()
        self.artifacts.write_fresh_json("development_review_receipt.json", value)
        return value

    def create_protected_ledger(self) -> dict[str, Any]:
        self.authenticate_current_source()
        if "protected_ledger.json" in self.artifacts.inventory():
            raise FileExistsError("protected ledger already exists")
        # Revalidation of the complete receipt is intentional; merely having a
        # file with the right basename never opens protected data.
        receipt = self.artifacts.read_json("development_review_receipt.json")
        report_bytes = self.artifacts.read_bytes("development_report.json")
        checkpoint_bytes = self.artifacts.read_bytes("development_model.pt")
        ledger_bytes = self.artifacts.read_bytes("development_ledger.json")
        body = {key: item for key, item in receipt.items() if key != "receipt_sha256"}
        recomputed = self._independent_review_recomputation()
        if (
            receipt.get("schema") != "dynamic_set_independent_development_review_v2"
            or receipt.get("decision") != "approved"
            or receipt.get("protocol_sha256") != self.protocol_sha256
            or receipt.get("development_report_sha256") != sha256_bytes(report_bytes)
            or receipt.get("development_checkpoint_sha256") != sha256_bytes(checkpoint_bytes)
            or receipt.get("development_ledger_sha256") != sha256_bytes(ledger_bytes)
            or any(receipt.get(name) != item for name, item in recomputed.items())
            or canonical_sha256(body) != receipt.get("receipt_sha256")
        ):
            raise PermissionError("development review receipt no longer binds current artifacts")
        protected = OrderedSplitLedger(
            self.artifacts,
            artifact_name="protected_ledger.json",
            protocol_sha256=self.protocol_sha256,
            split_order=PROTECTED_SPLIT_ORDER,
        )
        self.authenticate_current_source()
        return protected.create_fresh()

    def _protected_ledger(self) -> OrderedSplitLedger:
        return OrderedSplitLedger(
            self.artifacts,
            artifact_name="protected_ledger.json",
            protocol_sha256=self.protocol_sha256,
            split_order=PROTECTED_SPLIT_ORDER,
        )

    def execute_protected_split(
        self,
        split: ProtectedSplit,
        *,
        config_path: str | Path,
    ) -> dict[str, Any]:
        """Consume one formal protected split with the repository evaluator."""

        self._formal_repository_config(config_path)
        config = self.architecture_resolved_config()
        checkpoint_path = self.artifacts.root / "development_model.pt"

        def repository_evaluator(context: SplitExecutionContext) -> DynamicSetSplitEvidence:
            return evaluate_repository_dynamic_set_split(
                context,
                checkpoint_path=checkpoint_path,
                resolved_config=config,
                qualification_protocol=self.protocol,
            )

        return self._execute_protected_split_with_evaluator(
            split,
            repository_evaluator,
        )

    def execute_protected_split_test_only(
        self,
        split: ProtectedSplit,
        callback: Callable[[SplitExecutionContext], DynamicSetSplitEvidence],
    ) -> dict[str, Any]:
        """Consume an injected split only inside a non-formal unit harness."""

        self._require_evaluation_authority(_TEST_EVALUATION_AUTHORITY)
        return self._execute_protected_split_with_evaluator(split, callback)

    def _execute_protected_split_with_evaluator(
        self,
        split: ProtectedSplit,
        callback: Callable[[SplitExecutionContext], DynamicSetSplitEvidence],
    ) -> dict[str, Any]:
        if split not in PROTECTED_SPLIT_ORDER:
            raise ValueError("unknown protected split")
        if not callable(callback):
            raise TypeError("protected callback must be callable")
        self.authenticate_current_source()
        checkpoint_bytes = self.artifacts.read_bytes("development_model.pt")
        checkpoint_sha256 = sha256_bytes(checkpoint_bytes)
        development = self._development_report()
        checkpoint = development["checkpoint"]
        if type(checkpoint) is not dict or checkpoint.get("sha256") != checkpoint_sha256:
            raise PermissionError("protected checkpoint differs from passed development")
        # Manifest construction is public and deterministic.  Complete it
        # before minting the one-shot permit so a local allocation failure
        # cannot strand an opened protected population.
        physical_rows = physical_manifest(split)
        planning_rows = planning_manifest(split)
        ledger = self._protected_ledger()
        permit = ledger.begin(split)
        context = SplitExecutionContext(
            split=split,
            permit=permit,
            protocol_sha256=self.protocol_sha256,
            checkpoint_sha256=checkpoint_sha256,
            model_state_sha256=checkpoint["model_state_sha256"],
            completed_updates=checkpoint["completed_updates"],
            physical_rows=physical_rows,
            planning_rows=planning_rows,
            physical_manifest_sha256=FROZEN_PHYSICAL_MANIFEST_SHA256[split],
            planning_manifest_sha256=FROZEN_PLANNING_MANIFEST_SHA256[split],
            protected_ledger=ledger,
            **self._split_resource_context_fields(),
        )
        population_claims: dict[str, str] = {}
        try:
            evidence = callback(context)
            self.authenticate_current_source()
            if self.artifacts.read_bytes("development_model.pt") != checkpoint_bytes:
                raise OSError("protected checkpoint changed during evaluation")
            if self.protocol["evaluation_authority"] == _FORMAL_EVALUATION_AUTHORITY:
                expected_claims = {
                    PHYSICAL_POPULATION_CLAIM_PURPOSE: (
                        protected_physical_population_claim_binding(
                            split=split,
                            protocol_sha256=self.protocol_sha256,
                            manifest_sha256=context.physical_manifest_sha256,
                            row_count=len(physical_rows),
                        )
                    ),
                    PLANNING_POPULATION_CLAIM_PURPOSE: (
                        protected_planning_population_claim_binding(
                            split=split,
                            protocol_sha256=self.protocol_sha256,
                            manifest_sha256=context.planning_manifest_sha256,
                            row_count=len(planning_rows),
                        )
                    ),
                }
                population_claims = dict(ledger.active_claims(permit))
                if population_claims != expected_claims:
                    raise PermissionError(
                        "formal protected evaluation lacks both exact population claims"
                    )
            failures, evidence_payload = validate_split_evidence(evidence, context)
            if self.protocol["evaluation_authority"] == _FORMAL_EVALUATION_AUTHORITY and len(
                evidence.planning_pair_provenance
            ) != len(planning_rows):
                raise PermissionError(
                    "formal protected evidence lacks complete paired planning provenance"
                )
            self.authenticate_current_source()
            scored_evidence_binding = _write_scored_evidence_artifact(
                self.artifacts,
                evidence=evidence,
                evidence_payload=evidence_payload,
            )
            body = {
                "schema": "dynamic_set_protected_split_result_v1",
                "split": split,
                "protocol_sha256": self.protocol_sha256,
                "permit": asdict(permit),
                "callback_binding_sha256": context.callback_binding_sha256,
                "checkpoint_sha256": checkpoint_sha256,
                "model_state_sha256": checkpoint["model_state_sha256"],
                "physical_manifest_sha256": context.physical_manifest_sha256,
                "physical_row_count": len(physical_rows),
                "planning_manifest_sha256": context.planning_manifest_sha256,
                "planning_task_count": len(planning_rows),
                "population_claims": population_claims,
                "population_claims_sha256": canonical_sha256(
                    {
                        "schema": "dynamic_set_protected_population_claims_v1",
                        "split": split,
                        "claims": population_claims,
                    }
                ),
                "evidence_sha256": evidence_payload["evidence_sha256"],
                **scored_evidence_binding,
                "selection_score": evidence_payload["selection_score"],
                "gate_failures": list(failures),
                "passed": not failures,
                "error": None,
            }
        except BaseException as error:
            try:
                population_claims = dict(ledger.active_claims(permit))
            except (RuntimeError, ValueError):
                population_claims = {}
            body = {
                "schema": "dynamic_set_protected_split_result_v1",
                "split": split,
                "protocol_sha256": self.protocol_sha256,
                "permit": asdict(permit),
                "callback_binding_sha256": context.callback_binding_sha256,
                "checkpoint_sha256": checkpoint_sha256,
                "model_state_sha256": checkpoint["model_state_sha256"],
                "physical_manifest_sha256": context.physical_manifest_sha256,
                "physical_row_count": len(physical_rows),
                "planning_manifest_sha256": context.planning_manifest_sha256,
                "planning_task_count": len(planning_rows),
                "population_claims": population_claims,
                "population_claims_sha256": canonical_sha256(
                    {
                        "schema": "dynamic_set_protected_population_claims_v1",
                        "split": split,
                        "claims": population_claims,
                    }
                ),
                "evidence_sha256": None,
                "scored_evidence_artifact_name": None,
                "scored_evidence_artifact_sha256": None,
                "scored_evidence_artifact_bytes": None,
                "scored_evidence_record_sha256": None,
                "selection_score": None,
                "gate_failures": [f"callback_error:{type(error).__name__}:{error}"],
                "passed": False,
                "error": {"type": type(error).__name__, "message": str(error)},
            }
            result = _record_with_digest(body)
            self.artifacts.write_fresh_json(f"{split}_result.json", result)
            ledger.complete(permit, status="failed", result_sha256=result["record_sha256"])
            self._publish_qualification_report(status="failed_to_improve")
            raise
        result = _record_with_digest(body)
        self.artifacts.write_fresh_json(f"{split}_result.json", result)
        ledger.complete(
            permit,
            status="passed" if result["passed"] else "failed",
            result_sha256=result["record_sha256"],
        )
        if not result["passed"]:
            self._publish_qualification_report(status="failed_to_improve")
        elif split == "compositional_ood":
            self._publish_qualification_report(status="qualified_convergence")
        return result

    def _publish_qualification_report(self, *, status: QualificationStatus) -> dict[str, Any]:
        if status not in {"qualified_convergence", "failed_to_improve"}:
            raise ValueError("protected evaluation has only pass or gate-rejection outcomes")
        if "qualification_report.json" in self.artifacts.inventory():
            raise FileExistsError("qualification report is already published")
        ledger = self._protected_ledger().load()
        completed = [
            transition for transition in ledger["transitions"] if transition["event"] == "complete"
        ]
        result_names = [f"{transition['split']}_result.json" for transition in completed]
        results = [self.artifacts.read_json(name) for name in result_names]
        for item in results:
            split = item["split"]
            claims = _exact_dict(
                item.get("population_claims"),
                label="protected result population claims",
            )
            ledger_claims = {
                transition["purpose"]: transition["binding_sha256"]
                for transition in ledger["transitions"]
                if transition["event"] == "claim" and transition["split"] == split
            }
            if claims != ledger_claims or item.get("population_claims_sha256") != canonical_sha256(
                {
                    "schema": "dynamic_set_protected_population_claims_v1",
                    "split": split,
                    "claims": claims,
                }
            ):
                raise ValueError("protected result population claims differ from its ledger")
            if (
                self.protocol["evaluation_authority"] == _FORMAL_EVALUATION_AUTHORITY
                and item.get("evidence_sha256") is not None
                and set(claims)
                != {PHYSICAL_POPULATION_CLAIM_PURPOSE, PLANNING_POPULATION_CLAIM_PURPOSE}
            ):
                raise PermissionError("passed protected result lacks both population claims")
            if item.get("evidence_sha256") is None:
                if any(
                    item.get(name) is not None
                    for name in (
                        "scored_evidence_artifact_name",
                        "scored_evidence_artifact_sha256",
                        "scored_evidence_artifact_bytes",
                        "scored_evidence_record_sha256",
                    )
                ):
                    raise ValueError("failed protected result has a partial evidence binding")
                continue
            permit_payload = _exact_dict(item.get("permit"), label="protected result permit")
            permit = SplitPermit(**permit_payload)
            context = SplitExecutionContext(
                split=split,
                permit=permit,
                protocol_sha256=self.protocol_sha256,
                checkpoint_sha256=item["checkpoint_sha256"],
                model_state_sha256=item["model_state_sha256"],
                completed_updates=self._development_report()["checkpoint"]["completed_updates"],
                physical_rows=physical_manifest(split),
                planning_rows=planning_manifest(split),
                physical_manifest_sha256=FROZEN_PHYSICAL_MANIFEST_SHA256[split],
                planning_manifest_sha256=FROZEN_PLANNING_MANIFEST_SHA256[split],
                **self._split_resource_context_fields(),
            )
            _read_scored_evidence_artifact(
                self.artifacts,
                self._scored_evidence_binding(item),
                context,
                recompute=False,
            )
        if status == "qualified_convergence":
            if (
                ledger["terminal"] is not True
                or ledger["next_index"] != len(PROTECTED_SPLIT_ORDER)
                or [item["split"] for item in results] != list(PROTECTED_SPLIT_ORDER)
                or any(item["passed"] is not True for item in results)
            ):
                raise PermissionError("qualified convergence lacks four passed protected splits")
        elif not results or results[-1]["passed"] is not False or ledger["terminal"] is not True:
            raise PermissionError("protected rejection lacks one terminal failed split")
        development_report_bytes = self.artifacts.read_bytes("development_report.json")
        review_bytes = self.artifacts.read_bytes("development_review_receipt.json")
        body = {
            "schema": "dynamic_set_qualification_report_v1",
            "specification_version": SPECIFICATION_VERSION,
            "protocol_sha256": self.protocol_sha256,
            "evaluation_authority": self.protocol["evaluation_authority"],
            "formal": self.protocol["evaluation_authority"] == _FORMAL_EVALUATION_AUTHORITY,
            "status": status,
            "passed": status == "qualified_convergence",
            "development_report_sha256": sha256_bytes(development_report_bytes),
            "development_review_receipt_sha256": sha256_bytes(review_bytes),
            "checkpoint_sha256": sha256_bytes(self.artifacts.read_bytes("development_model.pt")),
            "protected_ledger_sha256": sha256_bytes(
                self.artifacts.read_bytes("protected_ledger.json")
            ),
            "opened_splits": [item["split"] for item in results],
            "stopped_after": results[-1]["split"],
            "results": [
                {
                    "split": item["split"],
                    "result_sha256": item["record_sha256"],
                    "passed": item["passed"],
                    "gate_failures": item["gate_failures"],
                    "population_claims_sha256": item["population_claims_sha256"],
                    **(
                        {
                            "scored_evidence_artifact_name": item["scored_evidence_artifact_name"],
                            "scored_evidence_artifact_sha256": item[
                                "scored_evidence_artifact_sha256"
                            ],
                            "scored_evidence_record_sha256": item["scored_evidence_record_sha256"],
                        }
                        if item["evidence_sha256"] is not None
                        else {
                            "scored_evidence_artifact_name": None,
                            "scored_evidence_artifact_sha256": None,
                            "scored_evidence_record_sha256": None,
                        }
                    ),
                }
                for item in results
            ],
        }
        report = _record_with_digest(body)
        self.artifacts.write_fresh_json("qualification_report.json", report)
        return report

    def status(self) -> dict[str, Any]:
        """Return a read-only summary without beginning any split."""

        inventory = self.artifacts.inventory()
        campaign = self._campaign()
        development_ledger = self._development_ledger().load()
        protected = (
            self._protected_ledger().load() if "protected_ledger.json" in inventory else None
        )
        return {
            "schema": "dynamic_set_qualification_status_v1",
            "protocol_sha256": self.protocol_sha256,
            "campaign_state": campaign["state"],
            "campaign_status": campaign["status"],
            "development_next_index": development_ledger["next_index"],
            "development_terminal": development_ledger["terminal"],
            "protected_next_split": (
                None
                if protected is None or protected["terminal"]
                else PROTECTED_SPLIT_ORDER[protected["next_index"]]
            ),
            "protected_terminal": None if protected is None else protected["terminal"],
            "qualification_status": (
                None
                if "qualification_report.json" not in inventory
                else self.artifacts.read_json("qualification_report.json")["status"]
            ),
            "artifact_inventory": sorted(inventory),
        }


def make_independent_review_receipt(
    qualification: DynamicSetQualification,
    *,
    reviewer: str,
) -> dict[str, Any]:
    """Build the exact digest receipt for an external review decision.

    Building the receipt does not publish it or open protected access.  A
    reviewer can generate/sign/archive this JSON independently, after which
    :meth:`DynamicSetQualification.record_independent_development_review`
    revalidates every referenced byte string.
    """

    if not isinstance(qualification, DynamicSetQualification):
        raise TypeError("qualification must be DynamicSetQualification")
    if type(reviewer) is not str or not reviewer.strip():
        raise ValueError("reviewer must be a nonempty string")
    recomputed = qualification._independent_review_recomputation()
    body = {
        "schema": "dynamic_set_independent_development_review_v2",
        "reviewer": reviewer,
        "decision": "approved",
        "protocol_sha256": qualification.protocol_sha256,
        "development_report_sha256": sha256_bytes(
            qualification.artifacts.read_bytes("development_report.json")
        ),
        "development_checkpoint_sha256": sha256_bytes(
            qualification.artifacts.read_bytes("development_model.pt")
        ),
        "development_ledger_sha256": sha256_bytes(
            qualification.artifacts.read_bytes("development_ledger.json")
        ),
        **recomputed,
    }
    return {**body, "receipt_sha256": canonical_sha256(body)}


__all__ = [
    "ARTIFACT_NAMES",
    "EXECUTION_PROGRESS_SCHEMA",
    "EXECUTION_SCHEDULE_SEED",
    "EXECUTION_TIMING_EVIDENCE_SCHEMA",
    "FOUNDATION_ARTIFACT_NAMES",
    "PAIRED_SCORE_SCHEMA",
    "PROTECTED_SPLIT_ORDER",
    "SCREEN_OBJECTIVE_REGRET_SCHEMA",
    "SCREEN_MANIFEST_SHA256",
    "SCREEN_ROWS",
    "DynamicSetQualification",
    "DynamicSetPromotionEvidence",
    "DynamicSetSourceFreeze",
    "DynamicSetSplitEvidence",
    "DynamicSetIntegrityEvidence",
    "PhysicalCellEvaluation",
    "QualificationStatus",
    "ScreenExecutionContext",
    "ScreenExecutionResult",
    "SplitExecutionContext",
    "authenticate_dynamic_set_source_freeze",
    "build_dynamic_set_protocol_binding",
    "build_dynamic_set_split_evidence",
    "capture_dynamic_set_source_freeze",
    "dynamic_set_source_sha256",
    "evaluate_repository_disposable_screen",
    "evaluate_repository_dynamic_set_split",
    "make_independent_review_receipt",
    "validate_known_action_foundation",
    "validate_split_evidence",
]
