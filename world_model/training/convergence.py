"""Objective plateau decisions for sustained closed-loop training campaigns.

Safe extension uses only validation-selected checkpoints. Plateau evidence
also considers raw primary scores from rejected candidates when their causal
training support is valid; training loss is never a convergence signal.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from world_model.training.trainer import (
    _ROLLOUT_SELECTION_METRIC_VERSION,
    _finite_nonnegative_integer,
    _model_state_hash,
    _rollout_validation_protocol_hash,
    _verified_selector_checkpoint,
)
from world_model.utils.config import OrpheusConfig

_VALIDATION_CHECKPOINT = re.compile(r"validation_step_(\d+)\.pt$")


@dataclass(frozen=True)
class ValidationCandidate:
    """One numbered broad-validation candidate with verified tensors."""

    step: int
    score: float
    accepted: bool
    training_support_passed: bool
    model_state_hash: str
    checkpoint_path: str


@dataclass(frozen=True)
class CampaignInspection:
    """Verified completion state for one trainer segment."""

    run_directory: str
    completed_steps: int
    protocol_hash: str
    best_step: int
    best_score: float
    reference_step: int
    validation_candidates: tuple[ValidationCandidate, ...]

    @property
    def accepted_validations(self) -> tuple[ValidationCandidate, ...]:
        return tuple(candidate for candidate in self.validation_candidates if candidate.accepted)


@dataclass(frozen=True)
class ConvergenceDecision:
    """Whether a completed campaign should receive another causal block."""

    status: str
    reason: str
    completed_steps: int
    next_total_steps: int | None
    best_step: int
    best_score: float
    tail_start_step: int
    prior_best_step: int | None
    prior_best_score: float | None
    tail_best_step: int | None
    tail_best_score: float | None
    relative_tail_gain: float | None
    plateau_candidate_steps: tuple[int, ...]
    plateau_candidate_scores: tuple[float, ...]
    plateau_candidate_accepted: tuple[bool, ...]
    plateau_candidate_training_support_passed: tuple[bool, ...]
    plateau_primary_gain: float | None
    minimum_relative_gain: float
    maximum_total_steps: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CampaignIncompleteError(RuntimeError):
    """Raised when a trainer segment has not produced complete artifacts."""


def _load_mapping(path: Path) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"checkpoint is not a mapping: {path}")
    return payload


def _integer_metric(metrics: Mapping[str, Any], key: str) -> int:
    value = metrics.get(key)
    if value is None:
        raise ValueError(f"checkpoint is missing {key}")
    return _finite_nonnegative_integer(value, name=key)


def _validation_candidate(path: Path, *, protocol_hash: str) -> ValidationCandidate:
    """Return one tensor-verified numbered validation candidate."""

    match = _VALIDATION_CHECKPOINT.fullmatch(path.name)
    if match is None:
        raise ValueError(f"unexpected validation checkpoint name: {path.name}")
    filename_step = int(match.group(1))
    payload = _load_mapping(path)
    checkpoint_step = _finite_nonnegative_integer(
        payload.get("step"),
        name="step",
    )
    if checkpoint_step != filename_step:
        raise ValueError(
            f"validation checkpoint step mismatch: {path.name} contains {checkpoint_step}"
        )
    metrics = payload.get("metrics")
    model_state = payload.get("model_state")
    if not isinstance(metrics, Mapping) or not isinstance(model_state, Mapping):
        raise ValueError(f"validation checkpoint lacks metrics/model_state: {path}")
    if metrics.get("rollout_validation_protocol_hash") != protocol_hash:
        raise ValueError(f"validation protocol changed within campaign: {path}")
    if float(metrics.get("rollout_selection_metric_version", math.nan)) != (
        _ROLLOUT_SELECTION_METRIC_VERSION
    ):
        raise ValueError(f"validation selector metric version changed within campaign: {path}")
    accepted_value = float(metrics.get("selection_accepted", math.nan))
    if accepted_value not in {0.0, 1.0}:
        raise ValueError(f"validation checkpoint has invalid acceptance state: {path}")
    accepted = accepted_value == 1.0
    support_required_value = float(metrics.get("selection_training_support_required", math.nan))
    support_passed_value = float(metrics.get("selection_training_support_passed", math.nan))
    if support_required_value not in {0.0, 1.0} or support_passed_value not in {0.0, 1.0}:
        raise ValueError(f"validation checkpoint has invalid training-support state: {path}")
    training_support_passed = support_passed_value == 1.0
    if accepted and not training_support_passed:
        raise ValueError(f"accepted validation checkpoint failed training support: {path}")
    score = float(metrics.get("validation_rollout_selection_score", math.nan))
    if not math.isfinite(score) or score < 0:
        raise ValueError(f"validation checkpoint has an invalid physical score: {path}")
    actual_hash = _model_state_hash(model_state)
    if metrics.get("checkpoint_model_state_hash") != actual_hash:
        raise ValueError(f"validation checkpoint tensor hash mismatch: {path}")
    if accepted:
        incumbent_score = float(metrics.get("best_rollout_selection_score", math.nan))
        if not math.isclose(
            score,
            incumbent_score,
            rel_tol=1.0e-7,
            abs_tol=1.0e-9,
        ):
            raise ValueError(f"accepted validation score is not linked to its incumbent: {path}")
        incumbent_step = _integer_metric(metrics, "best_rollout_checkpoint_step")
        if incumbent_step != checkpoint_step:
            raise ValueError(f"accepted validation does not contain its selected step: {path}")
        if metrics.get("best_rollout_model_state_hash") != actual_hash:
            raise ValueError(f"accepted validation is not linked to selected tensors: {path}")
        if float(metrics.get("checkpoint_contains_best_rollout_weights", 0.0)) != 1.0:
            raise ValueError(f"accepted validation does not declare selected tensors: {path}")
    return ValidationCandidate(
        step=checkpoint_step,
        score=score,
        accepted=accepted,
        training_support_passed=training_support_passed,
        model_state_hash=actual_hash,
        checkpoint_path=str(path.resolve()),
    )


def inspect_completed_campaign(
    run_directory: str | Path,
    config: OrpheusConfig,
) -> CampaignInspection:
    """Verify the summary, resumable iterate, selector files, and numbered history."""

    run_path = Path(run_directory).expanduser().resolve()
    summary_path = run_path / "train_summary.json"
    if not summary_path.is_file():
        raise CampaignIncompleteError(f"training summary does not exist yet: {summary_path}")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise CampaignIncompleteError(
            f"training summary is not readable: {summary_path}"
        ) from error
    if not isinstance(summary, Mapping):
        raise ValueError("training summary must contain a JSON object")
    completed_steps = _finite_nonnegative_integer(
        summary.get("completed_steps"),
        name="completed_steps",
    )

    checkpoint_directory = run_path / "checkpoints"
    last_path = checkpoint_directory / "last.pt"
    if not last_path.is_file():
        raise CampaignIncompleteError(f"resumable checkpoint does not exist: {last_path}")
    last_payload = _load_mapping(last_path)
    if (
        _finite_nonnegative_integer(
            last_payload.get("step"),
            name="step",
        )
        != completed_steps
    ):
        raise CampaignIncompleteError(
            "training summary and resumable checkpoint do not share a completed step"
        )
    last_metrics = last_payload.get("metrics")
    if not isinstance(last_metrics, Mapping):
        raise ValueError("resumable checkpoint does not contain metrics")
    protocol_hash = _rollout_validation_protocol_hash(config)
    if last_metrics.get("rollout_validation_protocol_hash") != protocol_hash:
        raise ValueError("resumable checkpoint uses a different validation protocol")

    best_hash = last_metrics.get("best_rollout_model_state_hash")
    reference_hash = last_metrics.get("reference_rollout_model_state_hash")
    if not isinstance(best_hash, str) or not isinstance(reference_hash, str):
        raise ValueError("resumable checkpoint lacks linked selector tensor hashes")
    best_step = _integer_metric(last_metrics, "best_rollout_checkpoint_step")
    reference_step = _integer_metric(last_metrics, "reference_rollout_checkpoint_step")
    verified_best = _verified_selector_checkpoint(
        checkpoint_directory / "best_rollout.pt",
        config,
        prefix="best_rollout",
        expected_model_state_hash=best_hash,
        expected_step=best_step,
    )
    verified_reference = _verified_selector_checkpoint(
        checkpoint_directory / "reference_rollout.pt",
        config,
        prefix="reference_rollout",
        expected_model_state_hash=reference_hash,
        expected_step=reference_step,
    )
    if verified_best is None or verified_reference is None:
        raise ValueError("linked best/reference selector checkpoint verification failed")
    best_selection, _, _ = verified_best

    candidates = tuple(
        candidate
        for path in sorted(checkpoint_directory.glob("validation_step_*.pt"))
        if (candidate := _validation_candidate(path, protocol_hash=protocol_hash)).step
        <= completed_steps
    )
    accepted = tuple(candidate for candidate in candidates if candidate.accepted)
    if not accepted:
        raise ValueError("campaign has no accepted numbered validation checkpoint")
    accepted_by_step = {candidate.step: candidate for candidate in accepted}
    accepted_best = accepted_by_step.get(best_step)
    if accepted_best is None:
        raise ValueError("best selector is not represented by an accepted numbered checkpoint")
    if accepted_best.model_state_hash != best_hash or not math.isclose(
        accepted_best.score,
        best_selection.score,
        rel_tol=1.0e-7,
        abs_tol=1.0e-9,
    ):
        raise ValueError("best selector and numbered validation provenance disagree")
    if int(summary.get("best_rollout_validated", False)) != 1:
        raise ValueError("training summary does not report a verified rollout selector")
    summary_best_score = float(summary.get("best_rollout_loss", math.nan))
    if not math.isclose(
        summary_best_score,
        best_selection.score,
        rel_tol=1.0e-7,
        abs_tol=1.0e-9,
    ):
        raise ValueError("training summary and best selector score disagree")

    return CampaignInspection(
        run_directory=str(run_path),
        completed_steps=completed_steps,
        protocol_hash=protocol_hash,
        best_step=best_step,
        best_score=best_selection.score,
        reference_step=reference_step,
        validation_candidates=candidates,
    )


def decide_continuation(
    inspection: CampaignInspection,
    *,
    minimum_total_steps: int,
    extension_steps: int,
    tail_steps: int,
    minimum_relative_gain: float,
    maximum_total_steps: int,
    plateau_validation_count: int = 4,
    validation_interval_steps: int = 512,
) -> ConvergenceDecision:
    """Apply the predeclared tail-improvement rule to safe validation winners."""

    if extension_steps <= 0 or tail_steps <= 0:
        raise ValueError("extension_steps and tail_steps must be positive")
    if plateau_validation_count <= 0 or validation_interval_steps <= 0:
        raise ValueError("plateau validation count and interval must be positive")
    if not 0.0 < minimum_relative_gain < 1.0:
        raise ValueError("minimum_relative_gain must lie in (0, 1)")
    if maximum_total_steps < minimum_total_steps:
        raise ValueError("maximum_total_steps cannot be below the minimum")
    completed = inspection.completed_steps
    if completed < minimum_total_steps:
        return ConvergenceDecision(
            status="incomplete",
            reason="declared minimum training coverage has not completed",
            completed_steps=completed,
            next_total_steps=None,
            best_step=inspection.best_step,
            best_score=inspection.best_score,
            tail_start_step=max(0, completed - tail_steps),
            prior_best_step=None,
            prior_best_score=None,
            tail_best_step=None,
            tail_best_score=None,
            relative_tail_gain=None,
            plateau_candidate_steps=(),
            plateau_candidate_scores=(),
            plateau_candidate_accepted=(),
            plateau_candidate_training_support_passed=(),
            plateau_primary_gain=None,
            minimum_relative_gain=minimum_relative_gain,
            maximum_total_steps=maximum_total_steps,
        )
    tail_start = completed - tail_steps
    prior = [item for item in inspection.accepted_validations if item.step <= tail_start]
    tail = [item for item in inspection.accepted_validations if tail_start < item.step <= completed]
    prior_best = min(prior, key=lambda item: item.score) if prior else None
    tail_best = min(tail, key=lambda item: item.score) if tail else None
    relative_gain = (
        (prior_best.score - tail_best.score) / prior_best.score
        if prior_best is not None and tail_best is not None and prior_best.score > 0
        else None
    )
    plateau_window_steps = plateau_validation_count * validation_interval_steps
    plateau_start = completed - plateau_window_steps
    expected_recent_steps = tuple(
        completed - validation_interval_steps * index
        for index in reversed(range(plateau_validation_count))
    )
    candidates_by_step = {
        candidate.step: candidate for candidate in inspection.validation_candidates
    }
    recent_candidates = tuple(
        candidates_by_step[step] for step in expected_recent_steps if step in candidates_by_step
    )
    plateau_prior = [item for item in inspection.accepted_validations if item.step <= plateau_start]
    plateau_prior_best = min(plateau_prior, key=lambda item: item.score) if plateau_prior else None
    # A support-collapsed candidate can report a deceptively low conditional
    # RMSE by tracking only easy objects. Ordinary supported rejections still
    # supply the raw primary-score evidence required by ADR-059.
    recent_supported_candidates = tuple(
        item for item in recent_candidates if item.training_support_passed
    )
    recent_candidate_best = (
        min(recent_supported_candidates, key=lambda item: item.score)
        if recent_supported_candidates
        else None
    )
    plateau_primary_gain = (
        (plateau_prior_best.score - recent_candidate_best.score) / plateau_prior_best.score
        if plateau_prior_best is not None
        and recent_candidate_best is not None
        and plateau_prior_best.score > 0
        else None
    )

    common = {
        "completed_steps": completed,
        "next_total_steps": None,
        "best_step": inspection.best_step,
        "best_score": inspection.best_score,
        "tail_start_step": tail_start,
        "prior_best_step": None if prior_best is None else prior_best.step,
        "prior_best_score": None if prior_best is None else prior_best.score,
        "tail_best_step": None if tail_best is None else tail_best.step,
        "tail_best_score": None if tail_best is None else tail_best.score,
        "relative_tail_gain": relative_gain,
        "plateau_candidate_steps": tuple(item.step for item in recent_candidates),
        "plateau_candidate_scores": tuple(item.score for item in recent_candidates),
        "plateau_candidate_accepted": tuple(item.accepted for item in recent_candidates),
        "plateau_candidate_training_support_passed": tuple(
            item.training_support_passed for item in recent_candidates
        ),
        "plateau_primary_gain": plateau_primary_gain,
        "minimum_relative_gain": minimum_relative_gain,
        "maximum_total_steps": maximum_total_steps,
    }
    recent_safe_gain = (
        tail_best is not None
        and inspection.best_step > tail_start
        and relative_gain is not None
        and relative_gain >= minimum_relative_gain
    )
    complete_plateau_window = tuple(item.step for item in recent_candidates) == (
        expected_recent_steps
    )
    complete_supported_plateau_window = complete_plateau_window and all(
        item.training_support_passed for item in recent_candidates
    )
    no_recent_acceptance = complete_plateau_window and not any(
        item.accepted for item in recent_candidates
    )
    subthreshold_primary_gain = (
        plateau_primary_gain is not None and plateau_primary_gain < minimum_relative_gain
    )
    if complete_supported_plateau_window and no_recent_acceptance and subthreshold_primary_gain:
        return ConvergenceDecision(
            status="plateau",
            reason=(
                "four consecutive 512-step validations accepted no candidate "
                "and primary-score improvement stayed below threshold"
            ),
            **common,
        )
    if completed >= maximum_total_steps:
        return ConvergenceDecision(
            status="limit_hit",
            reason=(
                "hard training limit reached without demonstrated plateau; "
                "this is a budget stop, not an objective-convergence claim"
            ),
            **common,
        )
    next_steps = completed + extension_steps
    if next_steps > maximum_total_steps:
        return ConvergenceDecision(
            status="limit_hit",
            reason=(
                "another complete extension block would exceed the hard limit "
                "without demonstrated plateau; this is not an "
                "objective-convergence claim"
            ),
            **common,
        )
    if prior_best is None:
        return ConvergenceDecision(
            status="continue",
            reason="plateau evidence is incomplete without a pre-window safe incumbent",
            **{**common, "next_total_steps": next_steps},
        )
    if recent_safe_gain:
        return ConvergenceDecision(
            status="continue",
            reason="best safe checkpoint is recent and exceeds the declared tail gain",
            **{**common, "next_total_steps": next_steps},
        )
    return ConvergenceDecision(
        status="continue",
        reason=(
            "recent-safe extension trigger is absent, but four-point plateau "
            "evidence is incomplete or contradictory"
        ),
        **{**common, "next_total_steps": next_steps},
    )


__all__ = [
    "CampaignIncompleteError",
    "CampaignInspection",
    "ConvergenceDecision",
    "ValidationCandidate",
    "decide_continuation",
    "inspect_completed_campaign",
]
