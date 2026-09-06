"""Durable execution of one specification-1.61 training block.

The qualification coordinator deliberately owns split access and checkpoint
acceptance.  This module owns the missing operation between those transitions:
train from the last accepted 512-update boundary to the next boundary, while
preserving an exact crash-resume state outside the sealed qualification
directory.

Resume uses a two-slot journal.  A complete checkpoint is written to the
inactive slot before the small JSON commit record is atomically replaced.  A
process interruption can therefore lose at most the uncommitted update and can
never make a partial checkpoint authoritative.
"""

from __future__ import annotations

import fcntl
import io
import json
import math
import os
import random
import stat
import subprocess
import tempfile
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, BinaryIO, Literal

import numpy as np
import torch

from world_model.runtime.online_world_model import OnlineWorldModel
from world_model.training.dynamic_set_adapter import DynamicSetEpisodeObjectiveAdapter
from world_model.training.dynamic_set_campaign import (
    DEFAULT_CAMPAIGN,
    DynamicSetTimeProjection,
    LimitHitReason,
    project_minimum_update_feasibility,
)
from world_model.training.dynamic_set_config import OrpheusConfig, load_config
from world_model.training.dynamic_set_evaluation_cache import (
    DEVELOPMENT_EVALUATION_CACHE_DIRECTORY_NAME,
    validate_development_evaluation_cache_directory,
)
from world_model.training.dynamic_set_protocol import (
    FROZEN_PHYSICAL_MANIFEST_SHA256,
    physical_manifest,
)
from world_model.training.dynamic_set_qualification import DynamicSetQualification
from world_model.training.dynamic_set_trainer import (
    CHECKPOINT_SCHEMA,
    DynamicSetTrainer,
    dynamic_set_model_state_sha256,
)
from world_model.training.dynamic_set_training_cache import (
    TRAINING_CACHE_DIRECTORY_NAME,
    DynamicSetTrainingCache,
    DynamicSetTrainingCacheBinding,
    validate_training_cache_directory,
)
from world_model.training.qualification_core import (
    canonical_sha256,
    sha256_bytes,
    validated_sha256,
)

EXECUTION_SCHEMA = "dynamic_set_campaign_execution_v3"
PROGRESS_SCHEMA = "dynamic_set_campaign_progress_v3"
TIMING_EVIDENCE_SCHEMA = "dynamic_set_update_timing_v3"
VALIDATION_TIMING_EVIDENCE_SCHEMA = "dynamic_set_validation_timing_v1"
DEFAULT_SCHEDULE_SEED = 161_061
_PROGRESS_NAME = "progress.json"
_LOCK_NAME = ".execution.lock"
_RESUME_NAMES = ("resume_a.pt", "resume_b.pt")
_ATTEMPT_DIRECTORY_NAMES = ("attempt_01", "attempt_02")
_MAXIMUM_CHECKPOINT_BYTES = 256 * 1024 * 1024
_MAXIMUM_GIT_OUTPUT_BYTES = 1024 * 1024
_ZERO_SHA256 = "0" * 64
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ExecutionStatus = Literal["continue", "limit_hit", "validation_timing_reconciled"]
ArchitectureChoice = Literal["base", "widen_perception", "widen_relation"]


@dataclass(frozen=True, slots=True)
class _ArchitectureExecutionBinding:
    """Qualification-authorized architecture attempt and cumulative carry."""

    architecture_attempt_index: int
    architecture_choice: ArchitectureChoice
    base_config_sha256: str
    resolved_config_sha256: str
    prior_attempt_cumulative_seconds: float
    architecture_attempt_sha256: str

    @classmethod
    def from_qualification(cls, value: Mapping[str, Any]) -> _ArchitectureExecutionBinding:
        expected = {
            "architecture_attempt_index",
            "architecture_choice",
            "base_config_sha256",
            "resolved_config_sha256",
            "prior_attempt_cumulative_seconds",
            "architecture_attempt_sha256",
        }
        if type(value) is not dict or set(value) != expected:
            raise ValueError("architecture execution binding schema differs")
        return cls(
            architecture_attempt_index=value["architecture_attempt_index"],
            architecture_choice=value["architecture_choice"],
            base_config_sha256=value["base_config_sha256"],
            resolved_config_sha256=value["resolved_config_sha256"],
            prior_attempt_cumulative_seconds=value["prior_attempt_cumulative_seconds"],
            architecture_attempt_sha256=value["architecture_attempt_sha256"],
        ).validate()

    def validate(self) -> _ArchitectureExecutionBinding:
        if (
            type(self.architecture_attempt_index) is not int
            or self.architecture_attempt_index not in {1, 2}
            or self.architecture_choice not in {"base", "widen_perception", "widen_relation"}
            or (self.architecture_attempt_index == 1) is not (self.architecture_choice == "base")
        ):
            raise ValueError("architecture execution attempt/choice differs")
        for label, value in (
            ("base configuration", self.base_config_sha256),
            ("resolved configuration", self.resolved_config_sha256),
            ("architecture attempt", self.architecture_attempt_sha256),
        ):
            validated_sha256(value, label=label)
        if self.architecture_attempt_sha256 == _ZERO_SHA256:
            raise ValueError("architecture attempt requires a nonzero ledger digest")
        if (
            type(self.prior_attempt_cumulative_seconds) is not float
            or not math.isfinite(self.prior_attempt_cumulative_seconds)
            or self.prior_attempt_cumulative_seconds < 0.0
            or (
                self.architecture_attempt_index == 1
                and self.prior_attempt_cumulative_seconds != 0.0
            )
            or (
                self.architecture_attempt_index == 1
                and self.resolved_config_sha256 != self.base_config_sha256
            )
            or (
                self.architecture_attempt_index == 2
                and self.resolved_config_sha256 == self.base_config_sha256
            )
        ):
            raise ValueError("architecture execution configuration/carry differs")
        return self

    def fields(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


_ARCHITECTURE_PROGRESS_FIELDS = frozenset(
    {
        "architecture_attempt_index",
        "architecture_choice",
        "base_config_sha256",
        "resolved_config_sha256",
        "prior_attempt_cumulative_seconds",
        "architecture_attempt_sha256",
    }
)


def _architecture_from_progress(
    progress: Mapping[str, Any],
) -> _ArchitectureExecutionBinding:
    if _ARCHITECTURE_PROGRESS_FIELDS - set(progress):
        raise ValueError("progress lacks its architecture execution binding")
    return _ArchitectureExecutionBinding(
        **{name: progress[name] for name in _ARCHITECTURE_PROGRESS_FIELDS}
    ).validate()


@dataclass(frozen=True)
class DynamicSetCampaignExecutionReport:
    protocol_sha256: str
    architecture_attempt_index: int
    architecture_choice: ArchitectureChoice
    base_config_sha256: str
    resolved_config_sha256: str
    prior_attempt_cumulative_seconds: float
    architecture_attempt_sha256: str
    status: ExecutionStatus
    completed_updates: int
    target_updates: int
    updates_executed: int
    validation_timing_reconciled: bool
    boundary_reached: bool
    stopped_for_time_limit: bool
    checkpoint_path: str
    checkpoint_sha256: str
    model_state_sha256: str
    progress_path: str
    progress_record_sha256: str
    validated_candidate_count: int
    validated_boundary_updates: int
    validated_candidate_sha256: str
    validated_checkpoint_sha256: str
    validated_model_state_sha256: str
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
    limit_hit_reason: LimitHitReason
    block_training_seconds: float
    rejected_update_count: int

    def to_dict(self) -> dict[str, Any]:
        return {"schema": EXECUTION_SCHEMA, **asdict(self)}


@dataclass(frozen=True, slots=True)
class _UpdateTimingEvidence:
    """Exact bounded measurements committed inside every resume checkpoint."""

    architecture_binding: _ArchitectureExecutionBinding
    completed_update_seconds: tuple[float, ...] = ()
    discarded_attempt_seconds: float = 0.0
    screen_wall_seconds: float = 0.0
    completed_validation_seconds: tuple[float, ...] = ()
    validation_timing_sha256: str = _ZERO_SHA256

    def validate(self, *, completed_updates: int | None = None) -> _UpdateTimingEvidence:
        self.architecture_binding.validate()
        if len(self.completed_update_seconds) > DEFAULT_CAMPAIGN.maximum_updates:
            raise ValueError("update timing evidence exceeds the campaign bound")
        if (
            completed_updates is not None
            and len(self.completed_update_seconds) != completed_updates
        ):
            raise ValueError("update timing count differs from the checkpoint cursor")
        for value in self.completed_update_seconds:
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise ValueError("completed-update timing evidence is invalid")
        for label, value in (
            ("discarded-attempt", self.discarded_attempt_seconds),
            ("screen-wall", self.screen_wall_seconds),
        ):
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{label} timing evidence is invalid")
        maximum_validations = (
            len(self.completed_update_seconds) // DEFAULT_CAMPAIGN.validation_interval_updates
        )
        if len(self.completed_validation_seconds) > maximum_validations:
            raise ValueError("validation timings exceed completed validation boundaries")
        for value in self.completed_validation_seconds:
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise ValueError("completed-validation timing evidence is invalid")
        validated_sha256(
            self.validation_timing_sha256,
            label="execution validation timing",
        )
        if bool(self.completed_validation_seconds) is (
            self.validation_timing_sha256 == _ZERO_SHA256
        ):
            raise ValueError("validation timing digest presence differs from its samples")
        return self

    @property
    def cumulative_validation_seconds(self) -> float:
        return float(math.fsum(self.completed_validation_seconds))

    @property
    def cumulative_training_seconds(self) -> float:
        return float(
            math.fsum(
                (
                    self.architecture_binding.prior_attempt_cumulative_seconds,
                    self.screen_wall_seconds,
                    *self.completed_update_seconds,
                    self.discarded_attempt_seconds,
                    *self.completed_validation_seconds,
                )
            )
        )

    @property
    def sha256(self) -> str:
        return canonical_sha256(self._body())

    def _body(self) -> dict[str, Any]:
        return {
            "schema": TIMING_EVIDENCE_SCHEMA,
            **self.architecture_binding.fields(),
            "completed_update_seconds": list(self.completed_update_seconds),
            "discarded_attempt_seconds": self.discarded_attempt_seconds,
            "screen_wall_seconds": self.screen_wall_seconds,
            "completed_validation_seconds": list(self.completed_validation_seconds),
            "validation_timing_sha256": self.validation_timing_sha256,
            "cumulative_training_seconds": self.cumulative_training_seconds,
        }

    def to_payload(self) -> dict[str, Any]:
        body = self._body()
        return {**body, "evidence_sha256": canonical_sha256(body)}

    def with_completed_update(self, seconds: float) -> _UpdateTimingEvidence:
        value = _validated_duration(seconds)
        return _UpdateTimingEvidence(
            architecture_binding=self.architecture_binding,
            completed_update_seconds=(*self.completed_update_seconds, value),
            discarded_attempt_seconds=self.discarded_attempt_seconds,
            screen_wall_seconds=self.screen_wall_seconds,
            completed_validation_seconds=self.completed_validation_seconds,
            validation_timing_sha256=self.validation_timing_sha256,
        ).validate()

    def with_discarded_attempt(self, seconds: float) -> _UpdateTimingEvidence:
        value = _validated_duration(seconds)
        return _UpdateTimingEvidence(
            architecture_binding=self.architecture_binding,
            completed_update_seconds=self.completed_update_seconds,
            discarded_attempt_seconds=float(math.fsum((self.discarded_attempt_seconds, value))),
            screen_wall_seconds=self.screen_wall_seconds,
            completed_validation_seconds=self.completed_validation_seconds,
            validation_timing_sha256=self.validation_timing_sha256,
        ).validate()

    def with_completed_validation(
        self,
        seconds: float,
        *,
        validation_timing_sha256: str,
    ) -> _UpdateTimingEvidence:
        value = _validated_duration(seconds)
        validated_sha256(validation_timing_sha256, label="candidate validation timing")
        if validation_timing_sha256 == _ZERO_SHA256:
            raise ValueError("completed validation timing requires a nonzero digest")
        return _UpdateTimingEvidence(
            architecture_binding=self.architecture_binding,
            completed_update_seconds=self.completed_update_seconds,
            discarded_attempt_seconds=self.discarded_attempt_seconds,
            screen_wall_seconds=self.screen_wall_seconds,
            completed_validation_seconds=(*self.completed_validation_seconds, value),
            validation_timing_sha256=validation_timing_sha256,
        ).validate()


def _validated_duration(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("measured update duration must be one finite scalar")
    duration = float(value)
    if not math.isfinite(duration) or duration < 0.0:
        raise ValueError("measured update duration must be finite and nonnegative")
    return duration


def _timing_from_payload(
    payload: Mapping[str, Any], *, completed_updates: int
) -> _UpdateTimingEvidence:
    raw = payload.get("execution_timing")
    expected_keys = {
        "schema",
        *_ARCHITECTURE_PROGRESS_FIELDS,
        "completed_update_seconds",
        "discarded_attempt_seconds",
        "screen_wall_seconds",
        "completed_validation_seconds",
        "validation_timing_sha256",
        "cumulative_training_seconds",
        "evidence_sha256",
    }
    if type(raw) is not dict or set(raw) != expected_keys:
        raise ValueError("dynamic-set checkpoint timing schema differs")
    body = {key: value for key, value in raw.items() if key != "evidence_sha256"}
    if (
        raw["schema"] != TIMING_EVIDENCE_SCHEMA
        or raw["evidence_sha256"] != canonical_sha256(body)
        or type(raw["completed_update_seconds"]) is not list
        or type(raw["completed_validation_seconds"]) is not list
    ):
        raise ValueError("dynamic-set checkpoint timing binding differs")
    architecture_binding = _architecture_from_progress(raw)
    evidence = _UpdateTimingEvidence(
        architecture_binding=architecture_binding,
        completed_update_seconds=tuple(raw["completed_update_seconds"]),
        discarded_attempt_seconds=raw["discarded_attempt_seconds"],
        screen_wall_seconds=raw["screen_wall_seconds"],
        completed_validation_seconds=tuple(raw["completed_validation_seconds"]),
        validation_timing_sha256=raw["validation_timing_sha256"],
    ).validate(completed_updates=completed_updates)
    if (
        type(raw["cumulative_training_seconds"]) is not float
        or raw["cumulative_training_seconds"] != evidence.cumulative_training_seconds
    ):
        raise ValueError("dynamic-set checkpoint cumulative timing differs")
    return evidence


def _checkpoint_with_timing(
    payload: Mapping[str, Any], timing: _UpdateTimingEvidence
) -> dict[str, Any]:
    completed, _ = _checkpoint_cursor(payload)
    timing.validate(completed_updates=completed)
    value = dict(payload)
    value["execution_timing"] = timing.to_payload()
    return value


def _trainer_checkpoint_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Remove executor-owned evidence before the trainer's exact-schema load."""

    value = dict(payload)
    if "execution_timing" not in value:
        raise ValueError("resume checkpoint lacks executor timing evidence")
    del value["execution_timing"]
    return value


def _time_projection(timing: _UpdateTimingEvidence) -> DynamicSetTimeProjection:
    timing.validate()
    return project_minimum_update_feasibility(
        completed_update_seconds=timing.completed_update_seconds,
        discarded_attempt_seconds=timing.discarded_attempt_seconds,
        prior_attempt_cumulative_seconds=(
            timing.architecture_binding.prior_attempt_cumulative_seconds
        ),
        screen_wall_seconds=timing.screen_wall_seconds,
        completed_validation_seconds=timing.completed_validation_seconds,
        config=DEFAULT_CAMPAIGN,
    )


def _timing_progress_fields(timing: _UpdateTimingEvidence) -> dict[str, Any]:
    projection = _time_projection(timing)
    return {
        **timing.architecture_binding.fields(),
        "cumulative_training_seconds": timing.cumulative_training_seconds,
        "screen_wall_seconds": timing.screen_wall_seconds,
        "completed_update_timing_count": len(timing.completed_update_seconds),
        "timing_evidence_sha256": timing.sha256,
        "discarded_attempt_seconds": timing.discarded_attempt_seconds,
        "execution_validation_timing_count": len(timing.completed_validation_seconds),
        "execution_cumulative_validation_seconds": timing.cumulative_validation_seconds,
        "execution_validation_timing_sha256": timing.validation_timing_sha256,
        "campaign_envelope_limit_seconds": projection.envelope_limit_seconds,
        "training_mutation_limit_seconds": projection.mutation_limit_seconds,
        "reserved_audit_seconds": DEFAULT_CAMPAIGN.reserved_audit_seconds,
        "projection_support_satisfied": projection.support_satisfied,
        "conservative_update_seconds": projection.conservative_update_seconds,
        "conservative_validation_seconds": projection.conservative_validation_seconds,
        "projected_remaining_validation_seconds": (
            projection.projected_remaining_validation_seconds
        ),
        "projected_minimum_training_seconds": (projection.projected_minimum_training_seconds),
        "projected_minimum_envelope_seconds": (projection.projected_minimum_envelope_seconds),
        "minimum_update_feasible": projection.minimum_update_feasible,
        "training_limit_reached": projection.limit_hit,
        "execution_status": "limit_hit" if projection.limit_hit else "continue",
        "limit_hit_reason": projection.limit_hit_reason,
    }


@dataclass(frozen=True, slots=True)
class _ValidatedCandidateLineage:
    candidate_count: int
    boundary_updates: int
    candidate_sha256: str
    checkpoint_sha256: str
    model_state_sha256: str

    def validate(self) -> _ValidatedCandidateLineage:
        if (
            isinstance(self.candidate_count, bool)
            or not isinstance(self.candidate_count, int)
            or self.candidate_count < 0
            or isinstance(self.boundary_updates, bool)
            or not isinstance(self.boundary_updates, int)
            or self.boundary_updates
            != self.candidate_count * DEFAULT_CAMPAIGN.validation_interval_updates
        ):
            raise ValueError("validated candidate lineage cursor differs")
        hashes = (
            self.candidate_sha256,
            self.checkpoint_sha256,
            self.model_state_sha256,
        )
        if self.candidate_count == 0:
            if hashes != (_ZERO_SHA256, _ZERO_SHA256, _ZERO_SHA256):
                raise ValueError("empty candidate lineage must use zero digests")
        else:
            for label, value in zip(
                ("candidate", "checkpoint", "model state"), hashes, strict=True
            ):
                validated_sha256(value, label=f"validated lineage {label}")
        return self


_EMPTY_LINEAGE = _ValidatedCandidateLineage(
    candidate_count=0,
    boundary_updates=0,
    candidate_sha256=_ZERO_SHA256,
    checkpoint_sha256=_ZERO_SHA256,
    model_state_sha256=_ZERO_SHA256,
)


def _atomic_write_bytes(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise OSError(f"refusing to replace non-regular path {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"temporary execution path already exists: {temporary}")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def _checkpoint_bytes(payload: Mapping[str, Any]) -> bytes:
    stream = io.BytesIO()
    torch.save(dict(payload), stream)
    contents = stream.getvalue()
    if not contents or len(contents) > _MAXIMUM_CHECKPOINT_BYTES:
        raise ValueError("dynamic-set checkpoint lies outside the persistence byte bound")
    return contents


def _safe_checkpoint_payload(contents: bytes) -> dict[str, Any]:
    if not contents or len(contents) > _MAXIMUM_CHECKPOINT_BYTES:
        raise ValueError("dynamic-set checkpoint lies outside the persistence byte bound")
    try:
        value = torch.load(io.BytesIO(contents), map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("dynamic-set resume checkpoint is not a safe tensor payload") from error
    if type(value) is not dict or value.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("dynamic-set resume checkpoint schema differs")
    model_state = value.get("model_state")
    if not isinstance(model_state, Mapping):
        raise ValueError("dynamic-set resume checkpoint lacks model state")
    expected = value.get("model_state_sha256")
    actual = dynamic_set_model_state_sha256(model_state)
    if expected != actual:
        raise ValueError("dynamic-set resume model-state digest differs")
    trainer_state = value.get("trainer_state")
    next_sample = value.get("next_sample_state")
    if not isinstance(trainer_state, Mapping) or not isinstance(next_sample, Mapping):
        raise ValueError("dynamic-set resume checkpoint lacks continuation state")
    completed = trainer_state.get("completed_updates")
    if (
        isinstance(completed, bool)
        or not isinstance(completed, int)
        or completed < 0
        or next_sample.get("absolute_update_index") != completed
    ):
        raise ValueError("dynamic-set resume update cursor differs")
    _timing_from_payload(value, completed_updates=completed)
    return value


def _stable_regular_bytes(path: Path) -> bytes:
    resolved = Path(path).absolute()
    before = os.lstat(resolved)
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size <= 0
        or before.st_size > _MAXIMUM_CHECKPOINT_BYTES
    ):
        raise OSError(f"execution input is not one bounded single-link regular file: {resolved}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(resolved, flags)
    try:
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        remaining = _MAXIMUM_CHECKPOINT_BYTES + 1
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
        (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_nlink)
        for item in (before, opened, after, final)
    }
    if len(identities) != 1:
        raise OSError(f"execution input changed during stable read: {resolved}")
    contents = b"".join(chunks)
    if len(contents) != final.st_size or len(contents) > _MAXIMUM_CHECKPOINT_BYTES:
        raise OSError(f"execution input differs from its bounded metadata: {resolved}")
    return contents


def _load_captured_config(contents: bytes, *, source_path: Path) -> OrpheusConfig:
    """Parse exactly the already-hashed bytes, never a second read of the source path."""

    if type(contents) is not bytes or not contents:
        raise ValueError("captured execution config must be nonempty exact bytes")
    with tempfile.TemporaryDirectory(prefix="orpheus-dynamic-set-config-") as directory:
        path = Path(directory) / "captured.yaml"
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            view = memoryview(contents)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("captured config write made no progress")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return replace(load_config(path), source_path=str(source_path))


def _git_bytes(repository_root: Path, arguments: tuple[str, ...], *, label: str) -> bytes:
    environment = dict(os.environ)
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    try:
        result = subprocess.run(
            ("git", *arguments),
            cwd=repository_root,
            env=environment,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=True,
            timeout=30.0,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise PermissionError(f"cannot authenticate dynamic-set {label}") from error
    if len(result.stdout) > _MAXIMUM_GIT_OUTPUT_BYTES:
        raise PermissionError(f"dynamic-set {label} exceeds its output bound")
    return result.stdout


def _git_identifier(repository_root: Path, revision: str, *, label: str) -> str:
    raw = _git_bytes(
        repository_root,
        ("rev-parse", "--verify", revision),
        label=label,
    )
    if raw.count(b"\n") != 1 or not raw.endswith(b"\n"):
        raise PermissionError(f"dynamic-set {label} has invalid Git output framing")
    try:
        value = raw[:-1].decode("ascii", errors="strict")
    except UnicodeDecodeError as error:
        raise PermissionError(f"dynamic-set {label} is not ASCII") from error
    if len(value) not in {40, 64} or value.lower() != value:
        raise PermissionError(f"dynamic-set {label} is not one Git object identifier")
    try:
        int(value, 16)
    except ValueError as error:
        raise PermissionError(f"dynamic-set {label} is not hexadecimal") from error
    return value


def _authenticate_current_source(
    source: Mapping[str, Any],
    *,
    repository_root: Path = _REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Fail closed unless the executing checkout is the exact clean published freeze."""

    required = {
        "source_sha256",
        "commit",
        "tree",
        "upstream_commit",
        "clean",
        "published",
    }
    if type(source) is not dict or set(source) != required:
        raise ValueError("qualification source binding schema differs at execution")
    root = Path(repository_root).resolve(strict=True)
    status_before = _git_bytes(
        root,
        ("status", "--porcelain=v1", "-z", "--untracked-files=all"),
        label="worktree status",
    )
    commit = _git_identifier(root, "HEAD", label="HEAD")
    tree = _git_identifier(root, "HEAD^{tree}", label="HEAD tree")
    upstream = _git_identifier(root, "@{upstream}", label="upstream")
    divergence = _git_bytes(
        root,
        ("rev-list", "--left-right", "--count", "HEAD...@{upstream}"),
        label="upstream divergence",
    )
    status_after = _git_bytes(
        root,
        ("status", "--porcelain=v1", "-z", "--untracked-files=all"),
        label="final worktree status",
    )
    final = (
        _git_identifier(root, "HEAD", label="final HEAD"),
        _git_identifier(root, "HEAD^{tree}", label="final HEAD tree"),
        _git_identifier(root, "@{upstream}", label="final upstream"),
    )
    if (
        status_before
        or status_after
        or divergence != b"0\t0\n"
        or final != (commit, tree, upstream)
        or source["clean"] is not True
        or source["published"] is not True
        or source["commit"] != commit
        or source["tree"] != tree
        or source["upstream_commit"] != upstream
    ):
        raise PermissionError("executing source differs from the clean published source freeze")
    validated_sha256(source["source_sha256"], label="dynamic-set source")
    return {
        "commit": commit,
        "tree": tree,
        "upstream_commit": upstream,
        "clean": True,
        "published": True,
    }


def _progress_record(body: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(body)
    value["record_sha256"] = canonical_sha256(value)
    return value


def _write_progress(path: Path, body: Mapping[str, Any]) -> dict[str, Any]:
    record = _progress_record(body)
    contents = (
        json.dumps(
            record,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    _atomic_write_bytes(path, contents)
    return record


def _read_progress(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(_stable_regular_bytes(path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("dynamic-set progress is not strict JSON") from error
    if type(value) is not dict or value.get("schema") != PROGRESS_SCHEMA:
        raise ValueError("dynamic-set progress schema differs")
    supplied = value.get("record_sha256")
    body = {key: item for key, item in value.items() if key != "record_sha256"}
    if supplied != canonical_sha256(body):
        raise ValueError("dynamic-set progress digest differs")
    expected_keys = {
        "schema",
        "protocol_sha256",
        "config_sha256",
        "source_sha256",
        *_ARCHITECTURE_PROGRESS_FIELDS,
        "active_resume_name",
        "checkpoint_sha256",
        "model_state_sha256",
        "completed_updates",
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
    if set(value) != expected_keys:
        raise ValueError("dynamic-set progress fields differ")
    if value["active_resume_name"] not in _RESUME_NAMES:
        raise ValueError("dynamic-set progress names an invalid resume slot")
    completed = value["completed_updates"]
    architecture = _architecture_from_progress(value)
    elapsed = value["cumulative_training_seconds"]
    screen_wall = value["screen_wall_seconds"]
    timing_count = value["completed_update_timing_count"]
    discarded = value["discarded_attempt_seconds"]
    validation_timing_count = value["execution_validation_timing_count"]
    cumulative_validation = value["execution_cumulative_validation_seconds"]
    validation_timing_sha256 = value["execution_validation_timing_sha256"]
    envelope_limit = value["campaign_envelope_limit_seconds"]
    mutation_limit = value["training_mutation_limit_seconds"]
    reserve = value["reserved_audit_seconds"]
    support = value["projection_support_satisfied"]
    conservative = value["conservative_update_seconds"]
    conservative_validation = value["conservative_validation_seconds"]
    projected_remaining_validation = value["projected_remaining_validation_seconds"]
    projected = value["projected_minimum_training_seconds"]
    projected_envelope = value["projected_minimum_envelope_seconds"]
    feasible = value["minimum_update_feasible"]
    rejected = value["rejected_update_count"]
    limit_reached = value["training_limit_reached"]
    execution_status = value["execution_status"]
    limit_reason = value["limit_hit_reason"]
    remaining_validations = max(
        0,
        DEFAULT_CAMPAIGN.minimum_updates // DEFAULT_CAMPAIGN.validation_interval_updates
        - validation_timing_count
        if type(validation_timing_count) is int
        else 0,
    )
    expected_support = (
        type(completed) is int
        and type(validation_timing_count) is int
        and completed >= DEFAULT_CAMPAIGN.minimum_timing_support_updates
        and (validation_timing_count > 0 or remaining_validations == 0)
    )
    if (
        isinstance(completed, bool)
        or not isinstance(completed, int)
        or completed < 0
        or type(elapsed) is not float
        or not math.isfinite(elapsed)
        or elapsed < 0.0
        or type(screen_wall) is not float
        or not math.isfinite(screen_wall)
        or screen_wall < 0.0
        or type(timing_count) is not int
        or timing_count != completed
        or type(discarded) is not float
        or not math.isfinite(discarded)
        or discarded < 0.0
        or type(validation_timing_count) is not int
        or not 0
        <= validation_timing_count
        <= completed // DEFAULT_CAMPAIGN.validation_interval_updates
        or validation_timing_count != value["validated_candidate_count"]
        or type(cumulative_validation) is not float
        or not math.isfinite(cumulative_validation)
        or cumulative_validation < 0.0
        or elapsed
        < math.fsum(
            (
                architecture.prior_attempt_cumulative_seconds,
                screen_wall,
                discarded,
                cumulative_validation,
            )
        )
        or (validation_timing_count == 0) is (validation_timing_sha256 != _ZERO_SHA256)
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
            projected is not None
            and projected_envelope
            != float(float(projected) + DEFAULT_CAMPAIGN.reserved_audit_seconds)
        )
        or (
            projected_envelope is not None
            and feasible
            is not (float(projected_envelope) <= DEFAULT_CAMPAIGN.maximum_training_hours * 3600.0)
        )
        or (
            support
            and conservative_validation is not None
            and projected_remaining_validation is not None
            and projected_remaining_validation
            != float(conservative_validation * remaining_validations)
        )
        or (
            support
            and conservative is not None
            and projected_remaining_validation is not None
            and projected is not None
            and projected
            != float(
                elapsed
                + conservative * max(0, DEFAULT_CAMPAIGN.minimum_updates - completed)
                + projected_remaining_validation
            )
        )
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
        or isinstance(rejected, bool)
        or not isinstance(rejected, int)
        or rejected < 0
        or type(limit_reached) is not bool
        or type(execution_status) is not str
        or execution_status not in {"continue", "limit_hit"}
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
            and elapsed < DEFAULT_CAMPAIGN.training_mutation_seconds
        )
        or (
            limit_reason == "minimum_update_projection_infeasible"
            and (
                elapsed >= DEFAULT_CAMPAIGN.training_mutation_seconds
                or not support
                or feasible is not False
                or (completed >= DEFAULT_CAMPAIGN.minimum_updates and remaining_validations == 0)
            )
        )
        or (limit_reason == "none" and elapsed >= DEFAULT_CAMPAIGN.training_mutation_seconds)
    ):
        raise ValueError("dynamic-set progress counters are invalid")
    validated_sha256(
        value["timing_evidence_sha256"],
        label="update timing evidence",
    )
    validated_sha256(
        value["execution_validation_timing_sha256"],
        label="execution validation timing evidence",
    )
    _lineage_from_progress(value)
    return value


def _lineage_from_progress(progress: Mapping[str, Any]) -> _ValidatedCandidateLineage:
    return _ValidatedCandidateLineage(
        candidate_count=progress["validated_candidate_count"],
        boundary_updates=progress["validated_boundary_updates"],
        candidate_sha256=progress["validated_candidate_sha256"],
        checkpoint_sha256=progress["validated_checkpoint_sha256"],
        model_state_sha256=progress["validated_model_state_sha256"],
    ).validate()


def _validation_timing_from_candidates(
    candidates: list[dict[str, Any]],
) -> tuple[tuple[float, ...], str]:
    """Replay the qualification-sealed validation timing chain exactly."""

    durations: list[float] = []
    previous_sha256 = _ZERO_SHA256
    required = {
        "sequence",
        "completed_updates",
        "checkpoint_sha256",
        "model_state_sha256",
        "execution_progress_record_sha256",
        "callback_binding_sha256",
        "validation_wall_seconds",
        "validation_timing_count",
        "cumulative_validation_wall_seconds",
        "previous_validation_timing_sha256",
        "validation_timing_sha256",
    }
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping) or not required <= set(candidate):
            raise ValueError("validation candidate lacks its timing-chain binding")
        completed_updates = (index + 1) * DEFAULT_CAMPAIGN.validation_interval_updates
        duration = candidate["validation_wall_seconds"]
        if type(duration) is not float or not math.isfinite(duration) or duration < 0.0:
            raise ValueError("candidate validation duration is invalid")
        durations.append(duration)
        cumulative = float(math.fsum(durations))
        body = {
            "schema": VALIDATION_TIMING_EVIDENCE_SCHEMA,
            "sequence": index,
            "completed_updates": candidate["completed_updates"],
            "checkpoint_sha256": candidate["checkpoint_sha256"],
            "model_state_sha256": candidate["model_state_sha256"],
            "execution_progress_record_sha256": candidate["execution_progress_record_sha256"],
            "callback_binding_sha256": candidate["callback_binding_sha256"],
            "validation_wall_seconds": duration,
            "validation_timing_count": candidate["validation_timing_count"],
            "cumulative_validation_wall_seconds": candidate["cumulative_validation_wall_seconds"],
            "previous_validation_timing_sha256": candidate["previous_validation_timing_sha256"],
        }
        for name in (
            "checkpoint_sha256",
            "model_state_sha256",
            "execution_progress_record_sha256",
            "callback_binding_sha256",
            "previous_validation_timing_sha256",
            "validation_timing_sha256",
        ):
            validated_sha256(candidate[name], label=f"candidate {name}")
        if (
            candidate["sequence"] != index
            or candidate["completed_updates"] != completed_updates
            or type(candidate["validation_timing_count"]) is not int
            or candidate["validation_timing_count"] != index + 1
            or type(candidate["cumulative_validation_wall_seconds"]) is not float
            or candidate["cumulative_validation_wall_seconds"] != cumulative
            or candidate["previous_validation_timing_sha256"] != previous_sha256
            or candidate["validation_timing_sha256"] != canonical_sha256(body)
        ):
            raise ValueError("candidate validation timing chain differs")
        previous_sha256 = candidate["validation_timing_sha256"]
    return tuple(durations), previous_sha256


def _candidate_lineage(
    *,
    candidates: list[dict[str, Any]],
    work_directory: Path,
) -> _ValidatedCandidateLineage:
    if not candidates:
        return _EMPTY_LINEAGE
    count = len(candidates)
    candidate = candidates[-1]
    expected_updates = count * DEFAULT_CAMPAIGN.validation_interval_updates
    required = {
        "completed_updates",
        "candidate_sha256",
        "checkpoint_sha256",
        "model_state_sha256",
    }
    if not isinstance(candidate, Mapping) or not required <= set(candidate):
        raise ValueError("latest validation candidate lacks execution lineage fields")
    if candidate["completed_updates"] != expected_updates:
        raise ValueError("latest validation candidate boundary differs")
    lineage = _ValidatedCandidateLineage(
        candidate_count=count,
        boundary_updates=expected_updates,
        candidate_sha256=candidate["candidate_sha256"],
        checkpoint_sha256=candidate["checkpoint_sha256"],
        model_state_sha256=candidate["model_state_sha256"],
    ).validate()
    boundary_path = work_directory / f"update_{expected_updates:06d}.pt"
    contents = _stable_regular_bytes(boundary_path)
    if sha256_bytes(contents) != lineage.checkpoint_sha256:
        raise ValueError("validated candidate differs from the campaign boundary checkpoint")
    payload = _safe_checkpoint_payload(contents)
    completed, _ = _checkpoint_cursor(payload)
    if (
        completed != lineage.boundary_updates
        or payload["model_state_sha256"] != lineage.model_state_sha256
    ):
        raise ValueError("validated candidate model/cursor differs from its boundary checkpoint")
    return lineage


def _validate_attempt_work_directory(path: Path) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise OSError("campaign attempt path must be one real directory")
    allowed_exact = {*_RESUME_NAMES, _PROGRESS_NAME}
    for item in path.iterdir():
        allowed = item.name in allowed_exact
        if item.name.startswith("update_") and item.name.endswith(".pt"):
            raw_update = item.name.removeprefix("update_").removesuffix(".pt")
            allowed = (
                len(raw_update) == 6
                and raw_update.isascii()
                and raw_update.isdigit()
                and int(raw_update) > 0
                and int(raw_update) <= DEFAULT_CAMPAIGN.maximum_updates
                and int(raw_update) % DEFAULT_CAMPAIGN.validation_interval_updates == 0
            )
        if allowed:
            metadata = os.lstat(item)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise OSError(
                    f"campaign attempt artifact must be one single-link regular file: {item}"
                )
            continue
        if item.name.startswith(".") and item.name.endswith(".tmp"):
            raise OSError(f"incomplete temporary execution artifact requires review: {item}")
        raise OSError(f"campaign attempt directory contains unsupported artifact: {item.name}")
    return path


def _attempt_work_directory(
    root: Path,
    *,
    attempt_index: int,
    create: bool,
) -> Path:
    if type(attempt_index) is not int or attempt_index not in {1, 2}:
        raise ValueError("architecture attempt index is outside the frozen bound")
    path = root / _ATTEMPT_DIRECTORY_NAMES[attempt_index - 1]
    if not path.exists():
        if not create:
            raise FileNotFoundError(f"campaign attempt directory is absent: {path.name}")
        path.mkdir(mode=0o700)
    return _validate_attempt_work_directory(path)


def _assert_dedicated_work_directory(path: Path, qualification_root: Path) -> Path:
    if path.is_symlink():
        raise OSError("campaign work path must not be a symbolic link")
    resolved = path.resolve()
    qualification = qualification_root.resolve()
    if (
        resolved == qualification
        or resolved in qualification.parents
        or qualification in resolved.parents
    ):
        raise ValueError(
            "campaign work directory must not overlap the sealed qualification directory"
        )
    if resolved.exists() and not resolved.is_dir():
        raise OSError("campaign work path must be one real directory")
    resolved.mkdir(parents=True, exist_ok=True)
    for item in resolved.iterdir():
        if item.name == DEVELOPMENT_EVALUATION_CACHE_DIRECTORY_NAME:
            metadata = os.lstat(item)
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise OSError("campaign development cache must be one real directory")
            validate_development_evaluation_cache_directory(item)
            continue
        if item.name == TRAINING_CACHE_DIRECTORY_NAME:
            metadata = os.lstat(item)
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise OSError("campaign training cache must be one real directory")
            validate_training_cache_directory(item)
            continue
        if item.name in _ATTEMPT_DIRECTORY_NAMES:
            metadata = os.lstat(item)
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise OSError("campaign attempt path must be one real directory")
            _validate_attempt_work_directory(item)
            continue
        if item.name == _LOCK_NAME:
            metadata = os.lstat(item)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise OSError("campaign execution lock must be one single-link regular file")
            continue
        if item.name.startswith(".") and item.name.endswith(".tmp"):
            raise OSError(f"incomplete temporary execution artifact requires review: {item}")
        raise OSError(f"campaign work directory contains unsupported artifact: {item.name}")
    return resolved


def _seed_fresh_process(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("schedule seed must be an integer")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def _fresh_repository_trainer(
    *,
    config: OrpheusConfig,
    source: Mapping[str, Any],
    schedule_seed: int,
    training_cache: DynamicSetTrainingCache,
) -> DynamicSetTrainer:
    if not isinstance(training_cache, DynamicSetTrainingCache):
        raise TypeError("repository training requires its authenticated lean cache")
    model = OnlineWorldModel.from_config(config, device="cpu")
    return DynamicSetTrainer.from_online_world_model(
        model=model,
        training_rows=physical_manifest("training"),
        objective_adapter=DynamicSetEpisodeObjectiveAdapter(),
        resolved_config=config,
        source_provenance=source,
        schedule_seed=schedule_seed,
        materializer=training_cache,
    )


def _checkpoint_cursor(payload: Mapping[str, Any]) -> tuple[int, int]:
    trainer_state = payload.get("trainer_state")
    next_sample = payload.get("next_sample_state")
    if not isinstance(trainer_state, Mapping) or not isinstance(next_sample, Mapping):
        raise ValueError("dynamic-set checkpoint lacks its exact cursor mappings")
    completed = trainer_state["completed_updates"]
    seed = next_sample["schedule_seed"]
    if type(completed) is not int or type(seed) is not int:
        raise ValueError("dynamic-set checkpoint cursor types differ")
    if seed != DEFAULT_SCHEDULE_SEED:
        raise ValueError("dynamic-set checkpoint schedule seed differs from the frozen campaign")
    return completed, seed


def _journal_commit(
    *,
    work_directory: Path,
    payload: Mapping[str, Any],
    previous_progress: Mapping[str, Any] | None,
    protocol_sha256: str,
    config_sha256: str,
    source_sha256: str,
    timing: _UpdateTimingEvidence,
    rejected_update_count: int,
    lineage: _ValidatedCandidateLineage = _EMPTY_LINEAGE,
) -> tuple[dict[str, Any], bytes]:
    lineage.validate()
    completed, _ = _checkpoint_cursor(payload)
    timing.validate(completed_updates=completed)
    trainer_state = payload.get("trainer_state")
    if (
        not isinstance(trainer_state, Mapping)
        or isinstance(rejected_update_count, bool)
        or not isinstance(rejected_update_count, int)
        or rejected_update_count < 0
        or trainer_state.get("rejected_update_count") != rejected_update_count
        or lineage.boundary_updates > completed
        or len(timing.completed_validation_seconds) != lineage.candidate_count
    ):
        raise ValueError("journal trainer counters/lineage differ from its checkpoint")
    if previous_progress is not None:
        previous_contents = _stable_regular_bytes(
            work_directory / previous_progress["active_resume_name"]
        )
        if sha256_bytes(previous_contents) != previous_progress["checkpoint_sha256"]:
            raise ValueError("previous journal checkpoint digest differs")
        previous_payload = _safe_checkpoint_payload(previous_contents)
        previous_completed, _ = _checkpoint_cursor(previous_payload)
        previous_timing = _timing_from_payload(
            previous_payload,
            completed_updates=previous_completed,
        )
        if (
            previous_completed != previous_progress["completed_updates"]
            or any(
                previous_progress[name] != value
                for name, value in _timing_progress_fields(previous_timing).items()
            )
            or completed not in {previous_completed, previous_completed + 1}
            or timing.completed_update_seconds[:previous_completed]
            != previous_timing.completed_update_seconds
            or timing.screen_wall_seconds != previous_timing.screen_wall_seconds
            or timing.completed_validation_seconds[
                : len(previous_timing.completed_validation_seconds)
            ]
            != previous_timing.completed_validation_seconds
            or len(timing.completed_validation_seconds)
            not in {
                len(previous_timing.completed_validation_seconds),
                len(previous_timing.completed_validation_seconds) + 1,
            }
            or (
                len(timing.completed_validation_seconds)
                == len(previous_timing.completed_validation_seconds)
                and timing.validation_timing_sha256 != previous_timing.validation_timing_sha256
            )
            or (
                len(timing.completed_validation_seconds)
                == len(previous_timing.completed_validation_seconds) + 1
                and completed != previous_completed
            )
            or timing.cumulative_training_seconds
            < float(previous_progress["cumulative_training_seconds"])
            or timing.discarded_attempt_seconds < previous_timing.discarded_attempt_seconds
        ):
            raise ValueError("journal timing evidence is not an exact append-only continuation")
        if previous_progress["training_limit_reached"] and (
            completed != previous_progress["completed_updates"]
            or timing.sha256 != previous_progress["timing_evidence_sha256"]
        ):
            raise ValueError("terminal campaign timing evidence is immutable")
    current = None if previous_progress is None else previous_progress["active_resume_name"]
    active_name = _RESUME_NAMES[0] if current != _RESUME_NAMES[0] else _RESUME_NAMES[1]
    timed_payload = _checkpoint_with_timing(payload, timing)
    contents = _checkpoint_bytes(timed_payload)
    _atomic_write_bytes(work_directory / active_name, contents)
    body = {
        "schema": PROGRESS_SCHEMA,
        "protocol_sha256": protocol_sha256,
        "config_sha256": config_sha256,
        "source_sha256": source_sha256,
        "active_resume_name": active_name,
        "checkpoint_sha256": sha256_bytes(contents),
        "model_state_sha256": payload["model_state_sha256"],
        "completed_updates": completed,
        **_timing_progress_fields(timing),
        "rejected_update_count": rejected_update_count,
        "validated_candidate_count": lineage.candidate_count,
        "validated_boundary_updates": lineage.boundary_updates,
        "validated_candidate_sha256": lineage.candidate_sha256,
        "validated_checkpoint_sha256": lineage.checkpoint_sha256,
        "validated_model_state_sha256": lineage.model_state_sha256,
    }
    return _write_progress(work_directory / _PROGRESS_NAME, body), contents


def _validated_resume(
    *,
    work_directory: Path,
    protocol_sha256: str,
    config_sha256: str,
    source_sha256: str,
    architecture_binding: _ArchitectureExecutionBinding,
) -> tuple[dict[str, Any], dict[str, Any], bytes] | None:
    architecture_binding.validate()
    progress_path = work_directory / _PROGRESS_NAME
    if not progress_path.exists():
        if any((work_directory / name).exists() for name in _RESUME_NAMES):
            raise OSError("resume checkpoint exists without a committed progress record")
        return None
    progress = _read_progress(progress_path)
    if (
        progress["protocol_sha256"] != protocol_sha256
        or progress["config_sha256"] != config_sha256
        or progress["source_sha256"] != source_sha256
        or _architecture_from_progress(progress) != architecture_binding
    ):
        raise ValueError("campaign progress binding differs from this qualification")
    contents = _stable_regular_bytes(work_directory / progress["active_resume_name"])
    if sha256_bytes(contents) != progress["checkpoint_sha256"]:
        raise ValueError("committed resume checkpoint digest differs")
    payload = _safe_checkpoint_payload(contents)
    completed, _ = _checkpoint_cursor(payload)
    timing = _timing_from_payload(payload, completed_updates=completed)
    trainer_state = payload["trainer_state"]
    if not isinstance(trainer_state, Mapping):
        raise ValueError("committed resume checkpoint lacks trainer state")
    if (
        completed != progress["completed_updates"]
        or payload["model_state_sha256"] != progress["model_state_sha256"]
        or trainer_state.get("rejected_update_count") != progress["rejected_update_count"]
        or any(progress[name] != value for name, value in _timing_progress_fields(timing).items())
    ):
        raise ValueError("committed progress and resume checkpoint differ")
    return progress, payload, contents


def _lock_execution(work_directory: Path) -> BinaryIO:
    path = work_directory / _LOCK_NAME
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        metadata = os.fstat(descriptor)
        path_metadata = os.lstat(path)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or (metadata.st_dev, metadata.st_ino) != (path_metadata.st_dev, path_metadata.st_ino)
        ):
            raise OSError("campaign execution lock must be one single-link regular file")
        handle = os.fdopen(descriptor, "a+b")
        descriptor = -1
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        final = os.lstat(path)
        if (
            not stat.S_ISREG(final.st_mode)
            or final.st_nlink != 1
            or (metadata.st_dev, metadata.st_ino) != (final.st_dev, final.st_ino)
        ):
            raise OSError("campaign execution lock path changed while being acquired")
    except (BlockingIOError, OSError) as error:
        if descriptor >= 0:
            os.close(descriptor)
        elif "handle" in locals():
            handle.close()
        raise RuntimeError(
            "another dynamic-set campaign executor holds this work directory"
        ) from error
    return handle


def execute_repository_training_block(
    *,
    run_directory: str | Path,
    work_directory: str | Path,
    config_path: str | Path,
    checkpoint_interval_updates: int = 1,
) -> DynamicSetCampaignExecutionReport:
    """Train or resume through exactly the next required validation boundary."""

    if (
        isinstance(checkpoint_interval_updates, bool)
        or not isinstance(checkpoint_interval_updates, int)
        or checkpoint_interval_updates != 1
    ):
        raise ValueError("governed dynamic-set execution requires per-update checkpoints")
    qualification = DynamicSetQualification.attach(run_directory)
    campaign = qualification._campaign()  # Fully validated, read-only replay.
    if campaign["state"] != "training":
        raise RuntimeError("repository training requires an active development campaign")
    architecture = _ArchitectureExecutionBinding.from_qualification(
        qualification.architecture_execution_binding()
    )
    screen_wall_seconds = qualification.screen_wall_seconds()
    candidates = campaign["validation_candidates"]
    if type(candidates) is not list:
        raise ValueError("campaign validation candidates must be one exact list")
    expected_validation_seconds, expected_validation_timing_sha256 = (
        _validation_timing_from_candidates(candidates)
    )
    accepted_boundary = len(candidates) * DEFAULT_CAMPAIGN.validation_interval_updates
    if accepted_boundary > DEFAULT_CAMPAIGN.maximum_updates:
        raise RuntimeError("dynamic-set campaign candidate history exceeds its hard update cap")
    waiting_at_campaign_cap = accepted_boundary == DEFAULT_CAMPAIGN.maximum_updates
    target = min(
        accepted_boundary + DEFAULT_CAMPAIGN.validation_interval_updates,
        DEFAULT_CAMPAIGN.maximum_updates,
    )

    config_file = Path(config_path).absolute()
    config_contents = _stable_regular_bytes(config_file)
    config_sha256 = sha256_bytes(config_contents)
    if config_sha256 != qualification.protocol["config_sha256"]:
        raise ValueError("execution config bytes differ from the qualification protocol")
    base_config = _load_captured_config(config_contents, source_path=config_file)
    if canonical_sha256(base_config.to_dict()) != architecture.base_config_sha256:
        raise ValueError("execution base config payload differs from the architecture ledger")
    config = qualification.architecture_resolved_config()
    if (
        not isinstance(config, OrpheusConfig)
        or canonical_sha256(config.to_dict()) != architecture.resolved_config_sha256
    ):
        raise ValueError("resolved execution config differs from the architecture ledger")
    source = qualification.protocol["source"]
    if not isinstance(source, Mapping):
        raise TypeError("qualification source binding is malformed")
    _authenticate_current_source(source)
    source_sha256 = canonical_sha256(dict(source))

    work = _assert_dedicated_work_directory(
        Path(work_directory).absolute(), qualification.artifacts.root
    )
    lock = _lock_execution(work)
    block_started = time.monotonic()
    try:
        for future_index in range(architecture.architecture_attempt_index + 1, 3):
            future_path = work / _ATTEMPT_DIRECTORY_NAMES[future_index - 1]
            if future_path.exists() or future_path.is_symlink():
                raise OSError("a future architecture-attempt work directory already exists")
        if architecture.architecture_attempt_index == 2:
            # Attempt one was a disposable screen, not a main-campaign run.
            # Its duration and diagnosis are sealed by the qualification
            # ledger, while the shared immutable training cache is the only
            # allowed work artifact.  Main training therefore starts fresh in
            # attempt_02 and must never depend on an invented attempt_01
            # progress journal.
            prior_work = work / _ATTEMPT_DIRECTORY_NAMES[0]
            if prior_work.exists() or prior_work.is_symlink():
                raise OSError("discarded screen attempt created unsupported main progress")
        attempt_work = _attempt_work_directory(
            work,
            attempt_index=architecture.architecture_attempt_index,
            create=True,
        )
        validation_timing_reconciled = False
        training_cache = DynamicSetTrainingCache(
            work / TRAINING_CACHE_DIRECTORY_NAME,
            binding=DynamicSetTrainingCacheBinding(
                source_sha256=source_sha256,
                config_sha256=config_sha256,
                training_manifest_sha256=FROZEN_PHYSICAL_MANIFEST_SHA256["training"],
            ),
        )
        resume = _validated_resume(
            work_directory=attempt_work,
            protocol_sha256=qualification.protocol_sha256,
            config_sha256=config_sha256,
            source_sha256=source_sha256,
            architecture_binding=architecture,
        )
        expected_lineage = _candidate_lineage(
            candidates=candidates,
            work_directory=attempt_work,
        )
        if resume is None:
            if accepted_boundary != 0 or expected_lineage != _EMPTY_LINEAGE:
                raise FileNotFoundError(
                    "campaign progress is required after the first validation boundary"
                )
            _seed_fresh_process(DEFAULT_SCHEDULE_SEED)
            trainer = _fresh_repository_trainer(
                config=config,
                source=source,
                schedule_seed=DEFAULT_SCHEDULE_SEED,
                training_cache=training_cache,
            )
            timing = _UpdateTimingEvidence(
                architecture_binding=architecture,
                screen_wall_seconds=screen_wall_seconds,
            ).validate(completed_updates=0)
            progress, _ = _journal_commit(
                work_directory=attempt_work,
                payload=trainer.checkpoint_payload(),
                previous_progress=None,
                protocol_sha256=qualification.protocol_sha256,
                config_sha256=config_sha256,
                source_sha256=source_sha256,
                timing=timing,
                rejected_update_count=0,
                lineage=_EMPTY_LINEAGE,
            )
        else:
            progress, payload, resume_contents = resume
            completed, schedule_seed = _checkpoint_cursor(payload)
            timing = _timing_from_payload(payload, completed_updates=completed)
            stored_lineage = _lineage_from_progress(progress)
            stored_validation_count = len(timing.completed_validation_seconds)
            if stored_validation_count > len(candidates):
                raise ValueError("resume contains unsealed validation timing evidence")
            stored_validation_sha256 = (
                _ZERO_SHA256
                if stored_validation_count == 0
                else candidates[stored_validation_count - 1]["validation_timing_sha256"]
            )
            if (
                timing.screen_wall_seconds != screen_wall_seconds
                or stored_validation_count != stored_lineage.candidate_count
                or timing.completed_validation_seconds
                != expected_validation_seconds[:stored_validation_count]
                or timing.validation_timing_sha256 != stored_validation_sha256
            ):
                raise ValueError("resume timing differs from its authenticated campaign history")
            if stored_lineage != expected_lineage:
                if not (
                    expected_lineage.candidate_count == stored_lineage.candidate_count + 1
                    and completed == expected_lineage.boundary_updates
                    and sha256_bytes(resume_contents) == expected_lineage.checkpoint_sha256
                    and payload["model_state_sha256"] == expected_lineage.model_state_sha256
                ):
                    raise ValueError(
                        "resume checkpoint is not descended from the latest validated candidate"
                    )
                timing = timing.with_completed_validation(
                    expected_validation_seconds[-1],
                    validation_timing_sha256=expected_validation_timing_sha256,
                )
                progress, _ = _journal_commit(
                    work_directory=attempt_work,
                    payload=payload,
                    previous_progress=progress,
                    protocol_sha256=qualification.protocol_sha256,
                    config_sha256=config_sha256,
                    source_sha256=source_sha256,
                    timing=timing,
                    rejected_update_count=int(progress["rejected_update_count"]),
                    lineage=expected_lineage,
                )
                validation_timing_reconciled = True
                # Reconciliation is its own durable, zero-mutation phase.  In
                # particular, a checkpoint that just qualified at the minimum
                # can now finish without accidentally starting an extension.
                # A later explicit executor call advances toward the next
                # boundary from this exact receipt.
                target = accepted_boundary
            elif (
                timing.completed_validation_seconds != expected_validation_seconds
                or timing.validation_timing_sha256 != expected_validation_timing_sha256
            ):
                raise ValueError("resume omitted or duplicated a validation timing receipt")
            _seed_fresh_process(schedule_seed)
            trainer = _fresh_repository_trainer(
                config=config,
                source=source,
                schedule_seed=schedule_seed,
                training_cache=training_cache,
            )
            trainer.load_checkpoint_payload(_trainer_checkpoint_payload(payload), restore_rng=True)
            if trainer.completed_updates != completed:
                raise ValueError("trainer restore did not reproduce its update cursor")

        if not accepted_boundary <= trainer.completed_updates <= target:
            raise ValueError(
                "resume cursor is outside the current qualification validation interval"
            )
        updates_executed = 0
        stopped_for_time = bool(progress["training_limit_reached"])
        limit_seconds = DEFAULT_CAMPAIGN.training_mutation_seconds
        while trainer.completed_updates < target:
            if stopped_for_time or timing.cumulative_training_seconds >= limit_seconds:
                stopped_for_time = True
                progress, _ = _journal_commit(
                    work_directory=attempt_work,
                    payload=trainer.checkpoint_payload(),
                    previous_progress=progress,
                    protocol_sha256=qualification.protocol_sha256,
                    config_sha256=config_sha256,
                    source_sha256=source_sha256,
                    timing=timing,
                    rejected_update_count=trainer.rejected_update_count,
                    lineage=expected_lineage,
                )
                break
            update_started = time.monotonic()
            try:
                trainer.run_update()
            except BaseException:
                timing = timing.with_discarded_attempt(time.monotonic() - update_started)
                _journal_commit(
                    work_directory=attempt_work,
                    payload=trainer.checkpoint_payload(),
                    previous_progress=progress,
                    protocol_sha256=qualification.protocol_sha256,
                    config_sha256=config_sha256,
                    source_sha256=source_sha256,
                    timing=timing,
                    rejected_update_count=trainer.rejected_update_count,
                    lineage=expected_lineage,
                )
                raise
            update_seconds = time.monotonic() - update_started
            try:
                _authenticate_current_source(source)
            except BaseException:
                committed_contents = _stable_regular_bytes(
                    attempt_work / progress["active_resume_name"]
                )
                committed_payload = _safe_checkpoint_payload(committed_contents)
                committed_timing = _timing_from_payload(
                    committed_payload,
                    completed_updates=int(progress["completed_updates"]),
                ).with_discarded_attempt(update_seconds)
                trainer.load_checkpoint_payload(
                    _trainer_checkpoint_payload(committed_payload), restore_rng=True
                )
                _journal_commit(
                    work_directory=attempt_work,
                    payload=committed_payload,
                    previous_progress=progress,
                    protocol_sha256=qualification.protocol_sha256,
                    config_sha256=config_sha256,
                    source_sha256=source_sha256,
                    timing=committed_timing,
                    rejected_update_count=trainer.rejected_update_count,
                    lineage=expected_lineage,
                )
                raise
            prospective_timing = timing.with_completed_update(update_seconds)
            if prospective_timing.cumulative_training_seconds > limit_seconds:
                # The atomic update crossed the hard budget. Roll it back to
                # the last committed exact-resume state while retaining the
                # time spent by the rejected over-budget attempt.
                committed_contents = _stable_regular_bytes(
                    attempt_work / progress["active_resume_name"]
                )
                committed_payload = _safe_checkpoint_payload(committed_contents)
                trainer.load_checkpoint_payload(
                    _trainer_checkpoint_payload(committed_payload), restore_rng=True
                )
                timing = timing.with_discarded_attempt(update_seconds)
                stopped_for_time = True
                progress, _ = _journal_commit(
                    work_directory=attempt_work,
                    payload=committed_payload,
                    previous_progress=progress,
                    protocol_sha256=qualification.protocol_sha256,
                    config_sha256=config_sha256,
                    source_sha256=source_sha256,
                    timing=timing,
                    rejected_update_count=trainer.rejected_update_count,
                    lineage=expected_lineage,
                )
                break
            timing = prospective_timing
            updates_executed += 1
            progress, _ = _journal_commit(
                work_directory=attempt_work,
                payload=trainer.checkpoint_payload(),
                previous_progress=progress,
                protocol_sha256=qualification.protocol_sha256,
                config_sha256=config_sha256,
                source_sha256=source_sha256,
                timing=timing,
                rejected_update_count=trainer.rejected_update_count,
                lineage=expected_lineage,
            )
            stopped_for_time = bool(progress["training_limit_reached"])
            if stopped_for_time:
                break

        # Defensive consistency guard: every supported update is already
        # journaled, including a time-limit transition.
        if progress["completed_updates"] != trainer.completed_updates:
            progress, _ = _journal_commit(
                work_directory=attempt_work,
                payload=trainer.checkpoint_payload(),
                previous_progress=progress,
                protocol_sha256=qualification.protocol_sha256,
                config_sha256=config_sha256,
                source_sha256=source_sha256,
                timing=timing,
                rejected_update_count=trainer.rejected_update_count,
                lineage=expected_lineage,
            )

        _authenticate_current_source(source)
        resume_path = attempt_work / progress["active_resume_name"]
        checkpoint_path = resume_path
        checkpoint_contents = _stable_regular_bytes(resume_path)
        boundary_reached = trainer.completed_updates == target and not stopped_for_time
        if boundary_reached and not validation_timing_reconciled and not waiting_at_campaign_cap:
            checkpoint_path = attempt_work / f"update_{target:06d}.pt"
            if checkpoint_path.exists():
                existing = _stable_regular_bytes(checkpoint_path)
                if existing != checkpoint_contents:
                    raise FileExistsError("validation-boundary checkpoint already differs")
                checkpoint_contents = existing
            else:
                _atomic_write_bytes(checkpoint_path, checkpoint_contents)

        return DynamicSetCampaignExecutionReport(
            protocol_sha256=qualification.protocol_sha256,
            **architecture.fields(),
            status=(
                "validation_timing_reconciled"
                if validation_timing_reconciled and not progress["training_limit_reached"]
                else progress["execution_status"]
            ),
            completed_updates=trainer.completed_updates,
            target_updates=target,
            updates_executed=updates_executed,
            validation_timing_reconciled=validation_timing_reconciled,
            boundary_reached=boundary_reached,
            stopped_for_time_limit=stopped_for_time,
            checkpoint_path=str(checkpoint_path),
            checkpoint_sha256=sha256_bytes(checkpoint_contents),
            model_state_sha256=str(progress["model_state_sha256"]),
            progress_path=str(attempt_work / _PROGRESS_NAME),
            progress_record_sha256=str(progress["record_sha256"]),
            validated_candidate_count=expected_lineage.candidate_count,
            validated_boundary_updates=expected_lineage.boundary_updates,
            validated_candidate_sha256=expected_lineage.candidate_sha256,
            validated_checkpoint_sha256=expected_lineage.checkpoint_sha256,
            validated_model_state_sha256=expected_lineage.model_state_sha256,
            cumulative_training_seconds=timing.cumulative_training_seconds,
            screen_wall_seconds=timing.screen_wall_seconds,
            completed_update_timing_count=len(timing.completed_update_seconds),
            timing_evidence_sha256=timing.sha256,
            discarded_attempt_seconds=timing.discarded_attempt_seconds,
            execution_validation_timing_count=len(timing.completed_validation_seconds),
            execution_cumulative_validation_seconds=timing.cumulative_validation_seconds,
            execution_validation_timing_sha256=timing.validation_timing_sha256,
            campaign_envelope_limit_seconds=float(progress["campaign_envelope_limit_seconds"]),
            training_mutation_limit_seconds=float(progress["training_mutation_limit_seconds"]),
            reserved_audit_seconds=float(progress["reserved_audit_seconds"]),
            projection_support_satisfied=bool(progress["projection_support_satisfied"]),
            conservative_update_seconds=progress["conservative_update_seconds"],
            conservative_validation_seconds=progress["conservative_validation_seconds"],
            projected_remaining_validation_seconds=progress[
                "projected_remaining_validation_seconds"
            ],
            projected_minimum_training_seconds=progress["projected_minimum_training_seconds"],
            projected_minimum_envelope_seconds=progress["projected_minimum_envelope_seconds"],
            minimum_update_feasible=progress["minimum_update_feasible"],
            limit_hit_reason=progress["limit_hit_reason"],
            block_training_seconds=time.monotonic() - block_started,
            rejected_update_count=trainer.rejected_update_count,
        )
    finally:
        lock.close()


__all__ = [
    "DEFAULT_SCHEDULE_SEED",
    "EXECUTION_SCHEMA",
    "PROGRESS_SCHEMA",
    "DynamicSetCampaignExecutionReport",
    "execute_repository_training_block",
]
