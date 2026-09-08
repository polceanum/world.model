"""Manifest-scoped compact run retention with fail-closed cleanup."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from world_model.utils.io import atomic_write_text

RUN_MANIFEST_SCHEMA = "world_model_run_manifest_v1"
MIB = 1 << 20
ArtifactCategory = Literal[
    "summary",
    "report",
    "checkpoint",
    "old_checkpoint",
    "rejected",
    "optimizer",
    "transient",
    "media",
    "debug",
]
_PROTECTED_CATEGORIES = frozenset({"summary", "report"})
_CLEAR_GROUPS = {
    "transient": frozenset({"transient", "optimizer"}),
    "media": frozenset({"media", "debug"}),
    "rejected": frozenset({"rejected"}),
}
_PRUNE_ORDER = {
    "transient": 0,
    "optimizer": 0,
    "rejected": 1,
    "media": 2,
    "debug": 2,
    "old_checkpoint": 3,
    "checkpoint": 3,
}


@dataclass(frozen=True, slots=True)
class RunArtifactPolicy:
    mode: Literal["compact", "debug"] = "compact"
    rolling_budget_bytes: int = 250 * MIB
    ordinary_completed_run_bytes: int = 5 * MIB
    newest_failed_debug_bytes: int = 25 * MIB
    promoted_checkpoints_to_keep: int = 2

    def validate(self) -> RunArtifactPolicy:
        if self.mode not in {"compact", "debug"}:
            raise ValueError("artifact policy mode must be compact or debug")
        if self.rolling_budget_bytes <= 0 or self.ordinary_completed_run_bytes <= 0:
            raise ValueError("artifact byte budgets must be positive")
        if self.newest_failed_debug_bytes > self.rolling_budget_bytes:
            raise ValueError("failed-run debug budget cannot exceed the rolling budget")
        if self.promoted_checkpoints_to_keep != 2:
            raise ValueError("compact policy keeps exactly two prior promoted checkpoints")
        return self


DEFAULT_RUN_ARTIFACT_POLICY = RunArtifactPolicy()


@dataclass(frozen=True, slots=True)
class ArtifactEntry:
    path: str
    category: ArtifactCategory
    bytes: int
    present: bool = True
    pinned: bool = False

    def validate(self) -> ArtifactEntry:
        _validate_relative_path(self.path)
        if self.category not in {
            "summary",
            "report",
            "checkpoint",
            "old_checkpoint",
            "rejected",
            "optimizer",
            "transient",
            "media",
            "debug",
        }:
            raise ValueError("unsupported artifact category")
        if isinstance(self.bytes, bool) or self.bytes < 0:
            raise ValueError("artifact bytes must be nonnegative")
        if not isinstance(self.present, bool) or not isinstance(self.pinned, bool):
            raise TypeError("artifact flags must be boolean")
        return self


@dataclass(frozen=True, slots=True)
class RunManifest:
    schema: str
    run_id: str
    created_at_utc: str
    role: Literal["incumbent", "promoted", "candidate", "rejected", "historical"]
    status: Literal["active", "completed", "failed"]
    pinned: bool
    policy: str
    artifacts: tuple[ArtifactEntry, ...]

    def validate(self) -> RunManifest:
        if self.schema != RUN_MANIFEST_SCHEMA:
            raise ValueError("unsupported run manifest schema")
        if not self.run_id or Path(self.run_id).name != self.run_id:
            raise ValueError("run_id must be one safe path component")
        if self.role not in {"incumbent", "promoted", "candidate", "rejected", "historical"}:
            raise ValueError("unsupported run role")
        if self.status not in {"active", "completed", "failed"}:
            raise ValueError("unsupported run status")
        if not isinstance(self.pinned, bool):
            raise TypeError("run pinned flag must be boolean")
        paths = [entry.validate().path for entry in self.artifacts]
        if len(paths) != len(set(paths)):
            raise ValueError("run manifest contains duplicate artifact paths")
        return self

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RunManifest:
        raw_artifacts = value.get("artifacts")
        if not isinstance(raw_artifacts, list):
            raise TypeError("run manifest artifacts must be a list")
        entries = tuple(
            ArtifactEntry(**dict(item)).validate()
            if isinstance(item, Mapping)
            else (_raise_manifest_type())
            for item in raw_artifacts
        )
        payload = dict(value)
        payload["artifacts"] = entries
        return cls(**payload).validate()


@dataclass(frozen=True, slots=True)
class CleanupAction:
    run_id: str
    relative_path: str
    category: str
    bytes: int
    reason: str


@dataclass(frozen=True, slots=True)
class CleanupPlan:
    total_bytes: int
    archive_bytes: int
    protected_bytes: int
    projected_bytes: int
    budget_bytes: int
    actions: tuple[CleanupAction, ...]
    warnings: tuple[str, ...]
    applied: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _raise_manifest_type() -> ArtifactEntry:
    raise TypeError("run manifest artifact entry must be a mapping")


def _validate_relative_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError("artifact path must be a nonempty string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError("artifact path must be normalized and relative")
    if path.parts[0] == ".archive":
        raise ValueError("archive artifacts cannot be managed by run manifests")
    return path


def _safe_member(run_directory: Path, relative_path: str, *, require_file: bool) -> Path:
    relative = _validate_relative_path(relative_path)
    current = run_directory
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"artifact path contains symlink: {relative_path}")
    resolved_run = run_directory.resolve()
    resolved = current.resolve(strict=False)
    if resolved != resolved_run and resolved_run not in resolved.parents:
        raise ValueError("artifact path escapes its run directory")
    if require_file and (not current.is_file() or current.is_symlink()):
        raise ValueError(f"artifact is missing or not a regular file: {relative_path}")
    return current


def _directory_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for item in path.rglob("*"):
        try:
            if item.is_file() and not item.is_symlink():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def write_run_manifest(
    run_directory: str | Path,
    *,
    role: Literal["incumbent", "promoted", "candidate", "rejected", "historical"],
    status: Literal["active", "completed", "failed"],
    artifacts: Mapping[str, ArtifactCategory],
    pinned: bool = False,
    policy: RunArtifactPolicy = DEFAULT_RUN_ARTIFACT_POLICY,
) -> RunManifest:
    """Write a manifest only after checking every managed file in place."""

    policy.validate()
    run = Path(run_directory).expanduser().resolve()
    if not run.is_dir() or run.is_symlink():
        raise ValueError("run directory must be a real directory")
    entries: list[ArtifactEntry] = []
    for relative_path, category in sorted(artifacts.items()):
        path = _safe_member(run, relative_path, require_file=True)
        entries.append(
            ArtifactEntry(
                path=relative_path,
                category=category,
                bytes=path.stat().st_size,
            ).validate()
        )
    debug_bytes = sum(entry.bytes for entry in entries if entry.category in {"debug", "transient"})
    if status == "failed" and debug_bytes > policy.newest_failed_debug_bytes:
        raise ValueError("failed-run debug artifacts exceed the per-run ceiling")
    manifest = RunManifest(
        schema=RUN_MANIFEST_SCHEMA,
        run_id=run.name,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        role=role,
        status=status,
        pinned=pinned,
        policy=policy.mode,
        artifacts=tuple(entries),
    ).validate()
    atomic_write_text(
        run / "run_manifest.json",
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return manifest


def read_run_manifest(run_directory: str | Path) -> RunManifest:
    path = Path(run_directory).expanduser().resolve() / "run_manifest.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise TypeError("run manifest root must be a mapping")
    manifest = RunManifest.from_dict(raw)
    if manifest.run_id != path.parent.name:
        raise ValueError("run manifest ID differs from its directory")
    return manifest


def inventory_runs(
    runs_root: str | Path = "runs",
    *,
    archive_root: str | Path = ".archive",
) -> dict[str, Any]:
    root = Path(runs_root).expanduser().resolve()
    records: list[dict[str, Any]] = []
    if root.exists():
        for run in sorted(root.iterdir()):
            if not run.is_dir() or run.is_symlink() or run.name == "progress":
                continue
            size = _directory_size(run)
            try:
                manifest = read_run_manifest(run)
                records.append(
                    {
                        "run_id": run.name,
                        "role": manifest.role,
                        "status": manifest.status,
                        "bytes": size,
                        "manifest": "valid",
                        "pruning_eligible": not manifest.pinned and manifest.role != "incumbent",
                    }
                )
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
                records.append(
                    {
                        "run_id": run.name,
                        "role": "protected-unmanifested",
                        "status": "unknown",
                        "bytes": size,
                        "manifest": f"invalid-or-missing: {error}",
                        "pruning_eligible": False,
                    }
                )
    return {
        "runs_root": str(root),
        "total_bytes": _directory_size(root),
        "archive_bytes": _directory_size(Path(archive_root).expanduser().resolve()),
        "runs": records,
    }


def _valid_manifests(root: Path) -> tuple[list[tuple[Path, RunManifest]], list[str]]:
    records: list[tuple[Path, RunManifest]] = []
    warnings: list[str] = []
    if not root.exists():
        return records, warnings
    for run in sorted(root.iterdir()):
        if not run.is_dir() or run.is_symlink() or run.name == "progress":
            continue
        try:
            records.append((run, read_run_manifest(run)))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            warnings.append(
                f"{run.name}: protected because manifest is invalid or missing ({error})"
            )
    return records, warnings


def _protected_promoted(records: Sequence[tuple[Path, RunManifest]], keep: int) -> set[str]:
    promoted = sorted(
        (manifest for _, manifest in records if manifest.role == "promoted"),
        key=lambda item: (item.created_at_utc, item.run_id),
        reverse=True,
    )
    return {manifest.run_id for manifest in promoted[:keep]}


def plan_cleanup(
    runs_root: str | Path = "runs",
    *,
    archive_root: str | Path = ".archive",
    category: Literal["transient", "media", "rejected"] | None = None,
    policy: RunArtifactPolicy = DEFAULT_RUN_ARTIFACT_POLICY,
) -> CleanupPlan:
    """Return a dry-run cleanup plan; no filesystem mutation occurs."""

    policy.validate()
    if category is not None and category not in _CLEAR_GROUPS:
        raise ValueError("clear category must be transient, media, or rejected")
    root = Path(runs_root).expanduser().resolve()
    records, warnings = _valid_manifests(root)
    protected_promoted = _protected_promoted(records, policy.promoted_checkpoints_to_keep)
    failed = sorted(
        (manifest for _, manifest in records if manifest.status == "failed"),
        key=lambda item: (item.created_at_utc, item.run_id),
        reverse=True,
    )
    newest_failed = failed[0].run_id if failed else None
    candidates: list[tuple[bool, int, str, CleanupAction]] = []
    protected_bytes = 0
    for run, manifest in records:
        whole_run_protected = manifest.pinned or manifest.role == "incumbent"
        for entry in manifest.artifacts:
            if not entry.present:
                continue
            try:
                path = _safe_member(run, entry.path, require_file=True)
            except ValueError as error:
                warnings.append(f"{manifest.run_id}/{entry.path}: protected ({error})")
                protected_bytes += entry.bytes
                continue
            actual_bytes = path.stat().st_size
            protected = (
                whole_run_protected
                or entry.pinned
                or entry.category in _PROTECTED_CATEGORIES
                or (manifest.run_id in protected_promoted and entry.category.endswith("checkpoint"))
            )
            if protected:
                protected_bytes += actual_bytes
                continue
            if category is not None:
                if entry.category not in _CLEAR_GROUPS[category]:
                    continue
                reason = f"clear category {category}"
            else:
                if entry.category not in _PRUNE_ORDER:
                    continue
                reason = (
                    "only newest failed run retains debug data"
                    if manifest.status == "failed"
                    and manifest.run_id != newest_failed
                    and entry.category == "debug"
                    else "rolling 250 MiB budget"
                )
            mandatory = reason == "only newest failed run retains debug data"
            candidates.append(
                (
                    mandatory,
                    _PRUNE_ORDER.get(entry.category, 9),
                    manifest.created_at_utc,
                    CleanupAction(
                        run_id=manifest.run_id,
                        relative_path=entry.path,
                        category=entry.category,
                        bytes=actual_bytes,
                        reason=reason,
                    ),
                )
            )
    total = _directory_size(root)
    selected: list[CleanupAction] = []
    projected = total
    candidates.sort(
        key=lambda item: (
            not item[0],
            item[1],
            item[2],
            item[3].run_id,
            item[3].relative_path,
        )
    )
    if category is not None:
        selected = [item[3] for item in candidates]
        projected -= sum(item.bytes for item in selected)
    else:
        for mandatory, _, _, action in candidates:
            if mandatory:
                selected.append(action)
                projected -= action.bytes
        for mandatory, _, _, action in candidates:
            if mandatory or projected <= policy.rolling_budget_bytes:
                continue
            selected.append(action)
            projected -= action.bytes
    if protected_bytes > policy.rolling_budget_bytes:
        warnings.append(
            "protected material exceeds the rolling budget; optional artifacts must not be retained"
        )
    if category is None and projected > policy.rolling_budget_bytes:
        warnings.append("managed eligible artifacts are insufficient to reach the rolling budget")
    return CleanupPlan(
        total_bytes=total,
        archive_bytes=_directory_size(Path(archive_root).expanduser().resolve()),
        protected_bytes=protected_bytes,
        projected_bytes=max(0, projected),
        budget_bytes=policy.rolling_budget_bytes,
        actions=tuple(selected),
        warnings=tuple(warnings),
    )


def apply_cleanup(plan: CleanupPlan, runs_root: str | Path = "runs") -> CleanupPlan:
    """Apply one previously computed manifest-scoped plan idempotently."""

    if not isinstance(plan, CleanupPlan):
        raise TypeError("apply_cleanup requires CleanupPlan")
    root = Path(runs_root).expanduser().resolve()
    by_run: dict[str, list[CleanupAction]] = {}
    for action in plan.actions:
        by_run.setdefault(action.run_id, []).append(action)
    warnings = list(plan.warnings)
    removed_bytes = 0
    for run_id, actions in by_run.items():
        run = root / run_id
        manifest = read_run_manifest(run)
        if manifest.pinned or manifest.role == "incumbent":
            raise ValueError(f"cleanup plan attempts to mutate protected run {run_id}")
        action_paths = {action.relative_path for action in actions}
        updated: list[ArtifactEntry] = []
        for entry in manifest.artifacts:
            if entry.path not in action_paths:
                updated.append(entry)
                continue
            if entry.pinned or entry.category in _PROTECTED_CATEGORIES:
                raise ValueError(f"cleanup plan attempts to remove protected artifact {entry.path}")
            path = _safe_member(run, entry.path, require_file=False)
            if path.exists():
                if not path.is_file() or path.is_symlink():
                    raise ValueError(f"cleanup target is no longer a regular file: {entry.path}")
                removed_bytes += path.stat().st_size
                path.unlink()
            updated.append(
                ArtifactEntry(
                    path=entry.path,
                    category=entry.category,
                    bytes=0,
                    present=False,
                    pinned=entry.pinned,
                )
            )
        updated_manifest = RunManifest(
            **{**manifest.to_dict(), "artifacts": tuple(updated)}
        ).validate()
        atomic_write_text(
            run / "run_manifest.json",
            json.dumps(updated_manifest.to_dict(), indent=2, sort_keys=True, allow_nan=False)
            + "\n",
        )
    projected = _directory_size(root)
    if removed_bytes < sum(action.bytes for action in plan.actions):
        warnings.append("some planned files were already absent; cleanup remained idempotent")
    return CleanupPlan(
        total_bytes=plan.total_bytes,
        archive_bytes=plan.archive_bytes,
        protected_bytes=plan.protected_bytes,
        projected_bytes=projected,
        budget_bytes=plan.budget_bytes,
        actions=plan.actions,
        warnings=tuple(warnings),
        applied=True,
    )


def enforce_run_budget(
    runs_root: str | Path = "runs",
    *,
    archive_root: str | Path = ".archive",
    policy: RunArtifactPolicy = DEFAULT_RUN_ARTIFACT_POLICY,
) -> CleanupPlan:
    """Apply rolling-cap cleanup at a run boundary."""

    return apply_cleanup(
        plan_cleanup(runs_root, archive_root=archive_root, policy=policy),
        runs_root,
    )


__all__ = [
    "MIB",
    "RUN_MANIFEST_SCHEMA",
    "ArtifactEntry",
    "CleanupAction",
    "CleanupPlan",
    "RunArtifactPolicy",
    "RunManifest",
    "apply_cleanup",
    "enforce_run_budget",
    "inventory_runs",
    "plan_cleanup",
    "read_run_manifest",
    "write_run_manifest",
]
