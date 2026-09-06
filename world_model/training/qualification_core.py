"""Reusable, dependency-light qualification scoring and artifact controls.

The historical RGB-D qualification harnesses remain byte-frozen.  New
qualifications use this smaller core for the parts that are genuinely common:
canonical scoring, bounded single-link artifacts, and a durable exactly-once
ordered split ledger.  Scientific materialization and private truth scoring
remain experiment-specific and are deliberately not abstracted here.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import secrets
import stat
import subprocess
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

GateOperator = Literal["le", "ge", "eq"]
SplitStatus = Literal["passed", "failed"]
_MAXIMUM_GIT_OUTPUT_BYTES = 1024 * 1024


def _json_native(value: Any, *, label: str = "value", active: set[int] | None = None) -> Any:
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a nonfinite float")
        return value
    if type(value) not in {dict, list, tuple}:
        raise TypeError(f"{label} contains unsupported type {type(value).__name__}")
    active = set() if active is None else active
    identity = id(value)
    if identity in active:
        raise ValueError(f"{label} contains a recursive container")
    active.add(identity)
    try:
        if type(value) is dict:
            if any(type(key) is not str for key in value):
                raise TypeError(f"{label} mapping keys must be exact strings")
            return {
                key: _json_native(item, label=f"{label}.{key}", active=active)
                for key, item in value.items()
            }
        return [
            _json_native(item, label=f"{label}[{index}]", active=active)
            for index, item in enumerate(value)
        ]
    finally:
        active.remove(identity)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_native(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_bytes(value: bytes) -> str:
    if type(value) is not bytes:
        raise TypeError("SHA-256 input must be exact bytes")
    return hashlib.sha256(value).hexdigest()


def validated_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or len(value) != 64 or value.lower() != value:
        raise ValueError(f"{label} must be one lowercase SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} must be lowercase hexadecimal") from error
    return value


@dataclass(frozen=True, slots=True)
class CleanPublishedGitState:
    """One stable clean HEAD whose configured upstream is exactly equal."""

    commit: str
    tree: str
    upstream_commit: str
    clean: bool = True
    published: bool = True


def _bounded_git_bytes(root: Path, arguments: tuple[str, ...], *, label: str) -> bytes:
    environment = dict(os.environ)
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    try:
        result = subprocess.run(
            ("git", *arguments),
            cwd=root,
            env=environment,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=True,
            timeout=30.0,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise PermissionError(f"cannot capture {label}") from error
    if len(result.stdout) > _MAXIMUM_GIT_OUTPUT_BYTES:
        raise PermissionError(f"{label} exceeds its output bound")
    return result.stdout


def _git_object_identifier(root: Path, revision: str, *, label: str) -> str:
    raw = _bounded_git_bytes(
        root,
        ("rev-parse", "--verify", revision),
        label=label,
    )
    if raw.count(b"\n") != 1 or not raw.endswith(b"\n"):
        raise PermissionError(f"{label} has invalid Git output framing")
    try:
        value = raw[:-1].decode("ascii", errors="strict")
    except UnicodeDecodeError as error:
        raise PermissionError(f"{label} is not ASCII") from error
    if len(value) not in {40, 64} or value.lower() != value:
        raise PermissionError(f"{label} is not one Git object identifier")
    try:
        int(value, 16)
    except ValueError as error:
        raise PermissionError(f"{label} is not hexadecimal") from error
    return value


def capture_clean_published_git_state(repository_root: str | Path) -> CleanPublishedGitState:
    """Capture a stable clean Git tree exactly equal to its local upstream ref.

    This intentionally performs no network access. Publication is established
    by the caller's already-frozen upstream ref, while this helper proves that
    the current worktree, HEAD, tree, and local upstream have not diverged.
    """

    root = Path(repository_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("repository root must be one real directory")
    status_before = _bounded_git_bytes(
        root,
        ("status", "--porcelain=v1", "-z", "--untracked-files=all"),
        label="worktree status",
    )
    commit = _git_object_identifier(root, "HEAD", label="HEAD")
    tree = _git_object_identifier(root, "HEAD^{tree}", label="HEAD tree")
    upstream = _git_object_identifier(root, "@{upstream}", label="upstream")
    divergence = _bounded_git_bytes(
        root,
        ("rev-list", "--left-right", "--count", "HEAD...@{upstream}"),
        label="upstream divergence",
    )
    status_after = _bounded_git_bytes(
        root,
        ("status", "--porcelain=v1", "-z", "--untracked-files=all"),
        label="final worktree status",
    )
    final = (
        _git_object_identifier(root, "HEAD", label="final HEAD"),
        _git_object_identifier(root, "HEAD^{tree}", label="final HEAD tree"),
        _git_object_identifier(root, "@{upstream}", label="final upstream"),
    )
    if (
        status_before
        or status_after
        or divergence != b"0\t0\n"
        or final != (commit, tree, upstream)
        or commit != upstream
    ):
        raise PermissionError("repository differs from one stable clean published source")
    return CleanPublishedGitState(
        commit=commit,
        tree=tree,
        upstream_commit=upstream,
    )


@dataclass(frozen=True, slots=True)
class MetricGate:
    name: str
    operator: GateOperator
    threshold: float

    def validate(self) -> MetricGate:
        if not self.name:
            raise ValueError("metric gate name must be nonempty")
        if self.operator not in {"le", "ge", "eq"}:
            raise ValueError("metric gate operator must be le, ge, or eq")
        if not math.isfinite(self.threshold):
            raise ValueError("metric gate threshold must be finite")
        return self


def metric_gate_failures(
    metrics: Mapping[str, float],
    gates: Sequence[MetricGate],
    *,
    exact_schema: bool = True,
) -> tuple[str, ...]:
    """Evaluate a frozen scalar schema without accepting unsupported zeros."""

    if len({gate.name for gate in gates}) != len(gates):
        raise ValueError("metric gate names must be unique")
    expected = {gate.name for gate in gates}
    if exact_schema and set(metrics) != expected:
        missing = sorted(expected - set(metrics))
        extra = sorted(set(metrics) - expected)
        return (f"metric_schema:missing={missing},extra={extra}",)
    failures: list[str] = []
    for gate in gates:
        gate.validate()
        if gate.name not in metrics:
            failures.append(f"{gate.name}:missing")
            continue
        value = metrics[gate.name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            failures.append(f"{gate.name}:not_scalar")
            continue
        resolved = float(value)
        if not math.isfinite(resolved):
            failures.append(f"{gate.name}:nonfinite")
        elif gate.operator == "le" and resolved > gate.threshold:
            failures.append(f"{gate.name}:{resolved:.12g}>{gate.threshold:.12g}")
        elif gate.operator == "ge" and resolved < gate.threshold:
            failures.append(f"{gate.name}:{resolved:.12g}<{gate.threshold:.12g}")
        elif gate.operator == "eq" and resolved != gate.threshold:
            failures.append(f"{gate.name}:{resolved:.12g}!={gate.threshold:.12g}")
    return tuple(failures)


def weighted_score(components: Mapping[str, float], weights: Mapping[str, float]) -> float:
    """Return one finite lower-is-better score under an exact frozen schema."""

    if set(components) != set(weights):
        missing = sorted(set(weights) - set(components))
        extra = sorted(set(components) - set(weights))
        raise ValueError(f"score components differ: missing={missing}, extra={extra}")
    if not math.isclose(sum(float(value) for value in weights.values()), 1.0, abs_tol=1e-12):
        raise ValueError("score weights must sum to one")
    for label, values in (("component", components), ("weight", weights)):
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in values.values()
        ):
            raise ValueError(f"score {label}s must be finite and nonnegative")
    return math.fsum(float(weights[name]) * float(components[name]) for name in sorted(weights))


class QualificationArtifactDirectory:
    """A fixed-inventory directory of bounded, single-link regular files."""

    def __init__(
        self,
        root: str | Path,
        *,
        allowed_names: Sequence[str],
        maximum_file_bytes: int = 16 * 1024 * 1024,
    ) -> None:
        self.root = Path(root)
        if not self.root.is_absolute():
            raise ValueError("qualification artifact root must be absolute")
        self.allowed_names = frozenset(allowed_names)
        if not self.allowed_names or len(self.allowed_names) != len(tuple(allowed_names)):
            raise ValueError("allowed artifact names must be nonempty and unique")
        for name in self.allowed_names:
            if type(name) is not str or not name or Path(name).name != name or name in {".", ".."}:
                raise ValueError("artifact names must be safe basenames")
        if (
            isinstance(maximum_file_bytes, bool)
            or not isinstance(maximum_file_bytes, int)
            or maximum_file_bytes <= 0
        ):
            raise ValueError("maximum_file_bytes must be a positive integer")
        self.maximum_file_bytes = maximum_file_bytes

    @classmethod
    def create_fresh(
        cls,
        root: str | Path,
        *,
        allowed_names: Sequence[str],
        maximum_file_bytes: int = 16 * 1024 * 1024,
    ) -> QualificationArtifactDirectory:
        instance = cls(
            Path(root).absolute(),
            allowed_names=allowed_names,
            maximum_file_bytes=maximum_file_bytes,
        )
        parent = instance.root.parent
        parent_stat = os.lstat(parent)
        if not stat.S_ISDIR(parent_stat.st_mode):
            raise ValueError("qualification artifact parent must be a directory")
        os.mkdir(instance.root, mode=0o700)
        instance._validate_root()
        return instance

    @classmethod
    def attach(
        cls,
        root: str | Path,
        *,
        allowed_names: Sequence[str],
        maximum_file_bytes: int = 16 * 1024 * 1024,
    ) -> QualificationArtifactDirectory:
        instance = cls(
            Path(root).absolute(),
            allowed_names=allowed_names,
            maximum_file_bytes=maximum_file_bytes,
        )
        instance._validate_root()
        return instance

    def _validate_root(self) -> None:
        metadata = os.lstat(self.root)
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError("qualification artifact root may not be a symlink")
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("qualification artifact root must be a real directory")
        if self.root.resolve(strict=True) != self.root:
            raise ValueError("qualification artifact root may not traverse symlinks")

    def _path(self, name: str) -> Path:
        self._validate_root()
        if name not in self.allowed_names:
            raise ValueError(f"artifact {name!r} is outside the frozen inventory")
        return self.root / name

    @staticmethod
    def _write_all(descriptor: int, contents: bytes) -> None:
        view = memoryview(contents)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("artifact write made no progress")
            view = view[written:]

    def _validate_open_file(
        self, descriptor: int, *, expected_size: int | None = None
    ) -> os.stat_result:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError("qualification artifacts must be single-link regular files")
        if metadata.st_size > self.maximum_file_bytes:
            raise ValueError("qualification artifact exceeds its byte limit")
        if expected_size is not None and metadata.st_size != expected_size:
            raise OSError("qualification artifact size differs after write")
        return metadata

    def write_fresh_bytes(self, name: str, contents: bytes) -> str:
        if type(contents) is not bytes:
            raise TypeError("artifact contents must be exact bytes")
        if len(contents) > self.maximum_file_bytes:
            raise ValueError("qualification artifact exceeds its byte limit")
        path = self._path(name)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags, 0o600)
        try:
            self._write_all(descriptor, contents)
            os.fsync(descriptor)
            self._validate_open_file(descriptor, expected_size=len(contents))
        finally:
            os.close(descriptor)
        self._fsync_root()
        return sha256_bytes(contents)

    def write_fresh_json(self, name: str, value: Mapping[str, Any]) -> str:
        return self.write_fresh_bytes(name, canonical_json_bytes(dict(value)) + b"\n")

    def replace_bytes(self, name: str, contents: bytes) -> str:
        """Atomically replace one already-owned artifact (used by ledgers)."""

        if type(contents) is not bytes:
            raise TypeError("artifact contents must be exact bytes")
        if len(contents) > self.maximum_file_bytes:
            raise ValueError("qualification artifact exceeds its byte limit")
        path = self._path(name)
        self.read_bytes(name)
        temporary_name = f".{name}.{secrets.token_hex(16)}.tmp"
        temporary_path = self.root / temporary_name
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(temporary_path, flags, 0o600)
        try:
            self._write_all(descriptor, contents)
            os.fsync(descriptor)
            self._validate_open_file(descriptor, expected_size=len(contents))
        finally:
            os.close(descriptor)
        try:
            os.replace(temporary_path, path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        self._fsync_root()
        self.read_bytes(name)
        return sha256_bytes(contents)

    def replace_json(self, name: str, value: Mapping[str, Any]) -> str:
        return self.replace_bytes(name, canonical_json_bytes(dict(value)) + b"\n")

    def read_bytes(self, name: str) -> bytes:
        path = self._path(name)
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        try:
            before = self._validate_open_file(descriptor)
            chunks: list[bytes] = []
            remaining = self.maximum_file_bytes + 1
            while remaining:
                chunk = os.read(descriptor, min(remaining, 1024 * 1024))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            after = self._validate_open_file(descriptor)
        finally:
            os.close(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise OSError("qualification artifact changed during read")
        path_metadata = os.lstat(path)
        if (path_metadata.st_dev, path_metadata.st_ino) != (after.st_dev, after.st_ino):
            raise OSError("qualification artifact path changed during read")
        contents = b"".join(chunks)
        if len(contents) > self.maximum_file_bytes:
            raise ValueError("qualification artifact exceeds its byte limit")
        return contents

    def read_json(self, name: str) -> dict[str, Any]:
        try:
            value = json.loads(self.read_bytes(name))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"artifact {name!r} is not strict JSON") from error
        if type(value) is not dict:
            raise ValueError(f"artifact {name!r} must contain one JSON object")
        _json_native(value)
        return value

    def inventory(self) -> frozenset[str]:
        self._validate_root()
        names = frozenset(os.listdir(self.root))
        unexpected = names - self.allowed_names
        if unexpected:
            raise ValueError(f"qualification artifact inventory contains {sorted(unexpected)}")
        for name in names:
            path = self.root / name
            metadata = os.lstat(path)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise ValueError("qualification artifacts must be single-link regular files")
        return names

    def _fsync_root(self) -> None:
        descriptor = os.open(self.root, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


@dataclass(frozen=True, slots=True)
class SplitPermit:
    split: str
    index: int
    nonce: str
    protocol_sha256: str


class OrderedSplitLedger:
    """Durably consume an exact split order once, stopping on any failure."""

    SCHEMA = "orpheus_ordered_split_ledger_v2"

    def __init__(
        self,
        artifacts: QualificationArtifactDirectory,
        *,
        artifact_name: str,
        protocol_sha256: str,
        split_order: Sequence[str],
    ) -> None:
        self.artifacts = artifacts
        self.artifact_name = artifact_name
        self.protocol_sha256 = validated_sha256(protocol_sha256, label="protocol_sha256")
        self.split_order = tuple(split_order)
        if (
            not self.split_order
            or len(set(self.split_order)) != len(self.split_order)
            or any(type(split) is not str or not split for split in self.split_order)
        ):
            raise ValueError("split_order must contain unique nonempty strings")
        if artifact_name not in artifacts.allowed_names:
            raise ValueError("ledger artifact is outside the frozen inventory")

    @contextmanager
    def _exclusive_transaction(self):
        """Serialize state-changing ledger operations across processes."""

        descriptor = os.open(
            self.artifacts.root,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _unsigned_record(
        self,
        *,
        transitions: list[dict[str, Any]],
        next_index: int,
        terminal: bool,
    ) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "protocol_sha256": self.protocol_sha256,
            "split_order": list(self.split_order),
            "next_index": next_index,
            "terminal": terminal,
            "transitions": transitions,
        }

    def _signed_record(self, unsigned: dict[str, Any]) -> dict[str, Any]:
        return {**unsigned, "record_sha256": canonical_sha256(unsigned)}

    def create_fresh(self) -> dict[str, Any]:
        record = self._signed_record(
            self._unsigned_record(transitions=[], next_index=0, terminal=False)
        )
        self.artifacts.write_fresh_json(self.artifact_name, record)
        return record

    def load(self) -> dict[str, Any]:
        record = self.artifacts.read_json(self.artifact_name)
        expected_keys = {
            "schema",
            "protocol_sha256",
            "split_order",
            "next_index",
            "terminal",
            "transitions",
            "record_sha256",
        }
        if set(record) != expected_keys:
            raise ValueError("ledger has an unexpected schema")
        supplied_digest = validated_sha256(record["record_sha256"], label="record_sha256")
        unsigned = {key: value for key, value in record.items() if key != "record_sha256"}
        if canonical_sha256(unsigned) != supplied_digest:
            raise ValueError("ledger record digest mismatch")
        if record["schema"] != self.SCHEMA:
            raise ValueError("ledger schema identifier mismatch")
        if record["protocol_sha256"] != self.protocol_sha256:
            raise ValueError("ledger protocol binding mismatch")
        if record["split_order"] != list(self.split_order):
            raise ValueError("ledger split order mismatch")
        if type(record["next_index"]) is not int or not 0 <= record["next_index"] <= len(
            self.split_order
        ):
            raise ValueError("ledger next_index is invalid")
        if type(record["terminal"]) is not bool or type(record["transitions"]) is not list:
            raise ValueError("ledger terminal/transitions fields are invalid")
        previous = "0" * 64
        active: tuple[str, int, str] | None = None
        claimed_purposes: set[str] = set()
        passed_count = 0
        failed = False
        for sequence, transition in enumerate(record["transitions"]):
            if type(transition) is not dict:
                raise ValueError("ledger transition must be an object")
            expected_transition_keys = {
                "binding_sha256",
                "sequence",
                "event",
                "split",
                "index",
                "nonce",
                "previous_sha256",
                "purpose",
                "result_sha256",
                "status",
                "transition_sha256",
            }
            if set(transition) != expected_transition_keys:
                raise ValueError("ledger transition has an unexpected schema")
            transition_digest = validated_sha256(
                transition["transition_sha256"], label="transition_sha256"
            )
            unsigned_transition = {
                key: value for key, value in transition.items() if key != "transition_sha256"
            }
            if canonical_sha256(unsigned_transition) != transition_digest:
                raise ValueError("ledger transition digest mismatch")
            if transition["sequence"] != sequence or transition["previous_sha256"] != previous:
                raise ValueError("ledger transition chain mismatch")
            previous = transition_digest
            if transition["event"] == "begin":
                if active is not None or failed or transition["index"] != passed_count:
                    raise ValueError("ledger begin transition is out of order")
                if transition["split"] != self.split_order[passed_count]:
                    raise ValueError("ledger begin split is out of order")
                if any(
                    transition[name] is not None
                    for name in ("status", "result_sha256", "purpose", "binding_sha256")
                ):
                    raise ValueError("ledger begin transition contains a result")
                active = (transition["split"], transition["index"], transition["nonce"])
                claimed_purposes.clear()
            elif transition["event"] == "claim":
                if active != (transition["split"], transition["index"], transition["nonce"]):
                    raise ValueError("ledger claim does not match its active split")
                purpose = transition["purpose"]
                if type(purpose) is not str or not purpose or purpose in claimed_purposes:
                    raise ValueError("ledger claim purpose is invalid or duplicated")
                validated_sha256(transition["binding_sha256"], label="claim binding_sha256")
                if transition["status"] is not None or transition["result_sha256"] is not None:
                    raise ValueError("ledger claim transition contains a completion result")
                claimed_purposes.add(purpose)
            elif transition["event"] == "complete":
                if active != (transition["split"], transition["index"], transition["nonce"]):
                    raise ValueError("ledger completion does not match its active split")
                if transition["status"] not in {"passed", "failed"}:
                    raise ValueError("ledger completion status is invalid")
                if transition["purpose"] is not None or transition["binding_sha256"] is not None:
                    raise ValueError("ledger completion contains a claim binding")
                validated_sha256(transition["result_sha256"], label="result_sha256")
                active = None
                claimed_purposes.clear()
                if transition["status"] == "passed":
                    passed_count += 1
                else:
                    failed = True
            else:
                raise ValueError("ledger transition event is invalid")
        if record["next_index"] != passed_count:
            raise ValueError("ledger next_index disagrees with transition history")
        expected_terminal = failed or passed_count == len(self.split_order)
        if record["terminal"] is not expected_terminal:
            raise ValueError("ledger terminal flag disagrees with transition history")
        return record

    def _append(self, record: dict[str, Any], transition: dict[str, Any]) -> dict[str, Any]:
        transitions = list(record["transitions"])
        previous = transitions[-1]["transition_sha256"] if transitions else "0" * 64
        unsigned_transition = {
            "binding_sha256": transition.get("binding_sha256"),
            "sequence": len(transitions),
            "event": transition["event"],
            "split": transition["split"],
            "index": transition["index"],
            "nonce": transition["nonce"],
            "previous_sha256": previous,
            "purpose": transition.get("purpose"),
            "result_sha256": transition.get("result_sha256"),
            "status": transition.get("status"),
        }
        transitions.append(
            {
                **unsigned_transition,
                "transition_sha256": canonical_sha256(unsigned_transition),
            }
        )
        next_index = record["next_index"] + (1 if transition.get("status") == "passed" else 0)
        terminal = transition.get("status") == "failed" or next_index == len(self.split_order)
        updated = self._signed_record(
            self._unsigned_record(
                transitions=transitions,
                next_index=next_index,
                terminal=terminal,
            )
        )
        self.artifacts.replace_json(self.artifact_name, updated)
        return self.load()

    def begin(self, split: str) -> SplitPermit:
        with self._exclusive_transaction():
            record = self.load()
            if record["terminal"]:
                raise RuntimeError("ordered split ledger is terminal")
            transitions = record["transitions"]
            if transitions and transitions[-1]["event"] in {"begin", "claim"}:
                raise RuntimeError("ordered split ledger already has an active split")
            index = record["next_index"]
            expected = self.split_order[index]
            if split != expected:
                raise RuntimeError(f"expected split {expected!r}, received {split!r}")
            nonce = secrets.token_hex(32)
            permit = SplitPermit(
                split=split,
                index=index,
                nonce=nonce,
                protocol_sha256=self.protocol_sha256,
            )
            self._append(
                record,
                {"event": "begin", "split": split, "index": index, "nonce": nonce},
            )
            return permit

    def claim_active(
        self,
        permit: SplitPermit,
        *,
        purpose: str,
        binding_sha256: str,
    ) -> dict[str, Any]:
        """Durably claim one purpose on the exact active split once."""

        if type(permit) is not SplitPermit:
            raise TypeError("split claim requires its exact permit")
        if type(purpose) is not str or not purpose:
            raise ValueError("split claim purpose must be a nonempty string")
        binding_sha256 = validated_sha256(binding_sha256, label="claim binding_sha256")
        with self._exclusive_transaction():
            record = self.load()
            if record["terminal"]:
                raise RuntimeError("ordered split ledger is terminal")
            if permit.protocol_sha256 != self.protocol_sha256:
                raise ValueError("split permit protocol binding mismatch")
            transitions = record["transitions"]
            active_begin = next(
                (
                    transition
                    for transition in reversed(transitions)
                    if transition["event"] in {"begin", "complete"}
                ),
                None,
            )
            if active_begin is None or active_begin["event"] != "begin":
                raise RuntimeError("ordered split ledger has no active split")
            if (permit.split, permit.index, permit.nonce) != (
                active_begin["split"],
                active_begin["index"],
                active_begin["nonce"],
            ):
                raise ValueError("split permit does not match the durable active split")
            active_claims = [
                transition
                for transition in transitions[active_begin["sequence"] + 1 :]
                if transition["event"] == "claim"
            ]
            if any(transition["purpose"] == purpose for transition in active_claims):
                raise RuntimeError(f"split purpose {purpose!r} is already claimed")
            return self._append(
                record,
                {
                    "event": "claim",
                    "split": permit.split,
                    "index": permit.index,
                    "nonce": permit.nonce,
                    "purpose": purpose,
                    "binding_sha256": binding_sha256,
                },
            )

    def active_claims(self, permit: SplitPermit) -> Mapping[str, str]:
        """Return authenticated purpose bindings for the current active split."""

        if type(permit) is not SplitPermit:
            raise TypeError("active claims require an exact permit")
        record = self.load()
        if permit.protocol_sha256 != self.protocol_sha256:
            raise ValueError("split permit protocol binding mismatch")
        transitions = record["transitions"]
        active_begin = next(
            (
                transition
                for transition in reversed(transitions)
                if transition["event"] in {"begin", "complete"}
            ),
            None,
        )
        if active_begin is None or active_begin["event"] != "begin":
            raise RuntimeError("ordered split ledger has no active split")
        if (permit.split, permit.index, permit.nonce) != (
            active_begin["split"],
            active_begin["index"],
            active_begin["nonce"],
        ):
            raise ValueError("split permit does not match the durable active split")
        return {
            transition["purpose"]: transition["binding_sha256"]
            for transition in transitions[active_begin["sequence"] + 1 :]
            if transition["event"] == "claim"
        }

    def complete(
        self,
        permit: SplitPermit,
        *,
        status: SplitStatus,
        result_sha256: str,
    ) -> dict[str, Any]:
        if type(permit) is not SplitPermit:
            raise TypeError("split completion requires its exact permit")
        if status not in {"passed", "failed"}:
            raise ValueError("split completion status must be passed or failed")
        validated_sha256(result_sha256, label="result_sha256")
        with self._exclusive_transaction():
            record = self.load()
            if record["terminal"]:
                raise RuntimeError("ordered split ledger is terminal")
            if permit.protocol_sha256 != self.protocol_sha256:
                raise ValueError("split permit protocol binding mismatch")
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
                raise RuntimeError("ordered split ledger has no active split")
            if (permit.split, permit.index, permit.nonce) != (
                active["split"],
                active["index"],
                active["nonce"],
            ):
                raise ValueError("split permit does not match the durable active split")
            return self._append(
                record,
                {
                    "event": "complete",
                    "split": permit.split,
                    "index": permit.index,
                    "nonce": permit.nonce,
                    "status": status,
                    "result_sha256": result_sha256,
                },
            )


__all__ = [
    "CleanPublishedGitState",
    "MetricGate",
    "OrderedSplitLedger",
    "QualificationArtifactDirectory",
    "SplitPermit",
    "canonical_json_bytes",
    "canonical_sha256",
    "capture_clean_published_git_state",
    "metric_gate_failures",
    "sha256_bytes",
    "validated_sha256",
    "weighted_score",
]
