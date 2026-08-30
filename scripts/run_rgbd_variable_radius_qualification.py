#!/usr/bin/env python3
"""Two-stage CLI for frozen seedless variable-radius architecture attempt v2.

The outer stage uses only the standard library. It proves that the three-file
publication surface is a clean, upstream-equal HEAD, then replaces itself with
a fresh interpreter. The internal stage consumes that one-shot pipe receipt
before loading project code. The qualification module is compiled from the
exact Git blob named by the receipt.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
import secrets
import stat
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPOSITORY_ROOT / "scripts" / "run_rgbd_variable_radius_qualification.py"
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "rgbd_variable_radius_cpu.yaml"
PUBLICATION_SURFACE_PATHS = {
    "qualification": "world_model/training/rgbd_variable_radius_qualification.py",
    "runner": "scripts/run_rgbd_variable_radius_qualification.py",
    "qualification_test": "tests/unit/test_rgbd_variable_radius_qualification.py",
}
_QUALIFICATION_MODULE = "world_model.training.rgbd_variable_radius_qualification"
_OUTER_RECEIPT_SCHEMA = "rgbd_variable_radius_outer_preflight_v2"
_OUTER_RECEIPT_ENV_PREFIX = "_RGBD_VARIABLE_RADIUS_OUTER_"
_OUTER_RECEIPT_FD_ENV = f"{_OUTER_RECEIPT_ENV_PREFIX}FD"
_OUTER_RECEIPT_SHA256_ENV = f"{_OUTER_RECEIPT_ENV_PREFIX}SHA256"
_OUTER_RECEIPT_NONCE_ENV = f"{_OUTER_RECEIPT_ENV_PREFIX}NONCE"
_OUTER_RECEIPT_MAX_BYTES = 16 * 1024
_PUBLICATION_FILE_MAX_BYTES = 2 * 1024 * 1024
_GIT_OUTPUT_MAX_BYTES = 4 * 1024 * 1024
_GIT_OBJECT_FORMAT = "sha1"
_GIT_OID_BYTES = 20
_INTERNAL_BOOTSTRAP = r"""
import hashlib as _bootstrap_hashlib
import os as _bootstrap_os
import stat as _bootstrap_stat
import sys as _bootstrap_sys

_bootstrap_fd_text = _bootstrap_sys.argv[1]
_bootstrap_expected = _bootstrap_sys.argv[2]
_bootstrap_path = _bootstrap_sys.argv[3]
if not _bootstrap_fd_text.isascii() or not _bootstrap_fd_text.isdecimal():
    raise PermissionError("runner bootstrap descriptor is malformed")
_bootstrap_fd = int(_bootstrap_fd_text)
if _bootstrap_fd < 3 or str(_bootstrap_fd) != _bootstrap_fd_text:
    raise PermissionError("runner bootstrap descriptor is not canonical")
try:
    _bootstrap_stat_value = _bootstrap_os.fstat(_bootstrap_fd)
    _bootstrap_identity = (
        _bootstrap_stat_value.st_dev,
        _bootstrap_stat_value.st_ino,
        _bootstrap_stat_value.st_mode,
        _bootstrap_stat_value.st_nlink,
        _bootstrap_stat_value.st_size,
    )
    if (
        not _bootstrap_stat.S_ISREG(_bootstrap_stat_value.st_mode)
        or _bootstrap_stat_value.st_nlink != 0
        or _bootstrap_stat_value.st_size > 2 * 1024 * 1024
    ):
        raise PermissionError("runner bootstrap authority is not one bounded anonymous file")
    _bootstrap_os.set_inheritable(_bootstrap_fd, False)
    _bootstrap_chunks = []
    _bootstrap_total = 0
    while True:
        _bootstrap_chunk = _bootstrap_os.read(_bootstrap_fd, 65536)
        if not _bootstrap_chunk:
            break
        _bootstrap_chunks.append(_bootstrap_chunk)
        _bootstrap_total += len(_bootstrap_chunk)
        if _bootstrap_total > 2 * 1024 * 1024:
            raise PermissionError("runner bootstrap source is oversized")
    _bootstrap_source = b"".join(_bootstrap_chunks)
    _bootstrap_final_stat = _bootstrap_os.fstat(_bootstrap_fd)
    _bootstrap_final_identity = (
        _bootstrap_final_stat.st_dev,
        _bootstrap_final_stat.st_ino,
        _bootstrap_final_stat.st_mode,
        _bootstrap_final_stat.st_nlink,
        _bootstrap_final_stat.st_size,
    )
    if (
        _bootstrap_final_identity != _bootstrap_identity
        or len(_bootstrap_source) != _bootstrap_stat_value.st_size
    ):
        raise PermissionError("runner bootstrap authority changed during read")
finally:
    _bootstrap_os.close(_bootstrap_fd)
if (
    len(_bootstrap_expected) != 64
    or _bootstrap_expected != _bootstrap_expected.lower()
    or _bootstrap_hashlib.sha256(_bootstrap_source).hexdigest() != _bootstrap_expected
):
    raise PermissionError("runner bootstrap source digest differs")
_bootstrap_sys.argv = [_bootstrap_path, *_bootstrap_sys.argv[4:]]
globals()["__file__"] = _bootstrap_path
globals()["__package__"] = None
globals()["__cached__"] = None
exec(compile(_bootstrap_source, _bootstrap_path, "exec", dont_inherit=True), globals(), globals())
"""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256(contents: bytes) -> str:
    if type(contents) is not bytes:
        raise TypeError("SHA-256 input must be exact bytes")
    return hashlib.sha256(contents).hexdigest()


def _validated_sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise ValueError(f"{label} must be one exact SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} must be lowercase hexadecimal") from error
    if value != value.lower():
        raise ValueError(f"{label} must be lowercase hexadecimal")
    return value


def _validated_git_oid(value: Any, *, label: str) -> str:
    if type(value) is not str or len(value) != 2 * _GIT_OID_BYTES:
        raise ValueError(f"{label} must be one exact Git object ID")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} must be lowercase hexadecimal") from error
    if value != value.lower():
        raise ValueError(f"{label} must be lowercase hexadecimal")
    return value


def _exact_equal(actual: Any, expected: Any, *, label: str) -> None:
    if type(actual) is not type(expected):
        raise PermissionError(f"{label} exact type differs")
    if type(expected) is dict:
        if set(actual) != set(expected):
            raise PermissionError(f"{label} exact mapping keys differ")
        for key in expected:
            if type(key) is not str:
                raise PermissionError(f"{label} mapping keys must be exact strings")
            _exact_equal(actual[key], expected[key], label=f"{label}.{key}")
        return
    if type(expected) is list:
        if len(actual) != len(expected):
            raise PermissionError(f"{label} exact list length differs")
        for index, value in enumerate(expected):
            _exact_equal(actual[index], value, label=f"{label}[{index}]")
        return
    if actual != expected:
        raise PermissionError(f"{label} exact value differs")


def _git_repository_paths() -> tuple[Path, Path, Path]:
    root = REPOSITORY_ROOT
    if not root.is_absolute() or root.resolve() != root:
        raise PermissionError("Git work tree must be one exact canonical Path")
    git_dir = root / ".git"
    git_stat = git_dir.lstat()
    if not stat.S_ISDIR(git_stat.st_mode) or git_dir.resolve() != git_dir:
        raise PermissionError("Git directory must be the exact canonical .git directory")
    common_dir = git_dir
    return git_dir, common_dir, root


def _git_environment() -> dict[str, str]:
    git_dir, common_dir, work_tree = _git_repository_paths()
    environment = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
    environment.update(
        {
            "GIT_DIR": os.fspath(git_dir),
            "GIT_COMMON_DIR": os.fspath(common_dir),
            "GIT_WORK_TREE": os.fspath(work_tree),
            "GIT_INDEX_FILE": os.fspath(git_dir / "index"),
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_LITERAL_PATHSPECS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
            "LC_ALL": "C",
        }
    )
    return environment


def _git_command(arguments: list[str]) -> list[str]:
    git_dir, _, work_tree = _git_repository_paths()
    return [
        "git",
        "--no-replace-objects",
        f"--git-dir={git_dir}",
        f"--work-tree={work_tree}",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.untrackedCache=false",
        *arguments,
    ]


def _git_bytes(arguments: list[str], *, label: str) -> bytes:
    result = subprocess.run(
        _git_command(arguments),
        cwd=REPOSITORY_ROOT,
        env=_git_environment(),
        check=True,
        capture_output=True,
        text=False,
    )
    value = result.stdout
    if type(value) is not bytes or len(value) > _GIT_OUTPUT_MAX_BYTES:
        raise RuntimeError(f"git returned invalid or oversized {label}")
    return value


def _git_text(arguments: list[str], *, label: str, allow_empty: bool = False) -> str:
    raw = _git_bytes(arguments, label=label)
    try:
        value = raw.decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError as error:
        raise RuntimeError(f"git returned non-UTF-8 {label}") from error
    if not value and not allow_empty:
        raise RuntimeError(f"git returned empty {label}")
    return value


def _git_object_oid(object_type: str, contents: bytes) -> str:
    if object_type not in {"blob", "commit", "tree"} or type(contents) is not bytes:
        raise TypeError("Git object framing requires an exact supported type and bytes")
    framed = f"{object_type} {len(contents)}\0".encode("ascii") + contents
    if _GIT_OBJECT_FORMAT == "sha1":
        return hashlib.sha1(framed).hexdigest()
    if _GIT_OBJECT_FORMAT == "sha256":
        return hashlib.sha256(framed).hexdigest()
    raise RuntimeError("unsupported pinned Git object format")


def _git_verified_object(
    oid: str,
    *,
    object_type: str,
    label: str,
    object_cache: dict[tuple[str, str], bytes] | None = None,
) -> bytes:
    if object_type not in {"blob", "commit", "tree"}:
        raise TypeError(f"{label} Git object type is unsupported")
    oid = _validated_git_oid(oid, label=f"{label} object")
    cache_key = (object_type, oid)
    cached = None if object_cache is None else object_cache.get(cache_key)
    if cached is not None:
        if type(cached) is not bytes or _git_object_oid(object_type, cached) != oid:
            raise PermissionError(f"{label} cached Git object framing differs")
        return cached
    reported_type = _git_text(
        ["cat-file", "-t", oid],
        label=f"{label} object type",
    )
    size_text = _git_text(
        ["cat-file", "-s", oid],
        label=f"{label} object size",
    )
    if not size_text.isascii() or not size_text.isdecimal():
        raise PermissionError(f"{label} Git object size is not canonical")
    reported_size = int(size_text)
    if str(reported_size) != size_text:
        raise PermissionError(f"{label} Git object size is not canonical")
    contents = _git_bytes(
        ["cat-file", object_type, oid],
        label=f"{label} object bytes",
    )
    if (
        reported_type != object_type
        or reported_size != len(contents)
        or _git_object_oid(object_type, contents) != oid
    ):
        raise PermissionError(f"{label} Git object type, size, or framed OID differs")
    if object_cache is not None:
        object_cache[cache_key] = contents
    return contents


def _git_commit_tree_oid(
    commit: str,
    *,
    label: str,
    object_cache: dict[tuple[str, str], bytes] | None = None,
) -> str:
    contents = _git_verified_object(
        commit,
        object_type="commit",
        label=label,
        object_cache=object_cache,
    )
    first_line = contents.split(b"\n", 1)[0]
    if not first_line.startswith(b"tree "):
        raise PermissionError(f"{label} commit has no exact leading tree binding")
    try:
        tree_oid = first_line.removeprefix(b"tree ").decode("ascii", errors="strict")
    except UnicodeDecodeError as error:
        raise PermissionError(f"{label} commit tree binding is not ASCII") from error
    return _validated_git_oid(tree_oid, label=f"{label} tree")


def _parse_git_tree(contents: bytes, *, label: str) -> dict[bytes, tuple[str, str]]:
    entries: dict[bytes, tuple[str, str]] = {}
    offset = 0
    while offset < len(contents):
        space = contents.find(b" ", offset)
        nul = contents.find(b"\0", space + 1)
        if space <= offset or nul <= space + 1:
            raise PermissionError(f"{label} Git tree framing differs")
        mode_bytes = contents[offset:space]
        name = contents[space + 1 : nul]
        oid_start = nul + 1
        oid_end = oid_start + _GIT_OID_BYTES
        if oid_end > len(contents) or b"/" in name or name in {b"", b".", b".."}:
            raise PermissionError(f"{label} Git tree entry framing differs")
        try:
            mode = mode_bytes.decode("ascii", errors="strict")
        except UnicodeDecodeError as error:
            raise PermissionError(f"{label} Git tree mode is not ASCII") from error
        if not mode.isdigit() or len(mode) not in {5, 6} or name in entries:
            raise PermissionError(f"{label} Git tree entry schema differs")
        entries[name] = (mode, contents[oid_start:oid_end].hex())
        offset = oid_end
    if offset != len(contents):
        raise PermissionError(f"{label} Git tree length differs")
    return entries


def _git_tree_entry(
    *,
    commit: str,
    relative: str,
    label: str,
    object_cache: dict[tuple[str, str], bytes] | None = None,
    missing_ok: bool = False,
) -> tuple[str, str] | None:
    if (
        type(relative) is not str
        or not relative
        or Path(relative).is_absolute()
        or any(part in {"", ".", ".."} for part in relative.split("/"))
    ):
        raise ValueError(f"{label} path must be one fixed repository-relative path")
    tree_oid = _git_commit_tree_oid(
        commit,
        label=f"{label} commit",
        object_cache=object_cache,
    )
    parts = relative.encode("utf-8", errors="strict").split(b"/")
    for index, part in enumerate(parts):
        tree = _git_verified_object(
            tree_oid,
            object_type="tree",
            label=f"{label} tree level {index}",
            object_cache=object_cache,
        )
        entry = _parse_git_tree(tree, label=f"{label} tree level {index}").get(part)
        if entry is None:
            if missing_ok:
                return None
            raise FileNotFoundError(f"{label} is absent from the exact commit tree")
        mode, oid = entry
        if index + 1 < len(parts):
            if mode != "40000":
                raise PermissionError(f"{label} parent is not an exact Git tree")
            tree_oid = oid
    return mode, _validated_git_oid(oid, label=f"{label} entry")


def _stable_read_file(path: Path, *, label: str) -> bytes:
    if type(path) is not type(Path()) or not path.is_absolute():
        raise TypeError(f"{label} path must be one absolute native Path")
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise PermissionError(f"{label} must be one single-link regular file")
    identity = (before.st_dev, before.st_ino, before.st_mode, before.st_nlink, before.st_size)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        opened_identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_nlink,
            opened.st_size,
        )
        if opened_identity != identity or opened.st_size > _PUBLICATION_FILE_MAX_BYTES:
            raise PermissionError(f"{label} identity or size changed before read")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(
                descriptor,
                min(64 * 1024, _PUBLICATION_FILE_MAX_BYTES + 1 - total),
            )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > _PUBLICATION_FILE_MAX_BYTES:
                raise PermissionError(f"{label} exceeds its exact size limit")
        contents = b"".join(chunks)
        final_opened = os.fstat(descriptor)
        final_identity = (
            final_opened.st_dev,
            final_opened.st_ino,
            final_opened.st_mode,
            final_opened.st_nlink,
            final_opened.st_size,
        )
        if final_identity != identity or len(contents) != opened.st_size:
            raise PermissionError(f"{label} identity changed during read")
    finally:
        os.close(descriptor)
    after = path.lstat()
    after_identity = (after.st_dev, after.st_ino, after.st_mode, after.st_nlink, after.st_size)
    if after_identity != identity:
        raise PermissionError(f"{label} pathname changed during read")
    return contents


def _git_state(
    *,
    object_cache: dict[tuple[str, str], bytes] | None = None,
) -> dict[str, Any]:
    git_dir, common_dir, work_tree = _git_repository_paths()
    repository = Path(
        _git_text(["rev-parse", "--show-toplevel"], label="repository root")
    ).resolve()
    resolved_git_dir = Path(
        _git_text(["rev-parse", "--absolute-git-dir"], label="Git directory")
    ).resolve()
    resolved_common_dir = Path(
        _git_text(
            ["rev-parse", "--path-format=absolute", "--git-common-dir"],
            label="Git common directory",
        )
    ).resolve()
    object_format = _git_text(
        ["rev-parse", "--show-object-format"],
        label="Git object format",
    )
    if (
        repository != work_tree
        or resolved_git_dir != git_dir
        or resolved_common_dir != common_dir
        or object_format != _GIT_OBJECT_FORMAT
    ):
        raise RuntimeError("runner is not executing from its exact owning repository")
    status = _git_bytes(
        ["status", "--porcelain=v1", "-z", "--untracked-files=all"],
        label="publication status",
    )
    if status:
        raise RuntimeError("qualification requires clean committed published source")
    commit = _validated_git_oid(
        _git_text(["rev-parse", "--verify", "HEAD"], label="HEAD commit"),
        label="HEAD commit",
    )
    tree_oid = _git_commit_tree_oid(
        commit,
        label="HEAD commit",
        object_cache=object_cache,
    )
    _git_verified_object(
        tree_oid,
        object_type="tree",
        label="HEAD tree",
        object_cache=object_cache,
    )
    upstream_ref = _git_text(
        ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"],
        label="upstream ref",
    )
    if upstream_ref == "HEAD":
        raise RuntimeError("qualification requires one configured branch upstream")
    upstream_commit = _validated_git_oid(
        _git_text(["rev-parse", "--verify", "@{upstream}"], label="upstream commit"),
        label="upstream commit",
    )
    _git_verified_object(
        upstream_commit,
        object_type="commit",
        label="upstream commit",
        object_cache=object_cache,
    )
    counts = _git_text(
        ["rev-list", "--left-right", "--count", "HEAD...@{upstream}"],
        label="ahead/behind counts",
    ).split()
    if len(counts) != 2:
        raise RuntimeError("git returned malformed ahead/behind counts")
    try:
        ahead, behind = (int(value) for value in counts)
    except ValueError as error:
        raise RuntimeError("git returned non-integer ahead/behind counts") from error
    if upstream_commit != commit or ahead != 0 or behind != 0:
        raise RuntimeError("qualification requires clean HEAD exactly equal to its upstream")
    return {
        "commit": commit,
        "tree_oid": tree_oid,
        "object_format": object_format,
        "upstream_ref": upstream_ref,
        "upstream_commit": upstream_commit,
        "ahead": ahead,
        "behind": behind,
    }


def _blob_binding(
    *,
    commit: str,
    name: str,
    relative: str,
    object_cache: dict[tuple[str, str], bytes] | None = None,
    missing_ok: bool = False,
) -> tuple[dict[str, str], bytes] | None:
    commit = _validated_git_oid(commit, label=f"{name} commit")
    entry = _git_tree_entry(
        commit=commit,
        relative=relative,
        label=name,
        object_cache=object_cache,
        missing_ok=missing_ok,
    )
    if entry is None:
        return None
    mode, blob_oid = entry
    if mode not in {"100644", "100755"}:
        raise PermissionError(f"{name} exact Git blob mode differs")
    blob = _git_verified_object(
        blob_oid,
        object_type="blob",
        label=f"{name} blob",
        object_cache=object_cache,
    )
    digest = _sha256(blob)
    return (
        {
            "path": relative,
            "mode": mode,
            "blob_oid": blob_oid,
            "blob_sha256": digest,
            "worktree_sha256": digest,
        },
        blob,
    )


def _capture_outer_publication() -> dict[str, Any]:
    object_cache: dict[tuple[str, str], bytes] = {}
    before_state = _git_state(object_cache=object_cache)
    commit = before_state["commit"]
    surface_sha256: dict[str, str] = {}
    surface_blobs: dict[str, dict[str, str]] = {}
    for name, relative in PUBLICATION_SURFACE_PATHS.items():
        resolved = _blob_binding(
            commit=commit,
            name=name,
            relative=relative,
            object_cache=object_cache,
        )
        assert resolved is not None
        binding, blob = resolved
        first = _stable_read_file(REPOSITORY_ROOT / relative, label=f"published {name}")
        second = _stable_read_file(REPOSITORY_ROOT / relative, label=f"published {name}")
        if first != blob or second != blob:
            raise RuntimeError(f"published {name} differs from its exact HEAD blob")
        surface_sha256[name] = binding["blob_sha256"]
        surface_blobs[name] = binding
    after_state = _git_state()
    _exact_equal(after_state, before_state, label="outer publication Git state")
    return {
        "repository_root": os.fspath(REPOSITORY_ROOT),
        "publication_git": before_state,
        "publication_surface_sha256": surface_sha256,
        "publication_surface_blobs": surface_blobs,
    }


def _reviewed_arguments(parsed: argparse.Namespace) -> dict[str, str] | None:
    if parsed.phase != "qualification":
        return None
    return {
        "checkpoint_sha256": parsed.reviewed_checkpoint_sha256,
        "report_sha256": parsed.reviewed_report_sha256,
        "ledger_sha256": parsed.reviewed_development_ledger_sha256,
    }


def _public_argv(parsed: argparse.Namespace) -> list[str]:
    result = ["--phase", parsed.phase]
    if parsed.phase == "qualification":
        result.extend(
            [
                "--reviewed-checkpoint-sha256",
                parsed.reviewed_checkpoint_sha256,
                "--reviewed-report-sha256",
                parsed.reviewed_report_sha256,
                "--reviewed-development-ledger-sha256",
                parsed.reviewed_development_ledger_sha256,
            ]
        )
    return result


def _outer_receipt(parsed: argparse.Namespace, *, nonce: str) -> dict[str, Any]:
    return {
        "schema": _OUTER_RECEIPT_SCHEMA,
        "pid": os.getpid(),
        "nonce": nonce,
        "phase": parsed.phase,
        "public_argv": _public_argv(parsed),
        "reviewed": _reviewed_arguments(parsed),
        "publication": _capture_outer_publication(),
    }


def _exec_internal(parsed: argparse.Namespace) -> int:
    if parsed.phase not in {"development", "qualification"} or parsed.internal_stage:
        raise PermissionError("only one outer formal stage may mint an internal receipt")
    nonce = secrets.token_hex(32)
    receipt = _outer_receipt(parsed, nonce=nonce)
    contents = _canonical_json(receipt)
    if not contents or len(contents) > _OUTER_RECEIPT_MAX_BYTES:
        raise RuntimeError("outer preflight receipt has an invalid exact size")
    read_fd, write_fd = os.pipe()
    write_fd_open = True
    try:
        os.set_inheritable(read_fd, True)
        written = 0
        while written < len(contents):
            count = os.write(write_fd, contents[written:])
            if type(count) is not int or count <= 0:
                raise RuntimeError("outer receipt pipe made no progress")
            written += count
        os.close(write_fd)
        write_fd_open = False
        environment = dict(os.environ)
        for name in tuple(environment):
            if name.startswith(_OUTER_RECEIPT_ENV_PREFIX):
                environment.pop(name)
        environment[_OUTER_RECEIPT_FD_ENV] = str(read_fd)
        environment[_OUTER_RECEIPT_SHA256_ENV] = _sha256(contents)
        environment[_OUTER_RECEIPT_NONCE_ENV] = nonce
        runner_binding = receipt["publication"]["publication_surface_blobs"]["runner"]
        if type(runner_binding) is not dict:
            raise PermissionError("outer runner blob binding must be an exact dict")
        runner_oid = _validated_git_oid(
            runner_binding.get("blob_oid"),
            label="outer runner blob",
        )
        runner_source = _git_verified_object(
            runner_oid,
            object_type="blob",
            label="outer runner bootstrap blob",
        )
        runner_sha256 = _sha256(runner_source)
        if runner_sha256 != _validated_sha256(
            runner_binding.get("blob_sha256"),
            label="outer runner blob",
        ) or runner_sha256 != _validated_sha256(
            runner_binding.get("worktree_sha256"),
            label="outer runner worktree",
        ):
            raise PermissionError("outer runner bootstrap blob digest differs")
        executable = os.path.realpath(sys.executable)
        with tempfile.TemporaryFile(mode="w+b") as runner_file:
            runner_file.write(runner_source)
            runner_file.flush()
            os.fsync(runner_file.fileno())
            runner_file.seek(0)
            runner_stat = os.fstat(runner_file.fileno())
            if (
                not stat.S_ISREG(runner_stat.st_mode)
                or runner_stat.st_nlink != 0
                or runner_stat.st_size != len(runner_source)
            ):
                raise PermissionError("outer runner authority is not one exact anonymous file")
            os.set_inheritable(runner_file.fileno(), True)
            argv = [
                executable,
                "-I",
                "-c",
                _INTERNAL_BOOTSTRAP,
                str(runner_file.fileno()),
                runner_sha256,
                os.fspath(RUNNER_PATH),
                *_public_argv(parsed),
                "--_internal-stage",
            ]
            os.execve(executable, argv, environment)
    finally:
        if write_fd_open:
            with contextlib.suppress(OSError):
                os.close(write_fd)
        with contextlib.suppress(OSError):
            os.close(read_fd)
    raise AssertionError("os.execve returned unexpectedly")


def _strict_json_loads(contents: bytes) -> Any:
    if type(contents) is not bytes:
        raise TypeError("outer receipt must be exact bytes")
    try:
        source = contents.decode("ascii", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("outer receipt must be canonical ASCII JSON") from error

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if type(key) is not str or key in result:
                raise ValueError("outer receipt has duplicate or non-string keys")
            result[key] = value
        return result

    def reject_float(value: str) -> Any:
        raise ValueError(f"outer receipt forbids floating-point value {value}")

    def reject_constant(value: str) -> Any:
        raise ValueError(f"outer receipt forbids non-finite value {value}")

    return json.loads(
        source,
        object_pairs_hook=object_pairs,
        parse_float=reject_float,
        parse_constant=reject_constant,
    )


def _consume_outer_receipt(parsed: argparse.Namespace) -> dict[str, Any]:
    names = (
        _OUTER_RECEIPT_FD_ENV,
        _OUTER_RECEIPT_SHA256_ENV,
        _OUTER_RECEIPT_NONCE_ENV,
    )
    values = {name: os.environ.pop(name, None) for name in names}
    fd_text = values[_OUTER_RECEIPT_FD_ENV]
    descriptor: int | None = None
    if type(fd_text) is str and fd_text.isascii() and fd_text.isdecimal():
        descriptor = int(fd_text)
    try:
        if any(name.startswith(_OUTER_RECEIPT_ENV_PREFIX) for name in os.environ):
            raise PermissionError("internal receipt environment has unknown authority fields")
        if any(type(values[name]) is not str or not values[name] for name in names):
            raise PermissionError("internal stage requires one complete outer receipt")
        if descriptor is None:
            raise PermissionError("outer receipt descriptor is malformed")
        if descriptor < 3 or str(descriptor) != fd_text:
            raise PermissionError("outer receipt descriptor is not canonical")
        expected_digest = _validated_sha256(
            values[_OUTER_RECEIPT_SHA256_ENV],
            label="outer receipt digest",
        )
        expected_nonce = _validated_sha256(
            values[_OUTER_RECEIPT_NONCE_ENV],
            label="outer receipt nonce",
        )
        descriptor_stat = os.fstat(descriptor)
        if not stat.S_ISFIFO(descriptor_stat.st_mode):
            raise PermissionError("outer receipt authority is not one pipe")
        os.set_inheritable(descriptor, False)
        os.set_blocking(descriptor, False)
        chunks: list[bytes] = []
        total = 0
        while True:
            try:
                chunk = os.read(descriptor, _OUTER_RECEIPT_MAX_BYTES + 1 - total)
            except BlockingIOError as error:
                raise PermissionError("outer receipt pipe was not closed by its issuer") from error
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > _OUTER_RECEIPT_MAX_BYTES:
                raise PermissionError("outer receipt exceeds its exact size limit")
        contents = b"".join(chunks)
    finally:
        if descriptor is not None and descriptor >= 3:
            with contextlib.suppress(OSError):
                os.close(descriptor)
    if not contents or not secrets.compare_digest(_sha256(contents), expected_digest):
        raise PermissionError("outer receipt digest differs")
    receipt = _strict_json_loads(contents)
    if type(receipt) is not dict or set(receipt) != {
        "schema",
        "pid",
        "nonce",
        "phase",
        "public_argv",
        "reviewed",
        "publication",
    }:
        raise PermissionError("outer receipt exact schema differs")
    if (
        receipt["schema"] != _OUTER_RECEIPT_SCHEMA
        or type(receipt["pid"]) is not int
        or receipt["pid"] != os.getpid()
        or type(receipt["nonce"]) is not str
        or not secrets.compare_digest(receipt["nonce"], expected_nonce)
        or receipt["phase"] != parsed.phase
    ):
        raise PermissionError("outer receipt issuer, stage, or nonce differs")
    _exact_equal(receipt["public_argv"], _public_argv(parsed), label="outer receipt argv")
    _exact_equal(receipt["reviewed"], _reviewed_arguments(parsed), label="outer receipt review")
    current = _capture_outer_publication()
    _exact_equal(current, receipt["publication"], label="outer receipt publication")
    return receipt


def _require_no_preloaded_project_modules(
    modules: dict[str, Any] | None = None,
) -> None:
    inspected = sys.modules if modules is None else modules
    preloaded = sorted(
        name for name in inspected if name == "world_model" or name.startswith("world_model.")
    )
    if preloaded:
        raise PermissionError(
            f"project module was loaded before receipt-bound import: {preloaded[0]}"
        )


class _ReceiptGitBlobLoader(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Load every repository-local project module from one immutable Git tree."""

    def __init__(self, *, commit: str) -> None:
        self.commit = _validated_git_oid(commit, label="receipt-bound loader commit")
        self._baseline_project_modules = {
            name: module
            for name, module in sys.modules.items()
            if name == "world_model" or name.startswith("world_model.")
        }
        self._resolved: dict[str, tuple[dict[str, str], bytes, bool]] = {}
        self._loaded: dict[str, tuple[types.ModuleType, dict[str, str]]] = {}
        self._git_object_cache: dict[tuple[str, str], bytes] = {}

    @staticmethod
    def _candidate_paths(fullname: str) -> tuple[tuple[str, bool], ...]:
        if fullname != "world_model" and not fullname.startswith("world_model."):
            raise ModuleNotFoundError("receipt-bound loader owns only world_model modules")
        components = fullname.split(".")
        if any(not component.isidentifier() for component in components):
            raise ModuleNotFoundError("project module name is not canonical")
        base = "/".join(components)
        return ((f"{base}/__init__.py", True), (f"{base}.py", False))

    def _resolve(self, fullname: str) -> tuple[dict[str, str], bytes, bool]:
        cached = self._resolved.get(fullname)
        if cached is not None:
            return cached
        for relative, is_package in self._candidate_paths(fullname):
            resolved_blob = _blob_binding(
                commit=self.commit,
                name=f"receipt-bound {fullname}",
                relative=relative,
                object_cache=self._git_object_cache,
                missing_ok=True,
            )
            if resolved_blob is None:
                continue
            binding, contents = resolved_blob
            if binding["mode"] not in {"100644", "100755"}:
                raise ImportError(f"receipt-bound {fullname} source mode differs")
            resolved = (binding, contents, is_package)
            self._resolved[fullname] = resolved
            return resolved
        raise ModuleNotFoundError(f"{fullname} is absent from the receipt-bound Git tree")

    def find_spec(
        self,
        fullname: str,
        path: Any = None,
        target: types.ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        del path, target
        if fullname != "world_model" and not fullname.startswith("world_model."):
            return None
        binding, _, is_package = self._resolve(fullname)
        origin = os.fspath(REPOSITORY_ROOT / binding["path"])
        return importlib.util.spec_from_loader(
            fullname,
            self,
            origin=origin,
            is_package=is_package,
        )

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> None:
        del spec
        return None

    def exec_module(self, module: types.ModuleType) -> None:
        fullname = module.__name__
        if fullname in self._loaded:
            raise ImportError(f"receipt-bound {fullname} cannot execute twice")
        binding, contents, is_package = self._resolve(fullname)
        spec = module.__spec__
        origin = os.fspath(REPOSITORY_ROOT / binding["path"])
        if (
            spec is None
            or spec.loader is not self
            or spec.origin != origin
            or (spec.submodule_search_locations is not None) is not is_package
        ):
            raise ImportError(f"receipt-bound {fullname} spec differs")
        module.__file__ = origin
        module.__cached__ = None
        code = compile(contents, origin, "exec", dont_inherit=True)
        exec(code, module.__dict__)
        self._loaded[fullname] = (module, binding)

    def validate_loaded_modules(
        self,
        modules: dict[str, Any] | None = None,
    ) -> None:
        inspected = sys.modules if modules is None else modules
        current = {
            name: module
            for name, module in inspected.items()
            if name == "world_model" or name.startswith("world_model.")
        }
        if set(current) != set(self._loaded):
            raise PermissionError("receipt-bound project module registry differs")
        for name, module in current.items():
            registration = self._loaded[name]
            spec = getattr(module, "__spec__", None)
            if (
                module is not registration[0]
                or getattr(module, "__loader__", None) is not self
                or spec is None
                or spec.loader is not self
                or getattr(module, "__file__", None)
                != os.fspath(REPOSITORY_ROOT / registration[1]["path"])
            ):
                raise PermissionError(f"receipt-bound project module {name} was substituted")


def _load_frozen_qualification(
    publication: dict[str, Any],
) -> tuple[types.ModuleType, _ReceiptGitBlobLoader]:
    _require_no_preloaded_project_modules()
    if type(publication) is not dict:
        raise TypeError("outer publication binding must be an exact dict")
    current = _capture_outer_publication()
    _exact_equal(current, publication, label="pre-import publication")
    git_state = publication.get("publication_git")
    blobs = publication.get("publication_surface_blobs")
    if type(git_state) is not dict or type(blobs) is not dict:
        raise PermissionError("outer publication Git/blob schema differs")
    if git_state.get("object_format") != _GIT_OBJECT_FORMAT:
        raise PermissionError("outer publication Git object format differs")
    loader = _ReceiptGitBlobLoader(commit=git_state.get("commit"))
    sys.meta_path.insert(0, loader)
    try:
        module = importlib.import_module(_QUALIFICATION_MODULE)
        loader.validate_loaded_modules()
        loaded_binding = loader._loaded[_QUALIFICATION_MODULE][1]
        _exact_equal(
            loaded_binding,
            blobs.get("qualification"),
            label="receipt-bound qualification blob",
        )
    except BaseException:
        _release_project_loader_preserving_error(loader)
        raise
    return module, loader


def _release_project_loader(loader: _ReceiptGitBlobLoader) -> None:
    problem: BaseException | None = None
    try:
        loader.validate_loaded_modules()
    except BaseException as error:
        problem = error
    if not sys.meta_path or sys.meta_path[0] is not loader:
        if problem is None:
            problem = PermissionError("receipt-bound project loader order changed")
        if loader in sys.meta_path:
            sys.meta_path.remove(loader)
    else:
        sys.meta_path.pop(0)
    current_project_names = {
        name for name in sys.modules if name == "world_model" or name.startswith("world_model.")
    }
    for name in current_project_names | set(loader._baseline_project_modules):
        current = sys.modules.get(name)
        baseline = loader._baseline_project_modules.get(name)
        if baseline is not None:
            if current is not baseline and problem is None:
                problem = PermissionError(
                    f"receipt-bound baseline project module {name} cleanup differs"
                )
            sys.modules[name] = baseline
        else:
            registration = loader._loaded.get(name)
            if (
                current is not None
                and registration is not None
                and current is not registration[0]
                and problem is None
            ):
                problem = PermissionError(f"receipt-bound project module {name} cleanup differs")
            sys.modules.pop(name, None)
    loader._loaded.clear()
    loader._resolved.clear()
    loader._git_object_cache.clear()
    loader._baseline_project_modules.clear()
    if problem is not None:
        raise problem


def _release_project_loader_preserving_error(loader: _ReceiptGitBlobLoader) -> None:
    active_error = sys.exc_info()[1]
    try:
        _release_project_loader(loader)
    except BaseException as cleanup_error:
        if active_error is None:
            raise
        cleanup_message = (
            f"receipt-bound project cleanup also failed: {type(cleanup_error).__name__}: "
            f"{cleanup_error}"
        )
        add_note = getattr(active_error, "add_note", None)
        if callable(add_note):
            add_note(cleanup_message)
        else:
            active_error.receipt_bound_cleanup_error = cleanup_message


def _record_secondary_error(
    primary: BaseException,
    secondary: BaseException,
    *,
    label: str,
) -> None:
    message = f"{label}: {type(secondary).__name__}: {secondary}"
    add_note = getattr(primary, "add_note", None)
    if callable(add_note):
        add_note(message)
    else:
        previous = getattr(primary, "receipt_bound_secondary_errors", ())
        primary.receipt_bound_secondary_errors = (*previous, message)


def _validate_loaded_publication(module: types.ModuleType, publication: dict[str, Any]) -> Any:
    if module.__name__ != _QUALIFICATION_MODULE:
        raise PermissionError("loaded qualification module identity differs")
    source = module.capture_published_source(REPOSITORY_ROOT)
    expected = {
        "repository_root": os.fspath(REPOSITORY_ROOT),
        "publication_git": source["publication_git"],
        "publication_surface_sha256": source["publication_surface_sha256"],
        "publication_surface_blobs": source["publication_surface_blobs"],
    }
    _exact_equal(expected, publication, label="post-import publication")
    if source["commit"] != source["publication_git"]["commit"] or source["dirty"] is not False:
        raise PermissionError("loaded qualification provenance metadata differs")
    return source


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect variable-radius architecture attempt v2, consume its distinct "
            "development run once, or consume the reviewed protected ladder."
        )
    )
    parser.add_argument(
        "--phase",
        choices=("protocol", "development", "qualification"),
        default="protocol",
    )
    parser.add_argument("--reviewed-checkpoint-sha256")
    parser.add_argument("--reviewed-report-sha256")
    parser.add_argument("--reviewed-development-ledger-sha256")
    parser.add_argument(
        "--_internal-stage",
        dest="internal_stage",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parsed = parser.parse_args(argv)
    reviewed_options = (
        parsed.reviewed_checkpoint_sha256,
        parsed.reviewed_report_sha256,
        parsed.reviewed_development_ledger_sha256,
    )
    if parsed.phase == "qualification" and any(value is None for value in reviewed_options):
        parser.error(
            "qualification requires --reviewed-checkpoint-sha256, "
            "--reviewed-report-sha256, and "
            "--reviewed-development-ledger-sha256"
        )
    if parsed.phase != "qualification" and any(value is not None for value in reviewed_options):
        parser.error("reviewed SHA-256 options are accepted only for qualification")
    if parsed.phase == "protocol" and parsed.internal_stage:
        parser.error("protocol inspection has no internal formal stage")
    return parsed


def main(argv: list[str] | None = None) -> int:
    args = arguments(argv)
    if args.phase == "protocol":
        qualification = importlib.import_module(_QUALIFICATION_MODULE)
        print(
            json.dumps(
                qualification.bridge_protocol(),
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if not args.internal_stage:
        return _exec_internal(args)

    receipt = _consume_outer_receipt(args)
    project_loader: _ReceiptGitBlobLoader | None = None
    try:
        qualification, project_loader = _load_frozen_qualification(receipt["publication"])
        source = _validate_loaded_publication(qualification, receipt["publication"])
        project_loader.validate_loaded_modules()
        torch = importlib.import_module("torch")
        config = qualification.require_frozen_config(CONFIG_PATH)
        torch.set_num_threads(1)
        checkpoint_path = qualification.canonical_checkpoint_path()
        if args.phase == "development":
            report_path = qualification.canonical_development_report_path()
            invocation_seal = None
            try:
                project_loader.validate_loaded_modules()
                invocation_seal = qualification._mint_runner_invocation_seal(
                    stage="development",
                    config=config,
                    config_path=CONFIG_PATH,
                    report_path=report_path,
                    checkpoint_path=checkpoint_path,
                    development_report_path=report_path,
                    source_provenance=source,
                    reviewed_development=None,
                )
                project_loader.validate_loaded_modules()
                result = qualification.run_development(
                    config,
                    config_path=CONFIG_PATH,
                    report_path=report_path,
                    checkpoint_path=checkpoint_path,
                    source_provenance=source,
                    invocation_seal=invocation_seal,
                )
            finally:
                if invocation_seal is not None:
                    active_error = sys.exc_info()[1]
                    loader_error: BaseException | None = None
                    try:
                        project_loader.validate_loaded_modules()
                    except BaseException as error:
                        loader_error = error
                    try:
                        qualification._release_runner_invocation_seal(invocation_seal)
                    except BaseException as release_error:
                        if active_error is not None:
                            _record_secondary_error(
                                active_error,
                                release_error,
                                label="runner seal release also failed",
                            )
                        elif loader_error is not None:
                            _record_secondary_error(
                                release_error,
                                loader_error,
                                label="pre-release loader validation also failed",
                            )
                            raise
                        else:
                            raise
                    if loader_error is not None:
                        if active_error is not None:
                            _record_secondary_error(
                                active_error,
                                loader_error,
                                label="pre-release loader validation also failed",
                            )
                        else:
                            raise loader_error
            print(
                "PASSED: variable-radius development is review-ready; "
                "protected splits remain unopened"
                if result == 0
                else "FAILED: variable-radius development gates; protected splits remain unopened"
            )
            print(f"report: {report_path}")
            print(f"development ledger: {qualification.development_ledger_path()}")
            if result == 0:
                print(f"checkpoint: {checkpoint_path}")
            return result

        report_path = qualification.canonical_qualification_report_path()
        development_report_path = qualification.canonical_development_report_path()
        reviewed_development = {
            "checkpoint_sha256": args.reviewed_checkpoint_sha256,
            "report_sha256": args.reviewed_report_sha256,
            "ledger_sha256": args.reviewed_development_ledger_sha256,
        }
        invocation_seal = None
        try:
            project_loader.validate_loaded_modules()
            invocation_seal = qualification._mint_runner_invocation_seal(
                stage="qualification",
                config=config,
                config_path=CONFIG_PATH,
                report_path=report_path,
                checkpoint_path=checkpoint_path,
                development_report_path=development_report_path,
                source_provenance=source,
                reviewed_development=reviewed_development,
            )
            project_loader.validate_loaded_modules()
            result = qualification.run_qualification(
                config,
                config_path=CONFIG_PATH,
                report_path=report_path,
                checkpoint_path=checkpoint_path,
                development_report_path=development_report_path,
                reviewed_checkpoint_sha256=args.reviewed_checkpoint_sha256,
                reviewed_report_sha256=args.reviewed_report_sha256,
                reviewed_development_ledger_sha256=(args.reviewed_development_ledger_sha256),
                source_provenance=source,
                invocation_seal=invocation_seal,
            )
        finally:
            if invocation_seal is not None:
                active_error = sys.exc_info()[1]
                loader_error = None
                try:
                    project_loader.validate_loaded_modules()
                except BaseException as error:
                    loader_error = error
                try:
                    qualification._release_runner_invocation_seal(invocation_seal)
                except BaseException as release_error:
                    if active_error is not None:
                        _record_secondary_error(
                            active_error,
                            release_error,
                            label="runner seal release also failed",
                        )
                    elif loader_error is not None:
                        _record_secondary_error(
                            release_error,
                            loader_error,
                            label="pre-release loader validation also failed",
                        )
                        raise
                    else:
                        raise
                if loader_error is not None:
                    if active_error is not None:
                        _record_secondary_error(
                            active_error,
                            loader_error,
                            label="pre-release loader validation also failed",
                        )
                    else:
                        raise loader_error
        print(
            "PASSED: selector, confirmation, and one-shot final variable-radius gates"
            if result == 0
            else "FAILED: protected qualification stopped before a later split"
        )
        print(f"report: {report_path}")
        print(f"qualification ledger: {qualification.qualification_ledger_path()}")
        return result
    finally:
        if project_loader is not None:
            _release_project_loader_preserving_error(project_loader)


if __name__ == "__main__":
    raise SystemExit(main())
