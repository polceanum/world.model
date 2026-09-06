"""Fresh-process resource evidence for specification 1.61 qualification.

The ordinary evaluator may live for many hours, so its process high-water mark
cannot prove the resource footprint of one checkpoint.  This module owns a
fixed subprocess protocol: a fresh CPU worker re-authenticates one trainer
checkpoint and resolved configuration, runs a repository-owned public N=6
workload, and returns digest-bound raw measurements.  No caller-supplied
function or protected simulator row crosses the boundary.
"""

from __future__ import annotations

import ctypes
import functools
import gc
import hashlib
import io
import json
import math
import os
import resource
import stat
import subprocess
import sys
import tempfile
import types
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from statistics import median
from typing import Any

import torch
from torch import Tensor, nn

from world_model.training.dynamic_set_config import OrpheusConfig, load_config, save_resolved_config
from world_model.training.dynamic_set_gates import ResourceMetrics
from world_model.training.dynamic_set_protocol import (
    PHYSICAL_CELLS,
    PhysicalManifestRow,
    canonical_sha256,
    physical_manifest,
)
from world_model.training.dynamic_set_scene import DYNAMIC_SET_FRAMES
from world_model.training.dynamic_set_trainer import (
    CHECKPOINT_SCHEMA,
    dynamic_set_model_state_sha256,
)
from world_model.training.qualification_core import canonical_json_bytes, validated_sha256

FRESH_RESOURCE_REQUEST_SCHEMA = "dynamic_set_fresh_resource_request_v1"
FRESH_RESOURCE_EVIDENCE_SCHEMA = "dynamic_set_fresh_resource_evidence_v1"
FRESH_RESOURCE_WORKLOAD_SCHEMA = "dynamic_set_fresh_resource_workload_v1"
DEFAULT_FRESH_RESOURCE_TIMEOUT_SECONDS = 15 * 60
MAXIMUM_RESOURCE_CHECKPOINT_BYTES = 512 * 1024 * 1024
MAXIMUM_RESOURCE_RESULT_BYTES = 4 * 1024 * 1024
_RESOURCE_WORKER_ARGUMENT = "--dynamic-set-resource-worker"


class FreshResourceWorkerError(RuntimeError):
    """Raised when fresh-process resource evidence cannot be authenticated."""


class _DarwinTimeValue(ctypes.Structure):
    _fields_ = (("seconds", ctypes.c_int32), ("microseconds", ctypes.c_int32))


class _DarwinMachTaskBasicInfo(ctypes.Structure):
    _fields_ = (
        ("virtual_size", ctypes.c_uint64),
        ("resident_size", ctypes.c_uint64),
        ("resident_size_max", ctypes.c_uint64),
        ("user_time", _DarwinTimeValue),
        ("system_time", _DarwinTimeValue),
        ("policy", ctypes.c_int32),
        ("suspend_count", ctypes.c_int32),
    )


_DARWIN_MACH_TASK_BASIC_INFO = 20
_DARWIN_MACH_TASK_BASIC_INFO_COUNT = 12
_DARWIN_KERN_SUCCESS = 0


def _resource_workload_rows() -> tuple[PhysicalManifestRow, ...]:
    """Freeze one public N=6 row for each contact/lifecycle physical cell."""

    rows = physical_manifest("development")
    selected: list[PhysicalManifestRow] = []
    for cell_index, cell in enumerate(PHYSICAL_CELLS):
        if cell.object_count != 6:
            continue
        selected.append(
            next(
                row
                for row in rows
                if row.cell_index == cell_index and row.object_count == 6 and not row.known_action
            )
        )
    if len(selected) != 4 or len({row.cell_index for row in selected}) != 4:
        raise RuntimeError("fresh resource workload must cover the four N=6 physical cells")
    return tuple(selected)


FRESH_RESOURCE_WORKLOAD_ROWS = _resource_workload_rows()
FRESH_RESOURCE_WORKLOAD_SHA256 = canonical_sha256(
    {
        "schema": FRESH_RESOURCE_WORKLOAD_SCHEMA,
        "rows": [asdict(row) for row in FRESH_RESOURCE_WORKLOAD_ROWS],
    }
)


def _storage_key(value: Tensor) -> tuple[str, int | None, int, int]:
    storage = value.untyped_storage()
    return (value.device.type, value.device.index, storage.data_ptr(), storage.nbytes())


def learned_state_tensor_bytes(model: nn.Module) -> int:
    """Count unique learned-parameter storage.

    Buffers are model-owned persistent tensors, not learned weights, even when
    they are serialized in ``state_dict``.  Keeping those categories disjoint
    prevents a non-persistent buffer from falling through both resource gates.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("resource inventory requires a torch.nn.Module")
    storages = {_storage_key(value) for value in model.parameters()}
    return sum(key[-1] for key in storages)


@dataclass(frozen=True, slots=True)
class ModelTensorInventory:
    """Exhaustive Python-reachable tensor storage split by registration."""

    registered_bytes: int
    parameter_bytes: int
    persistent_buffer_bytes: int
    nonpersistent_buffer_bytes: int
    unregistered_bytes: int
    unregistered_paths: tuple[str, ...]
    non_cpu_paths: tuple[str, ...]
    non_float32_floating_paths: tuple[str, ...]


_ATOMIC_VALUES = (
    str,
    bytes,
    bytearray,
    int,
    float,
    complex,
    bool,
    type(None),
    type,
    Path,
    torch.dtype,
    torch.device,
    types.BuiltinFunctionType,
    types.ModuleType,
    weakref.ReferenceType,
)


def model_tensor_inventory(model: nn.Module) -> ModelTensorInventory:
    """Walk named modules, buffers, dataclasses, slots, and arbitrary attributes.

    Tensors reachable through ordinary Python state but absent from registered
    parameters/buffers are runtime-persistent storage.  Traversal is by object
    identity and storage identity, so aliases and cycles cannot inflate counts.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("resource inventory requires a torch.nn.Module")
    parameters = {_storage_key(value) for value in model.parameters()}
    persistent_buffers: set[tuple[str, int | None, int, int]] = set()
    nonpersistent_buffers: set[tuple[str, int | None, int, int]] = set()
    for module in model.modules():
        for name, value in module._buffers.items():
            if value is None:
                continue
            target = (
                nonpersistent_buffers
                if name in module._non_persistent_buffers_set
                else persistent_buffers
            )
            target.add(_storage_key(value))
    # A storage aliased through multiple registration mechanisms is charged
    # exactly once, with learned parameters taking precedence.
    persistent_buffers -= parameters
    nonpersistent_buffers -= parameters | persistent_buffers
    registered = parameters | persistent_buffers | nonpersistent_buffers
    seen_objects: set[int] = set()
    seen_registered: set[tuple[str, int | None, int, int]] = set()
    unregistered_paths_by_storage: dict[tuple[str, int | None, int, int], str] = {}
    non_cpu_paths_by_storage: dict[tuple[str, int | None, int, int], str] = {}
    non_float32_paths_by_storage: dict[tuple[str, int | None, int, int], str] = {}
    visited = 0

    def visit(value: object, path: str) -> None:
        nonlocal visited
        if isinstance(value, Tensor):
            key = _storage_key(value)
            if value.device.type != "cpu":
                non_cpu_paths_by_storage.setdefault(key, path)
            if (
                value.is_floating_point() or value.is_complex()
            ) and value.dtype is not torch.float32:
                non_float32_paths_by_storage.setdefault(key, path)
            if key in registered:
                seen_registered.add(key)
            else:
                unregistered_paths_by_storage.setdefault(key, path)
            return
        if isinstance(value, _ATOMIC_VALUES):
            return
        identity = id(value)
        if identity in seen_objects:
            return
        seen_objects.add(identity)
        visited += 1
        if visited > 1_000_000:
            raise FreshResourceWorkerError("model attribute graph exceeds the resource bound")

        if isinstance(value, Mapping):
            for index, (key, item) in enumerate(value.items()):
                visit(key, f"{path}.key[{index}]")
                visit(item, f"{path}[{key!r}]")
            return
        if isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")
            return
        if isinstance(value, (set, frozenset)):
            for index, item in enumerate(value):
                visit(item, f"{path}.set[{index}]")
            return
        if isinstance(value, types.MethodType):
            visit(value.__self__, f"{path}.__self__")
            visit(value.__func__, f"{path}.__func__")
            return
        if isinstance(value, types.FunctionType):
            visit(value.__defaults__, f"{path}.__defaults__")
            visit(value.__kwdefaults__, f"{path}.__kwdefaults__")
            closure = value.__closure__ or ()
            for index, cell in enumerate(closure):
                try:
                    contents = cell.cell_contents
                except ValueError:
                    continue
                visit(contents, f"{path}.__closure__[{index}]")
            visit(value.__dict__, f"{path}.__dict__")
            return
        if isinstance(value, functools.partial):
            visit(value.func, f"{path}.func")
            visit(value.args, f"{path}.args")
            visit(value.keywords, f"{path}.keywords")
            visit(value.__dict__, f"{path}.__dict__")
            return
        if is_dataclass(value) and not isinstance(value, type):
            for item in fields(value):
                visit(getattr(value, item.name), f"{path}.{item.name}")
            return

        namespace = getattr(value, "__dict__", None)
        if isinstance(namespace, Mapping):
            for name, item in namespace.items():
                visit(item, f"{path}.{name}")
        slots = getattr(type(value), "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for name in slots:
            if name not in {"__dict__", "__weakref__"} and hasattr(value, name):
                visit(getattr(value, name), f"{path}.{name}")

    visit(model, "model")
    if seen_registered != registered:
        raise FreshResourceWorkerError("registered model tensors escaped resource traversal")
    hidden = tuple(sorted(unregistered_paths_by_storage.values()))
    return ModelTensorInventory(
        registered_bytes=sum(key[-1] for key in registered),
        parameter_bytes=sum(key[-1] for key in parameters),
        persistent_buffer_bytes=sum(key[-1] for key in persistent_buffers),
        nonpersistent_buffer_bytes=sum(key[-1] for key in nonpersistent_buffers),
        unregistered_bytes=sum(key[-1] for key in unregistered_paths_by_storage),
        unregistered_paths=hidden,
        non_cpu_paths=tuple(sorted(non_cpu_paths_by_storage.values())),
        non_float32_floating_paths=tuple(sorted(non_float32_paths_by_storage.values())),
    )


def runtime_tensor_bytes(model: nn.Module) -> int:
    """Return all model-owned non-parameter tensor storage."""

    inventory = model_tensor_inventory(model)
    return (
        inventory.persistent_buffer_bytes
        + inventory.nonpersistent_buffer_bytes
        + inventory.unregistered_bytes
    )


def process_peak_rss_bytes() -> int:
    """Return this process's Unix high-water RSS in bytes."""

    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def process_current_rss_bytes() -> int:
    """Return current RSS from an in-process kernel interface."""

    if sys.platform.startswith("linux"):
        statm = Path(f"/proc/{os.getpid()}/statm").read_text(encoding="ascii").split()
        if len(statm) < 2:
            raise FreshResourceWorkerError("Linux current-RSS record is malformed")
        try:
            rss = int(statm[1]) * int(os.sysconf("SC_PAGE_SIZE"))
        except (OSError, TypeError, ValueError) as error:
            raise FreshResourceWorkerError("Linux current-RSS record is malformed") from error
    elif sys.platform == "darwin":
        if ctypes.sizeof(_DarwinMachTaskBasicInfo) != (
            _DARWIN_MACH_TASK_BASIC_INFO_COUNT * ctypes.sizeof(ctypes.c_uint32)
        ):
            raise FreshResourceWorkerError("Darwin current-RSS record layout is malformed")
        try:
            libsystem = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
            mach_task_self = libsystem.mach_task_self
            task_info = libsystem.task_info
        except (AttributeError, OSError) as error:
            raise FreshResourceWorkerError("Darwin current-RSS interface is unavailable") from error
        mach_task_self.argtypes = ()
        mach_task_self.restype = ctypes.c_uint32
        task_info.argtypes = (
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
        )
        task_info.restype = ctypes.c_int32
        info = _DarwinMachTaskBasicInfo()
        count = ctypes.c_uint32(_DARWIN_MACH_TASK_BASIC_INFO_COUNT)
        result = task_info(
            mach_task_self(),
            _DARWIN_MACH_TASK_BASIC_INFO,
            ctypes.byref(info),
            ctypes.byref(count),
        )
        if result != _DARWIN_KERN_SUCCESS or count.value != _DARWIN_MACH_TASK_BASIC_INFO_COUNT:
            raise FreshResourceWorkerError("Darwin current-RSS query failed")
        rss = int(info.resident_size)
    else:
        raise FreshResourceWorkerError("fresh resource accounting requires Linux or Darwin")
    if rss <= 0:
        raise FreshResourceWorkerError("current process RSS sample is malformed")
    return rss


def _evidence_payload(
    *,
    request_sha256: str,
    checkpoint_sha256: str,
    model_state_sha256: str,
    config_sha256: str,
    source_sha256: str,
    workload_sha256: str,
    worker_pid: int,
    baseline_current_rss_bytes: int,
    model_current_rss_bytes: int,
    maximum_current_rss_bytes: int,
    baseline_peak_rss_bytes: int,
    maximum_peak_rss_bytes: int,
    peak_rss_delta_bytes: int,
    perception_latency_samples_seconds: Sequence[float],
    rollout_latency_samples_seconds: Sequence[float],
    learned_weight_bytes: int,
    persistent_tensor_bytes: int,
    hidden_preload_tensor_bytes: int,
) -> dict[str, Any]:
    return {
        "schema": FRESH_RESOURCE_EVIDENCE_SCHEMA,
        "request_sha256": request_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "model_state_sha256": model_state_sha256,
        "config_sha256": config_sha256,
        "source_sha256": source_sha256,
        "workload_sha256": workload_sha256,
        "worker_pid": worker_pid,
        "baseline_current_rss_bytes": baseline_current_rss_bytes,
        "model_current_rss_bytes": model_current_rss_bytes,
        "maximum_current_rss_bytes": maximum_current_rss_bytes,
        "baseline_peak_rss_bytes": baseline_peak_rss_bytes,
        "maximum_peak_rss_bytes": maximum_peak_rss_bytes,
        "peak_rss_delta_bytes": peak_rss_delta_bytes,
        "perception_latency_samples_seconds": list(perception_latency_samples_seconds),
        "rollout_latency_samples_seconds": list(rollout_latency_samples_seconds),
        "learned_weight_bytes": learned_weight_bytes,
        "persistent_tensor_bytes": persistent_tensor_bytes,
        "hidden_preload_tensor_bytes": hidden_preload_tensor_bytes,
    }


@dataclass(frozen=True, slots=True)
class FreshWorkerResourceEvidence:
    request_sha256: str
    checkpoint_sha256: str
    model_state_sha256: str
    config_sha256: str
    source_sha256: str
    workload_sha256: str
    worker_pid: int
    baseline_current_rss_bytes: int
    model_current_rss_bytes: int
    maximum_current_rss_bytes: int
    baseline_peak_rss_bytes: int
    maximum_peak_rss_bytes: int
    peak_rss_delta_bytes: int
    perception_latency_samples_seconds: tuple[float, ...]
    rollout_latency_samples_seconds: tuple[float, ...]
    learned_weight_bytes: int
    persistent_tensor_bytes: int
    hidden_preload_tensor_bytes: int
    evidence_sha256: str

    @classmethod
    def create(cls, **values: Any) -> FreshWorkerResourceEvidence:
        payload = _evidence_payload(**values)
        return cls(
            **{
                key: value
                for key, value in payload.items()
                if key
                not in {
                    "schema",
                    "perception_latency_samples_seconds",
                    "rollout_latency_samples_seconds",
                }
            },
            perception_latency_samples_seconds=tuple(payload["perception_latency_samples_seconds"]),
            rollout_latency_samples_seconds=tuple(payload["rollout_latency_samples_seconds"]),
            evidence_sha256=canonical_sha256(payload),
        ).validate()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> FreshWorkerResourceEvidence:
        expected = {
            "schema",
            "request_sha256",
            "checkpoint_sha256",
            "model_state_sha256",
            "config_sha256",
            "source_sha256",
            "workload_sha256",
            "worker_pid",
            "baseline_current_rss_bytes",
            "model_current_rss_bytes",
            "maximum_current_rss_bytes",
            "baseline_peak_rss_bytes",
            "maximum_peak_rss_bytes",
            "peak_rss_delta_bytes",
            "perception_latency_samples_seconds",
            "rollout_latency_samples_seconds",
            "learned_weight_bytes",
            "persistent_tensor_bytes",
            "hidden_preload_tensor_bytes",
            "evidence_sha256",
        }
        if type(value) is not dict or set(value) != expected:
            raise ValueError("fresh resource evidence schema differs")
        if value["schema"] != FRESH_RESOURCE_EVIDENCE_SCHEMA:
            raise ValueError("fresh resource evidence version differs")
        return cls(
            **{
                key: value[key]
                for key in expected
                if key
                not in {
                    "schema",
                    "perception_latency_samples_seconds",
                    "rollout_latency_samples_seconds",
                }
            },
            perception_latency_samples_seconds=tuple(value["perception_latency_samples_seconds"]),
            rollout_latency_samples_seconds=tuple(value["rollout_latency_samples_seconds"]),
        ).validate()

    def validate(
        self,
        *,
        expected_request_sha256: str | None = None,
        parent_pid: int | None = None,
    ) -> FreshWorkerResourceEvidence:
        for label, value in (
            ("request", self.request_sha256),
            ("checkpoint", self.checkpoint_sha256),
            ("model state", self.model_state_sha256),
            ("config", self.config_sha256),
            ("source", self.source_sha256),
            ("workload", self.workload_sha256),
            ("evidence", self.evidence_sha256),
        ):
            validated_sha256(value, label=f"fresh resource {label}")
        if expected_request_sha256 is not None and self.request_sha256 != validated_sha256(
            expected_request_sha256,
            label="expected fresh resource request",
        ):
            raise ValueError("fresh resource evidence differs from its request")
        if type(self.worker_pid) is not int or self.worker_pid <= 0:
            raise ValueError("fresh resource worker PID must be positive")
        if parent_pid is not None and self.worker_pid == parent_pid:
            raise ValueError("resource evidence was measured in the parent process")
        counters = (
            "baseline_current_rss_bytes",
            "model_current_rss_bytes",
            "maximum_current_rss_bytes",
            "baseline_peak_rss_bytes",
            "maximum_peak_rss_bytes",
            "peak_rss_delta_bytes",
            "learned_weight_bytes",
            "persistent_tensor_bytes",
            "hidden_preload_tensor_bytes",
        )
        if any(
            type(getattr(self, name)) is not int or getattr(self, name) < 0 for name in counters
        ):
            raise ValueError("fresh resource byte counters must be nonnegative integers")
        if self.maximum_current_rss_bytes < max(
            self.baseline_current_rss_bytes,
            self.model_current_rss_bytes,
        ):
            raise ValueError("fresh resource current-RSS maximum is inconsistent")
        if self.baseline_peak_rss_bytes < self.baseline_current_rss_bytes or (
            self.maximum_peak_rss_bytes < self.maximum_current_rss_bytes
        ):
            raise ValueError("fresh resource current/peak RSS evidence is inconsistent")
        if self.maximum_peak_rss_bytes < self.baseline_peak_rss_bytes or (
            self.peak_rss_delta_bytes != self.maximum_peak_rss_bytes - self.baseline_peak_rss_bytes
        ):
            raise ValueError("fresh resource peak-RSS delta is inconsistent")
        if self.hidden_preload_tensor_bytes != 0:
            raise ValueError("fresh model contains unregistered preloaded tensor storage")
        for name, values in (
            ("perception", self.perception_latency_samples_seconds),
            ("rollout", self.rollout_latency_samples_seconds),
        ):
            if not values or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                or float(item) < 0.0
                for item in values
            ):
                raise ValueError(f"fresh resource {name} latency samples are invalid")
        expected_perception_samples = len(FRESH_RESOURCE_WORKLOAD_ROWS) * DYNAMIC_SET_FRAMES
        if len(self.perception_latency_samples_seconds) != expected_perception_samples or len(
            self.rollout_latency_samples_seconds
        ) != len(FRESH_RESOURCE_WORKLOAD_ROWS):
            raise ValueError("fresh resource latency sample population differs")
        payload = _evidence_payload(
            **{
                name: getattr(self, name)
                for name in _evidence_payload.__annotations__
                if name != "return"
            }
        )
        if canonical_sha256(payload) != self.evidence_sha256:
            raise ValueError("fresh resource evidence digest differs")
        return self

    @property
    def resource_metrics(self) -> ResourceMetrics:
        self.validate()
        return ResourceMetrics(
            perception_latency_seconds=float(median(self.perception_latency_samples_seconds)),
            six_horizon_rollout_seconds=float(median(self.rollout_latency_samples_seconds)),
            learned_weight_bytes=self.learned_weight_bytes,
            persistent_tensor_bytes=self.persistent_tensor_bytes,
            process_rss_bytes=self.maximum_peak_rss_bytes,
        )

    def to_mapping(self) -> dict[str, Any]:
        payload = _evidence_payload(
            **{
                name: getattr(self, name)
                for name in _evidence_payload.__annotations__
                if name != "return"
            }
        )
        return {**payload, "evidence_sha256": self.evidence_sha256}


def _stable_regular_bytes(path: str | Path, *, maximum_bytes: int) -> bytes:
    target = Path(path).absolute()
    before = os.lstat(target)
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size <= 0
        or before.st_size > maximum_bytes
    ):
        raise OSError(f"resource input is not bounded single-link data: {target}")
    descriptor = os.open(target, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    final = os.lstat(target)
    identities = {
        (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_nlink)
        for item in (before, opened, after, final)
    }
    contents = b"".join(chunks)
    if len(identities) != 1 or len(contents) != final.st_size or len(contents) > maximum_bytes:
        raise OSError(f"resource input changed during bounded read: {target}")
    return contents


def _validated_checkpoint_payload(
    contents: bytes,
    *,
    checkpoint_sha256: str,
    model_state_sha256: str,
    config_sha256: str,
    source_sha256: str,
) -> Mapping[str, Any]:
    if hashlib.sha256(contents).hexdigest() != checkpoint_sha256:
        raise ValueError("fresh resource checkpoint digest differs")
    try:
        payload = torch.load(io.BytesIO(contents), map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("fresh resource checkpoint is not a safe tensor payload") from error
    if type(payload) is not dict or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("fresh resource checkpoint schema differs")
    bindings = payload.get("bindings")
    if type(bindings) is not dict:
        raise ValueError("fresh resource checkpoint bindings are missing")
    for key, expected in (("config_sha256", config_sha256), ("source_sha256", source_sha256)):
        if bindings.get(key) != expected:
            raise ValueError(f"fresh resource checkpoint {key} differs")
    provenance = payload.get("source_provenance")
    if not isinstance(provenance, Mapping) or canonical_sha256(dict(provenance)) != source_sha256:
        raise ValueError("fresh resource checkpoint source provenance differs")
    state = payload.get("model_state")
    if not isinstance(state, Mapping):
        raise ValueError("fresh resource checkpoint model state is missing")
    computed = dynamic_set_model_state_sha256(state)
    if computed != model_state_sha256 or payload.get("model_state_sha256") != computed:
        raise ValueError("fresh resource checkpoint model-state digest differs")
    return payload


def _request_payload(
    *,
    checkpoint_path: str,
    checkpoint_sha256: str,
    model_state_sha256: str,
    config_path: str,
    config_sha256: str,
    source_sha256: str,
    workload_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": FRESH_RESOURCE_REQUEST_SCHEMA,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "model_state_sha256": model_state_sha256,
        "config_path": config_path,
        "config_sha256": config_sha256,
        "source_sha256": source_sha256,
        "workload_sha256": workload_sha256,
    }


def _validate_request(value: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    expected = {
        "schema",
        "checkpoint_path",
        "checkpoint_sha256",
        "model_state_sha256",
        "config_path",
        "config_sha256",
        "source_sha256",
        "workload_sha256",
        "request_sha256",
    }
    if type(value) is not dict or set(value) != expected:
        raise ValueError("fresh resource request schema differs")
    if value["schema"] != FRESH_RESOURCE_REQUEST_SCHEMA:
        raise ValueError("fresh resource request version differs")
    body = {key: value[key] for key in expected if key != "request_sha256"}
    request_sha256 = validated_sha256(value["request_sha256"], label="resource request")
    if canonical_sha256(body) != request_sha256:
        raise ValueError("fresh resource request digest differs")
    for name in (
        "checkpoint_sha256",
        "model_state_sha256",
        "config_sha256",
        "source_sha256",
        "workload_sha256",
    ):
        validated_sha256(body[name], label=f"resource request {name}")
    if body["workload_sha256"] != FRESH_RESOURCE_WORKLOAD_SHA256:
        raise ValueError("fresh resource request workload differs from repository source")
    if any(
        type(body[name]) is not str or not Path(body[name]).is_absolute()
        for name in (
            "checkpoint_path",
            "config_path",
        )
    ):
        raise ValueError("fresh resource paths must be absolute strings")
    return body, request_sha256


def _write_exclusive(path: Path, contents: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(contents):
            offset += os.write(descriptor, contents[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _run_resource_worker(request_path: Path, output_path: Path) -> None:
    from world_model.runtime import OnlineWorldModel
    from world_model.runtime.prepared import tensor_identity_version_signature
    from world_model.training.dynamic_set_evaluation import run_public_dynamic_set_episode
    from world_model.training.dynamic_set_materializer import materialize_dynamic_set_episode

    request_bytes = _stable_regular_bytes(request_path, maximum_bytes=64 * 1024)
    request_value = json.loads(request_bytes)
    body, request_sha256 = _validate_request(request_value)
    baseline_current = process_current_rss_bytes()
    baseline_peak = process_peak_rss_bytes()

    config = load_config(body["config_path"])
    if canonical_sha256(config.to_dict()) != body["config_sha256"]:
        raise ValueError("fresh resource resolved config digest differs")
    contents = _stable_regular_bytes(
        body["checkpoint_path"],
        maximum_bytes=MAXIMUM_RESOURCE_CHECKPOINT_BYTES,
    )
    payload = _validated_checkpoint_payload(
        contents,
        checkpoint_sha256=body["checkpoint_sha256"],
        model_state_sha256=body["model_state_sha256"],
        config_sha256=body["config_sha256"],
        source_sha256=body["source_sha256"],
    )
    model = OnlineWorldModel.from_config(config, device="cpu")
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    del payload, contents
    gc.collect()

    preload = model_tensor_inventory(model)
    if preload.non_cpu_paths or preload.non_float32_floating_paths:
        raise TypeError("fresh resource model must use CPU float32 floating tensors")
    if preload.unregistered_bytes:
        raise FreshResourceWorkerError(
            "fresh model contains unregistered preloaded tensors: "
            + ", ".join(preload.unregistered_paths[:8])
        )
    learned_bytes = learned_state_tensor_bytes(model)
    model_current = process_current_rss_bytes()
    maximum_current = max(baseline_current, model_current)

    episodes = tuple(
        tuple(materialize_dynamic_set_episode(row).public_frames())
        for row in FRESH_RESOURCE_WORKLOAD_ROWS
    )
    public_signature = tensor_identity_version_signature(episodes)
    # One untimed warmup uses the exact first public row and leaves no hidden
    # state because the measured pass resets the repository runtime.
    run_public_dynamic_set_episode(model, episodes[0])
    if tensor_identity_version_signature(episodes) != public_signature:
        raise FreshResourceWorkerError("resource warmup mutated its public workload")

    perception_samples: list[float] = []
    rollout_samples: list[float] = []
    maximum_persistent = 0
    for episode in episodes:
        trace = run_public_dynamic_set_episode(model, episode)
        perception_samples.extend(trace.perception_latencies_seconds)
        rollout_samples.append(trace.six_horizon_latency_seconds)
        inventory = model_tensor_inventory(model)
        if inventory.non_cpu_paths or inventory.non_float32_floating_paths:
            raise TypeError("fresh resource runtime must use CPU float32 floating tensors")
        maximum_persistent = max(
            maximum_persistent,
            inventory.persistent_buffer_bytes
            + inventory.nonpersistent_buffer_bytes
            + inventory.unregistered_bytes,
        )
        maximum_current = max(maximum_current, process_current_rss_bytes())
    if tensor_identity_version_signature(episodes) != public_signature:
        raise FreshResourceWorkerError("resource measurement mutated its public workload")
    maximum_peak = process_peak_rss_bytes()
    evidence = FreshWorkerResourceEvidence.create(
        request_sha256=request_sha256,
        checkpoint_sha256=body["checkpoint_sha256"],
        model_state_sha256=body["model_state_sha256"],
        config_sha256=body["config_sha256"],
        source_sha256=body["source_sha256"],
        workload_sha256=body["workload_sha256"],
        worker_pid=os.getpid(),
        baseline_current_rss_bytes=baseline_current,
        model_current_rss_bytes=model_current,
        maximum_current_rss_bytes=maximum_current,
        baseline_peak_rss_bytes=baseline_peak,
        maximum_peak_rss_bytes=maximum_peak,
        peak_rss_delta_bytes=maximum_peak - baseline_peak,
        perception_latency_samples_seconds=perception_samples,
        rollout_latency_samples_seconds=rollout_samples,
        learned_weight_bytes=learned_bytes,
        persistent_tensor_bytes=maximum_persistent,
        hidden_preload_tensor_bytes=preload.unregistered_bytes,
    )
    _write_exclusive(output_path, canonical_json_bytes(evidence.to_mapping()))


def measure_fresh_worker_resources(
    *,
    checkpoint_path: str | Path,
    resolved_config: OrpheusConfig,
    expected_checkpoint_sha256: str,
    expected_model_state_sha256: str,
    expected_config_sha256: str,
    expected_source_sha256: str,
    expected_workload_sha256: str = FRESH_RESOURCE_WORKLOAD_SHA256,
    timeout_seconds: int = DEFAULT_FRESH_RESOURCE_TIMEOUT_SECONDS,
) -> FreshWorkerResourceEvidence:
    """Measure one authenticated checkpoint in a repository-owned subprocess."""

    if type(resolved_config) is not OrpheusConfig:
        raise TypeError("fresh resource measurement requires one exact OrpheusConfig")
    resolved_config.validate()
    checkpoint_sha256 = validated_sha256(
        expected_checkpoint_sha256,
        label="expected resource checkpoint",
    )
    model_state_sha256 = validated_sha256(
        expected_model_state_sha256,
        label="expected resource model state",
    )
    config_sha256 = validated_sha256(expected_config_sha256, label="expected resource config")
    source_sha256 = validated_sha256(expected_source_sha256, label="expected resource source")
    workload_sha256 = validated_sha256(
        expected_workload_sha256,
        label="expected resource workload",
    )
    if workload_sha256 != FRESH_RESOURCE_WORKLOAD_SHA256:
        raise ValueError("expected resource workload differs from repository source")
    if canonical_sha256(resolved_config.to_dict()) != config_sha256:
        raise ValueError("resolved resource config differs from its digest")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int)
        or timeout_seconds <= 0
    ):
        raise ValueError("fresh resource timeout_seconds must be a positive integer")
    checkpoint = Path(checkpoint_path).absolute()
    checkpoint_contents = _stable_regular_bytes(
        checkpoint,
        maximum_bytes=MAXIMUM_RESOURCE_CHECKPOINT_BYTES,
    )
    _validated_checkpoint_payload(
        checkpoint_contents,
        checkpoint_sha256=checkpoint_sha256,
        model_state_sha256=model_state_sha256,
        config_sha256=config_sha256,
        source_sha256=source_sha256,
    )

    repository_root = Path(__file__).resolve().parents[2]
    parent_pid = os.getpid()
    with tempfile.TemporaryDirectory(prefix="dynamic-set-resource-") as directory:
        root = Path(directory)
        config_path = root / "resolved_config.yaml"
        request_path = root / "request.json"
        output_path = root / "evidence.json"
        save_resolved_config(resolved_config, config_path)
        request_body = _request_payload(
            checkpoint_path=str(checkpoint),
            checkpoint_sha256=checkpoint_sha256,
            model_state_sha256=model_state_sha256,
            config_path=str(config_path),
            config_sha256=config_sha256,
            source_sha256=source_sha256,
            workload_sha256=workload_sha256,
        )
        request_sha256 = canonical_sha256(request_body)
        _write_exclusive(
            request_path,
            canonical_json_bytes({**request_body, "request_sha256": request_sha256}),
        )
        environment = dict(os.environ)
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": "",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "PYTHONPATH": str(repository_root),
            }
        )
        completed = subprocess.run(
            (
                sys.executable,
                "-m",
                "world_model.training.dynamic_set_resources",
                _RESOURCE_WORKER_ARGUMENT,
                str(request_path),
                str(output_path),
            ),
            cwd=repository_root,
            env=environment,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip()[-2000:]
            raise FreshResourceWorkerError(
                f"fresh resource worker failed with exit {completed.returncode}: {detail}"
            )
        output = _stable_regular_bytes(output_path, maximum_bytes=MAXIMUM_RESOURCE_RESULT_BYTES)
        try:
            evidence = FreshWorkerResourceEvidence.from_mapping(json.loads(output))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise FreshResourceWorkerError("fresh resource worker output is invalid") from error

    if (
        hashlib.sha256(
            _stable_regular_bytes(checkpoint, maximum_bytes=MAXIMUM_RESOURCE_CHECKPOINT_BYTES)
        ).hexdigest()
        != checkpoint_sha256
    ):
        raise FreshResourceWorkerError("resource checkpoint changed during worker execution")
    evidence.validate(expected_request_sha256=request_sha256, parent_pid=parent_pid)
    expected_bindings = (
        evidence.checkpoint_sha256 == checkpoint_sha256
        and evidence.model_state_sha256 == model_state_sha256
        and evidence.config_sha256 == config_sha256
        and evidence.source_sha256 == source_sha256
        and evidence.workload_sha256 == workload_sha256
    )
    if not expected_bindings:
        raise FreshResourceWorkerError("fresh resource evidence bindings differ")
    return evidence


def _main(arguments: Sequence[str]) -> int:
    if len(arguments) != 3 or arguments[0] != _RESOURCE_WORKER_ARGUMENT:
        raise SystemExit("this module is only an internal dynamic-set resource worker")
    request_path = Path(arguments[1])
    output_path = Path(arguments[2])
    if not request_path.is_absolute() or not output_path.is_absolute():
        raise SystemExit("fresh resource worker paths must be absolute")
    _run_resource_worker(request_path, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))


__all__ = [
    "DEFAULT_FRESH_RESOURCE_TIMEOUT_SECONDS",
    "FRESH_RESOURCE_WORKLOAD_ROWS",
    "FRESH_RESOURCE_WORKLOAD_SHA256",
    "FreshResourceWorkerError",
    "FreshWorkerResourceEvidence",
    "ModelTensorInventory",
    "learned_state_tensor_bytes",
    "measure_fresh_worker_resources",
    "model_tensor_inventory",
    "process_current_rss_bytes",
    "process_peak_rss_bytes",
    "runtime_tensor_bytes",
]
