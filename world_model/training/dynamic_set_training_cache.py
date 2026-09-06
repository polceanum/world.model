"""Authenticated lean materialization for specification-1.61 training only.

Public evaluation continues to consume :class:`DynamicSetMaterialization` and
its complete 56-frame RGB-D episode.  The optimizer, however, observes only
one rotating perception frame together with compact state and event traces.
This module caches exactly that compact dependency surface after the ordinary
full materializer and scene preflight have accepted a row.  A requested RGB-D
frame and its segmentation labels are then re-rendered from the accepted
trace with the same public renderer.

Cache entries are content addressed and namespace-bound to source, config,
the frozen training manifest, simulator version, and the exact manifest row.
Every load is a bounded ``weights_only`` decode followed by schema, tensor,
binding, filename, and content-digest validation.  Cache files are created
atomically and are required to be single-link regular files; cache corruption
or ambiguity fails closed instead of silently regenerating different data.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import secrets
import stat
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from world_model.simulator.camera import CameraFrame
from world_model.simulator.labels import make_perception_labels
from world_model.simulator.physics import SphereState
from world_model.training.dynamic_set_materializer import (
    DynamicSetMaterialization,
    materialize_dynamic_set_episode,
)
from world_model.training.dynamic_set_protocol import (
    PHYSICAL_SPLIT_SIZES,
    SIMULATOR_VERSION,
    PhysicalManifestRow,
)
from world_model.training.dynamic_set_rendering import (
    PrevalidatedSphereRenderCache,
    prevalidated_sphere_render_cache,
    render_spheres,
)
from world_model.training.dynamic_set_scene import (
    DYNAMIC_SET_DRAG,
    DYNAMIC_SET_FRAME_RATE_HZ,
    DYNAMIC_SET_FRAMES,
    DYNAMIC_SET_FRICTION,
    DYNAMIC_SET_IMAGE_SIZE,
    DYNAMIC_SET_MASS,
    DYNAMIC_SET_MAX_OBJECTS,
    DYNAMIC_SET_RADIUS_M,
    DYNAMIC_SET_RESTITUTION,
    DynamicSetSceneCertificate,
)
from world_model.training.qualification_core import (
    canonical_json_bytes,
    canonical_sha256,
    validated_sha256,
)

TRAINING_CACHE_DIRECTORY_NAME = "training_cache"
TRAINING_EPISODE_SCHEMA = "dynamic_set_lean_training_episode_v1"
TRAINING_CACHE_SCHEMA = "dynamic_set_training_cache_v1"
TRAINING_CACHE_NAMESPACE_SCHEMA = "dynamic_set_training_cache_namespace_v1"
TRAINING_CACHE_ENTRY_SCHEMA = "dynamic_set_training_cache_entry_v1"

_NAMESPACE_NAME = "namespace.json"
_ENTRIES_NAME = "entries"
_MAXIMUM_ENTRY_BYTES = 2 * 1024 * 1024
_MAXIMUM_NAMESPACE_BYTES = 16 * 1024
_MEMORY_ENTRY_LIMIT = 128

_CAMERA_FIELDS: Mapping[str, tuple[tuple[int, ...], torch.dtype]] = {
    "world_from_camera": ((4, 4), torch.float32),
    "camera_from_world": ((4, 4), torch.float32),
    "intrinsics": ((3, 3), torch.float32),
    "position": ((3,), torch.float32),
    "target": ((3,), torch.float32),
}
_OBJECT_FIELDS: Mapping[str, tuple[tuple[int, ...], torch.dtype]] = {
    "id": ((DYNAMIC_SET_FRAMES, DYNAMIC_SET_MAX_OBJECTS), torch.int64),
    "active": ((DYNAMIC_SET_FRAMES, DYNAMIC_SET_MAX_OBJECTS), torch.bool),
    "position": ((DYNAMIC_SET_FRAMES, DYNAMIC_SET_MAX_OBJECTS, 3), torch.float32),
    "velocity": ((DYNAMIC_SET_FRAMES, DYNAMIC_SET_MAX_OBJECTS, 3), torch.float32),
    "orientation": ((DYNAMIC_SET_FRAMES, DYNAMIC_SET_MAX_OBJECTS, 4), torch.float32),
    "angular_velocity": (
        (DYNAMIC_SET_FRAMES, DYNAMIC_SET_MAX_OBJECTS, 3),
        torch.float32,
    ),
    "radius": ((DYNAMIC_SET_FRAMES, DYNAMIC_SET_MAX_OBJECTS, 1), torch.float32),
    "mass": ((DYNAMIC_SET_FRAMES, DYNAMIC_SET_MAX_OBJECTS, 1), torch.float32),
    "restitution": ((DYNAMIC_SET_FRAMES, DYNAMIC_SET_MAX_OBJECTS, 1), torch.float32),
    "drag": ((DYNAMIC_SET_FRAMES, DYNAMIC_SET_MAX_OBJECTS, 1), torch.float32),
    "friction": ((DYNAMIC_SET_FRAMES, DYNAMIC_SET_MAX_OBJECTS, 1), torch.float32),
    "albedo": ((DYNAMIC_SET_FRAMES, DYNAMIC_SET_MAX_OBJECTS, 3), torch.float32),
    # The renderer does not consume sleep state, but retaining it lets the
    # reconstructed SphereState pass the same structural validation.
    "sleeping": ((DYNAMIC_SET_FRAMES, DYNAMIC_SET_MAX_OBJECTS), torch.bool),
}
_EVENT_FIELDS: Mapping[str, tuple[tuple[int, ...], torch.dtype]] = {
    "pair_contact": (
        (DYNAMIC_SET_FRAMES, DYNAMIC_SET_MAX_OBJECTS, DYNAMIC_SET_MAX_OBJECTS),
        torch.bool,
    ),
    "pair_collision": (
        (DYNAMIC_SET_FRAMES, DYNAMIC_SET_MAX_OBJECTS, DYNAMIC_SET_MAX_OBJECTS),
        torch.bool,
    ),
    "externally_actuated": ((DYNAMIC_SET_FRAMES, DYNAMIC_SET_MAX_OBJECTS), torch.bool),
    "known_action_observed": ((DYNAMIC_SET_FRAMES, DYNAMIC_SET_MAX_OBJECTS), torch.bool),
    "known_action_timestamp": (
        (DYNAMIC_SET_FRAMES, DYNAMIC_SET_MAX_OBJECTS),
        torch.float32,
    ),
    "known_impulse_world": (
        (DYNAMIC_SET_FRAMES, DYNAMIC_SET_MAX_OBJECTS, 3),
        torch.float32,
    ),
    "known_action_object_id": (
        (DYNAMIC_SET_FRAMES, DYNAMIC_SET_MAX_OBJECTS),
        torch.int64,
    ),
}


@dataclass(frozen=True, slots=True)
class DynamicSetTrainingCacheBinding:
    """Trusted namespace inputs supplied by the governed executor."""

    source_sha256: str
    config_sha256: str
    training_manifest_sha256: str
    simulator_version: str = SIMULATOR_VERSION

    def validate(self) -> DynamicSetTrainingCacheBinding:
        validated_sha256(self.source_sha256, label="training-cache source")
        validated_sha256(self.config_sha256, label="training-cache config")
        validated_sha256(
            self.training_manifest_sha256,
            label="training-cache manifest",
        )
        if type(self.simulator_version) is not str or not self.simulator_version:
            raise ValueError("training-cache simulator version must be nonempty")
        if self.simulator_version != SIMULATOR_VERSION:
            raise ValueError("training-cache simulator version differs from source")
        return self

    @property
    def namespace_sha256(self) -> str:
        return canonical_sha256(
            {
                "schema": TRAINING_CACHE_NAMESPACE_SCHEMA,
                **asdict(self.validate()),
            }
        )


@dataclass(frozen=True)
class DynamicSetTrainingMaterialization:
    """One lean, adapter-only view reconstructed from a certified cache row."""

    row: PhysicalManifestRow
    episode: Mapping[str, Any]
    known_action_observed: Tensor
    certificate: DynamicSetSceneCertificate
    accepted_seed: int
    attempt_count: int
    rejection_reasons: tuple[str, ...]
    cache_content_sha256: str
    cache_hit: bool

    @property
    def rejection_count(self) -> int:
        return len(self.rejection_reasons)

    @property
    def rejection_rate(self) -> float:
        return self.rejection_count / self.attempt_count


def _validate_real_directory(path: Path, *, create: bool) -> Path:
    absolute = path.absolute()
    if create:
        absolute.mkdir(parents=True, exist_ok=True, mode=0o700)
    metadata = os.lstat(absolute)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise OSError(f"training cache path must be one real directory: {absolute}")
    if absolute.resolve(strict=True) != absolute:
        raise OSError(f"training cache path may not traverse symbolic links: {absolute}")
    return absolute


def validate_training_cache_directory(path: str | Path) -> Path:
    """Validate the cache root shape without opening any entry payload."""

    root = _validate_real_directory(Path(path), create=False)
    allowed = {_NAMESPACE_NAME, _ENTRIES_NAME}
    for item in root.iterdir():
        if item.name not in allowed:
            raise OSError(f"training cache contains unsupported artifact: {item.name}")
        metadata = os.lstat(item)
        if item.name == _NAMESPACE_NAME:
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
            ):
                raise OSError("training-cache namespace must be single-link regular file")
        elif stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise OSError("training-cache entries path must be one real directory")
    return root


def _write_all(descriptor: int, contents: bytes) -> None:
    view = memoryview(contents)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("training-cache write made no progress")
        view = view[written:]


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_fresh(path: Path, contents: bytes) -> None:
    if type(contents) is not bytes or not contents:
        raise ValueError("training-cache artifacts must contain nonempty exact bytes")
    parent = _validate_real_directory(path.parent, create=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"training-cache artifact already exists: {path}")
    temporary = parent / f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        _write_all(descriptor, contents)
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size != len(contents)
        ):
            raise OSError("training-cache temporary artifact changed during write")
    finally:
        os.close(descriptor)
    try:
        # Linking is the portable no-replace commit primitive here: unlike
        # rename(), it atomically fails when another writer already committed
        # the content-addressed destination.  Removing the private temporary
        # name immediately restores the required single-link invariant.
        os.link(temporary, path, follow_symlinks=False)
        temporary.unlink()
        committed = os.lstat(path)
        if not stat.S_ISREG(committed.st_mode) or committed.st_nlink != 1:
            raise OSError("committed training-cache artifact is not single-link data")
        _fsync_directory(parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _stable_regular_bytes(path: Path, *, maximum_bytes: int) -> bytes:
    before = os.lstat(path)
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size <= 0
        or before.st_size > maximum_bytes
    ):
        raise OSError(f"training-cache entry is not bounded single-link data: {path}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
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
    final = os.lstat(path)
    identities = {
        (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_nlink)
        for value in (before, opened, after, final)
    }
    if len(identities) != 1:
        raise OSError(f"training-cache entry changed during read: {path}")
    contents = b"".join(chunks)
    if len(contents) != final.st_size or len(contents) > maximum_bytes:
        raise OSError(f"training-cache entry differs from bounded metadata: {path}")
    return contents


def _tensor_tree_sha256(metadata: Mapping[str, Any], tensors: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    header = canonical_json_bytes(dict(metadata))
    digest.update(len(header).to_bytes(8, byteorder="big"))
    digest.update(header)
    for name in sorted(tensors):
        value = tensors[name]
        if type(name) is not str or not name or not isinstance(value, Tensor):
            raise TypeError("training-cache tensor tree must use named tensors")
        tensor = value.detach().to(device="cpu").contiguous()
        tensor_header = json.dumps(
            {"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        digest.update(len(tensor_header).to_bytes(8, byteorder="big"))
        digest.update(tensor_header)
        raw = tensor.view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, byteorder="big"))
        digest.update(raw)
    return digest.hexdigest()


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _owned_tensor(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"training-cache tensor {name!r} is missing")
    if value.device.type != "cpu" or value.layout is not torch.strided:
        raise ValueError(f"training-cache tensor {name!r} must be strided CPU data")
    if value.shape != shape or value.dtype is not dtype or value.requires_grad:
        raise ValueError(f"training-cache tensor {name!r} has an incompatible schema")
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise ValueError(f"training-cache tensor {name!r} contains NaN or Inf")
    # A fresh exact-span clone prevents loaded storage aliases from crossing
    # the validation boundary or reaching the objective adapter.
    return value.detach().contiguous().clone()


def _validated_tensor_group(
    value: object,
    specification: Mapping[str, tuple[tuple[int, ...], torch.dtype]],
    *,
    label: str,
) -> dict[str, Tensor]:
    mapping = _mapping(value, label=label)
    if set(mapping) != set(specification):
        raise ValueError(f"training-cache {label} tensor schema differs")
    return {
        name: _owned_tensor(
            mapping[name],
            name=f"{label}/{name}",
            shape=shape,
            dtype=dtype,
        )
        for name, (shape, dtype) in specification.items()
    }


def _flatten_content_tensors(content: Mapping[str, Any]) -> dict[str, Tensor]:
    tensors = {"timestamps": content["timestamps"]}
    for group in ("camera", "objects", "events"):
        values = _mapping(content[group], label=f"cache content {group}")
        tensors.update({f"{group}/{name}": value for name, value in values.items()})
    if not all(isinstance(value, Tensor) for value in tensors.values()):
        raise TypeError("training-cache content contains a non-tensor field")
    return tensors  # type: ignore[return-value]


def _content_metadata(envelope: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "accepted_seed": envelope["accepted_seed"],
        "attempt_count": envelope["attempt_count"],
        "rejection_reasons": envelope["rejection_reasons"],
        "certificate": envelope["certificate"],
    }


def _certificate_payload(certificate: DynamicSetSceneCertificate) -> dict[str, Any]:
    if not isinstance(certificate, DynamicSetSceneCertificate):
        raise TypeError("full materializer returned an invalid scene certificate")
    return asdict(certificate)


def _validated_certificate(value: object) -> DynamicSetSceneCertificate:
    mapping = _mapping(value, label="training-cache certificate")
    expected = set(DynamicSetSceneCertificate.__dataclass_fields__)
    if set(mapping) != expected:
        raise ValueError("training-cache certificate schema differs")
    try:
        certificate = DynamicSetSceneCertificate(**dict(mapping))
    except TypeError as error:
        raise ValueError("training-cache certificate values differ") from error
    for name in (
        "peak_object_count",
        "natural_pair_collision_count",
        "action_induced_pair_collision_count",
        "lifecycle_birth_count",
        "lifecycle_removal_count",
        "known_action_count",
    ):
        if type(getattr(certificate, name)) is not int or getattr(certificate, name) < 0:
            raise ValueError("training-cache certificate integer is invalid")
    for name in ("minimum_visible_fraction", "minimum_image_clearance_pixels"):
        value = getattr(certificate, name)
        if type(value) is not float or not torch.isfinite(torch.tensor(value)):
            raise ValueError("training-cache certificate scalar is invalid")
    return certificate


def _extract_compact_content(materialization: DynamicSetMaterialization) -> dict[str, Any]:
    episode = materialization.episode
    timestamps = _owned_tensor(
        episode.get("timestamps"),
        name="timestamps",
        shape=(DYNAMIC_SET_FRAMES,),
        dtype=torch.float32,
    )
    camera_source = _mapping(episode.get("camera"), label="full episode camera")
    camera: dict[str, Tensor] = {}
    for name, (shape, dtype) in _CAMERA_FIELDS.items():
        full = _owned_tensor(
            camera_source.get(name),
            name=f"camera/{name}",
            shape=(DYNAMIC_SET_FRAMES, *shape),
            dtype=dtype,
        )
        first = full[0].clone()
        if not torch.equal(full, first.unsqueeze(0).expand_as(full)):
            raise ValueError("lean training cache requires the frozen fixed camera trajectory")
        camera[name] = first
    object_source = _mapping(episode.get("objects"), label="full episode objects")
    objects = {
        name: _owned_tensor(
            object_source.get(name),
            name=f"objects/{name}",
            shape=shape,
            dtype=dtype,
        )
        for name, (shape, dtype) in _OBJECT_FIELDS.items()
    }
    event_source = _mapping(episode.get("events"), label="full episode events")
    events = {
        name: _owned_tensor(
            event_source.get(name),
            name=f"events/{name}",
            shape=shape,
            dtype=dtype,
        )
        for name, (shape, dtype) in _EVENT_FIELDS.items()
    }
    return {
        "timestamps": timestamps,
        "camera": camera,
        "objects": objects,
        "events": events,
    }


def _validate_compact_content(value: object, row: PhysicalManifestRow) -> dict[str, Any]:
    content = _mapping(value, label="training-cache content")
    if set(content) != {"timestamps", "camera", "objects", "events"}:
        raise ValueError("training-cache content schema differs")
    timestamps = _owned_tensor(
        content["timestamps"],
        name="timestamps",
        shape=(DYNAMIC_SET_FRAMES,),
        dtype=torch.float32,
    )
    expected_timestamps = torch.arange(DYNAMIC_SET_FRAMES, dtype=torch.float32) / float(
        DYNAMIC_SET_FRAME_RATE_HZ
    )
    if not torch.equal(timestamps, expected_timestamps):
        raise ValueError("training-cache timestamps differ from the frozen observation clock")
    camera = _validated_tensor_group(content["camera"], _CAMERA_FIELDS, label="camera")
    objects = _validated_tensor_group(content["objects"], _OBJECT_FIELDS, label="objects")
    events = _validated_tensor_group(content["events"], _EVENT_FIELDS, label="events")
    active = objects["active"]
    object_id = objects["id"]
    if bool((active & object_id.lt(0)).any()) or bool(((~active) & object_id.ne(-1)).any()):
        raise ValueError("training-cache active/ID trace is inconsistent")
    if int(active.sum(dim=-1).max()) != row.object_count:
        raise ValueError("training-cache cardinality differs from its manifest row")
    for name, expected in (
        ("radius", DYNAMIC_SET_RADIUS_M),
        ("mass", DYNAMIC_SET_MASS),
        ("restitution", DYNAMIC_SET_RESTITUTION),
        ("drag", DYNAMIC_SET_DRAG),
        ("friction", DYNAMIC_SET_FRICTION),
    ):
        observed = objects[name][active]
        if not torch.equal(observed, observed.new_full(observed.shape, expected)):
            raise ValueError(f"training-cache fixed {name} trace differs")
    if not torch.equal(events["externally_actuated"], events["known_action_observed"]):
        raise ValueError("training-cache contains hidden or unsupported actuation")
    known = events["known_action_observed"]
    if int(known.sum()) != int(row.known_action):
        raise ValueError("training-cache known-action trace differs from its manifest row")
    known_timestamp = events["known_action_timestamp"]
    known_impulse = events["known_impulse_world"]
    known_object_id = events["known_action_object_id"]
    if row.known_action:
        frame, slot = torch.nonzero(known, as_tuple=False)[0].tolist()
        if (
            known_timestamp[frame, slot] != timestamps[frame]
            or int(known_object_id[frame, slot]) != int(object_id[frame, slot])
            or not bool(known_impulse[frame, slot].ne(0).any())
            or bool(known_timestamp[~known].ne(-1.0).any())
            or bool(known_object_id[~known].ne(-1).any())
            or bool(known_impulse[~known].ne(0).any())
        ):
            raise ValueError("training-cache public action payload differs from its state trace")
    elif (
        bool(known_timestamp.ne(-1.0).any())
        or bool(known_impulse.ne(0).any())
        or bool(known_object_id.ne(-1).any())
    ):
        raise ValueError("training-cache action-free sentinels differ")
    pair_contact = events["pair_contact"]
    pair_collision = events["pair_collision"]
    if not torch.equal(pair_contact, pair_contact.transpose(-1, -2)) or not torch.equal(
        pair_collision,
        pair_collision.transpose(-1, -2),
    ):
        raise ValueError("training-cache pair event traces are not symmetric")
    if bool(pair_contact.any() or pair_collision.any()) != row.contact:
        raise ValueError("training-cache contact trace differs from its manifest row")
    return {
        "timestamps": timestamps,
        "camera": camera,
        "objects": objects,
        "events": events,
    }


class DynamicSetTrainingCache:
    """Content-addressed cache for optimizer-only dynamic-set materialization."""

    def __init__(
        self,
        root: str | Path,
        *,
        binding: DynamicSetTrainingCacheBinding,
        full_materializer: Callable[[PhysicalManifestRow], DynamicSetMaterialization] = (
            materialize_dynamic_set_episode
        ),
    ) -> None:
        if not callable(full_materializer):
            raise TypeError("full_materializer must be callable")
        self.binding = binding.validate()
        self.root = _validate_real_directory(Path(root), create=True)
        self.entries_root = _validate_real_directory(self.root / _ENTRIES_NAME, create=True)
        self._full_materializer = full_materializer
        self._memory: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._render_caches: dict[int, PrevalidatedSphereRenderCache] = {}
        self.cold_materializations = 0
        self.warm_hits = 0
        self.bytes_written = 0
        self.cold_materialization_seconds = 0.0
        self.warm_materialization_seconds = 0.0
        self._bind_namespace()
        validate_training_cache_directory(self.root)

    def __call__(self, row: PhysicalManifestRow) -> DynamicSetMaterialization:
        """Retain generic materializer compatibility outside the trainer path."""

        return self._full_materializer(row)

    def _namespace_payload(self) -> dict[str, Any]:
        return {
            "schema": TRAINING_CACHE_NAMESPACE_SCHEMA,
            **asdict(self.binding),
            "namespace_sha256": self.binding.namespace_sha256,
        }

    def _bind_namespace(self) -> None:
        path = self.root / _NAMESPACE_NAME
        expected = canonical_json_bytes(self._namespace_payload()) + b"\n"
        if path.exists() or path.is_symlink():
            actual = _stable_regular_bytes(path, maximum_bytes=_MAXIMUM_NAMESPACE_BYTES)
            if actual != expected:
                raise ValueError("training-cache namespace binding differs")
            return
        _atomic_write_fresh(path, expected)

    @staticmethod
    def _validate_row(row: PhysicalManifestRow) -> None:
        if not isinstance(row, PhysicalManifestRow) or row.split != "training":
            raise ValueError("lean training cache accepts only frozen training rows")
        size = PHYSICAL_SPLIT_SIZES["training"]
        if row.ordinal not in range(size):
            raise ValueError("training-cache row ordinal lies outside the frozen manifest")

    @staticmethod
    def _row_sha256(row: PhysicalManifestRow) -> str:
        return canonical_sha256(asdict(row))

    def _shard(self, row: PhysicalManifestRow, *, create: bool) -> Path:
        shard = self.entries_root / f"{row.ordinal // 1000:03d}"
        if not shard.exists() and not create:
            return shard
        return _validate_real_directory(shard, create=create)

    def _entry_candidates(self, row: PhysicalManifestRow) -> tuple[Path, ...]:
        shard = self._shard(row, create=False)
        if not shard.exists():
            return ()
        _validate_real_directory(shard, create=False)
        prefix = f"{row.ordinal:06d}-{self._row_sha256(row)}-"
        candidates: list[Path] = []
        for item in shard.iterdir():
            if item.name.startswith(".") and item.name.endswith(".tmp"):
                raise OSError("incomplete training-cache write requires review")
            if not item.name.startswith(prefix):
                continue
            suffix = item.name.removeprefix(prefix)
            if not suffix.endswith(".pt"):
                raise OSError("training-cache entry name has an invalid suffix")
            validated_sha256(suffix.removesuffix(".pt"), label="cache filename content")
            candidates.append(item)
        if len(candidates) > 1:
            raise OSError("training-cache row has ambiguous content-addressed entries")
        return tuple(candidates)

    def _entry_binding(self, row: PhysicalManifestRow) -> dict[str, Any]:
        return {
            "namespace_sha256": self.binding.namespace_sha256,
            "source_sha256": self.binding.source_sha256,
            "config_sha256": self.binding.config_sha256,
            "training_manifest_sha256": self.binding.training_manifest_sha256,
            "simulator_version": self.binding.simulator_version,
            "row_sha256": self._row_sha256(row),
            "row_ordinal": row.ordinal,
        }

    def _validated_envelope(
        self,
        raw: object,
        *,
        row: PhysicalManifestRow,
        filename_content_sha256: str,
    ) -> dict[str, Any]:
        envelope = _mapping(raw, label="training-cache entry")
        expected_keys = {
            "schema",
            "binding",
            "accepted_seed",
            "attempt_count",
            "rejection_reasons",
            "certificate",
            "content",
            "content_sha256",
        }
        if set(envelope) != expected_keys or envelope.get("schema") != TRAINING_CACHE_ENTRY_SCHEMA:
            raise ValueError("training-cache entry schema differs")
        if dict(_mapping(envelope["binding"], label="cache entry binding")) != self._entry_binding(
            row
        ):
            raise ValueError("training-cache entry binding differs")
        accepted_seed = envelope["accepted_seed"]
        attempt_count = envelope["attempt_count"]
        reasons = envelope["rejection_reasons"]
        if type(accepted_seed) is not int or accepted_seed < 0:
            raise ValueError("training-cache accepted seed is invalid")
        if type(attempt_count) is not int or attempt_count <= 0:
            raise ValueError("training-cache attempt count is invalid")
        if (
            type(reasons) is not list
            or len(reasons) != attempt_count - 1
            or any(type(reason) is not str for reason in reasons)
        ):
            raise ValueError("training-cache rejection audit is invalid")
        certificate = _validated_certificate(envelope["certificate"])
        if certificate.peak_object_count != row.object_count:
            raise ValueError("training-cache certificate cardinality differs")
        expected_lifecycle = {
            "none": (0, 0),
            "birth": (1, 0),
            "removal": (0, 1),
            "remove_then_birth": (1, 1),
        }[row.lifecycle_schedule]
        if (
            certificate.lifecycle_birth_count,
            certificate.lifecycle_removal_count,
        ) != expected_lifecycle:
            raise ValueError("training-cache certificate lifecycle differs")
        if certificate.known_action_count != int(row.known_action):
            raise ValueError("training-cache certificate action count differs")
        if certificate.minimum_visible_fraction != 1.0:
            raise ValueError("training-cache certificate does not prove complete visibility")
        if certificate.minimum_image_clearance_pixels < 0.0:
            raise ValueError("training-cache certificate does not prove image containment")
        natural_count = certificate.natural_pair_collision_count
        induced_count = certificate.action_induced_pair_collision_count
        if (
            (row.contact_origin == "none" and (natural_count != 0 or induced_count != 0))
            or (row.contact_origin == "natural" and (natural_count <= 0 or induced_count != 0))
            or (
                row.contact_origin == "action_induced"
                and (natural_count != 0 or induced_count <= 0)
            )
        ):
            raise ValueError("training-cache certificate contact provenance differs")
        content = _validate_compact_content(envelope["content"], row)
        claimed = validated_sha256(envelope["content_sha256"], label="cache content")
        actual = _tensor_tree_sha256(
            _content_metadata(envelope),
            _flatten_content_tensors(content),
        )
        if claimed != filename_content_sha256 or actual != claimed:
            raise ValueError("training-cache content digest differs")
        return {
            "schema": TRAINING_CACHE_ENTRY_SCHEMA,
            "binding": self._entry_binding(row),
            "accepted_seed": accepted_seed,
            "attempt_count": attempt_count,
            "rejection_reasons": list(reasons),
            "certificate": asdict(certificate),
            "content": content,
            "content_sha256": claimed,
        }

    def _load_entry(self, path: Path, row: PhysicalManifestRow) -> dict[str, Any]:
        contents = _stable_regular_bytes(path, maximum_bytes=_MAXIMUM_ENTRY_BYTES)
        try:
            raw = torch.load(io.BytesIO(contents), map_location="cpu", weights_only=True)
        except Exception as error:
            raise ValueError("training-cache entry is not a safe tensor payload") from error
        content_sha256 = path.name.rsplit("-", 1)[-1].removesuffix(".pt")
        return self._validated_envelope(
            raw,
            row=row,
            filename_content_sha256=content_sha256,
        )

    def _build_entry(self, row: PhysicalManifestRow) -> dict[str, Any]:
        materialization = self._full_materializer(row)
        if not isinstance(materialization, DynamicSetMaterialization):
            raise TypeError("full materializer must return DynamicSetMaterialization")
        if materialization.row != row:
            raise ValueError("full materializer returned a different manifest row")
        content = _extract_compact_content(materialization)
        envelope: dict[str, Any] = {
            "schema": TRAINING_CACHE_ENTRY_SCHEMA,
            "binding": self._entry_binding(row),
            "accepted_seed": materialization.accepted_seed,
            "attempt_count": materialization.attempt_count,
            "rejection_reasons": list(materialization.rejection_reasons),
            "certificate": _certificate_payload(materialization.certificate),
            "content": content,
        }
        envelope["content_sha256"] = _tensor_tree_sha256(
            _content_metadata(envelope),
            _flatten_content_tensors(content),
        )
        validated = self._validated_envelope(
            envelope,
            row=row,
            filename_content_sha256=envelope["content_sha256"],
        )
        stream = io.BytesIO()
        torch.save(validated, stream)
        contents = stream.getvalue()
        if not contents or len(contents) > _MAXIMUM_ENTRY_BYTES:
            raise ValueError("training-cache entry exceeds its persistence bound")
        shard = self._shard(row, create=True)
        path = shard / (
            f"{row.ordinal:06d}-{self._row_sha256(row)}-{validated['content_sha256']}.pt"
        )
        try:
            _atomic_write_fresh(path, contents)
            self.bytes_written += len(contents)
        except FileExistsError as error:
            existing = self._load_entry(path, row)
            if existing["content_sha256"] != validated["content_sha256"]:
                raise OSError("concurrent training-cache entry differs") from error
            validated = existing
        self.cold_materializations += 1
        return validated

    def _remember(self, row_sha256: str, entry: dict[str, Any]) -> None:
        self._memory[row_sha256] = entry
        self._memory.move_to_end(row_sha256)
        while len(self._memory) > _MEMORY_ENTRY_LIMIT:
            self._memory.popitem(last=False)

    def _entry(self, row: PhysicalManifestRow) -> tuple[dict[str, Any], bool]:
        row_sha256 = self._row_sha256(row)
        remembered = self._memory.get(row_sha256)
        if remembered is not None:
            self._memory.move_to_end(row_sha256)
            self.warm_hits += 1
            return remembered, True
        candidates = self._entry_candidates(row)
        if candidates:
            entry = self._load_entry(candidates[0], row)
            hit = True
            self.warm_hits += 1
        else:
            entry = self._build_entry(row)
            hit = False
        self._remember(row_sha256, entry)
        return entry, hit

    @staticmethod
    def _state_at(content: Mapping[str, Any], frame_index: int) -> SphereState:
        objects = _mapping(content["objects"], label="cache state objects")
        return SphereState(
            object_id=objects["id"][frame_index].clone(),
            active=objects["active"][frame_index].clone(),
            position=objects["position"][frame_index].clone(),
            velocity=objects["velocity"][frame_index].clone(),
            radius=objects["radius"][frame_index].clone(),
            mass=objects["mass"][frame_index].clone(),
            restitution=objects["restitution"][frame_index].clone(),
            drag=objects["drag"][frame_index].clone(),
            friction=objects["friction"][frame_index].clone(),
            albedo=objects["albedo"][frame_index].clone(),
            orientation=objects["orientation"][frame_index].clone(),
            angular_velocity=objects["angular_velocity"][frame_index].clone(),
            sleeping=objects["sleeping"][frame_index].clone(),
            sleep_counter=torch.zeros(DYNAMIC_SET_MAX_OBJECTS, dtype=torch.int64),
        )

    @staticmethod
    def _camera_at(content: Mapping[str, Any], frame_index: int) -> CameraFrame:
        camera = _mapping(content["camera"], label="cache static camera")
        timestamps = content["timestamps"]
        return CameraFrame(
            timestamp=float(timestamps[frame_index]),
            world_from_camera=camera["world_from_camera"].clone(),
            camera_from_world=camera["camera_from_world"].clone(),
            intrinsics=camera["intrinsics"].clone(),
            position=camera["position"].clone(),
            target=camera["target"].clone(),
        )

    @staticmethod
    def _clone_group(value: Mapping[str, Tensor]) -> dict[str, Tensor]:
        return {name: tensor.clone() for name, tensor in value.items()}

    def _training_materialization(
        self,
        row: PhysicalManifestRow,
        entry: Mapping[str, Any],
        *,
        perception_frame_index: int,
        cache_hit: bool,
    ) -> DynamicSetTrainingMaterialization:
        content = _mapping(entry["content"], label="validated cache content")
        state = self._state_at(content, perception_frame_index)
        camera = self._camera_at(content, perception_frame_index)
        render_cache = self._render_caches.get(row.camera_stratum)
        if render_cache is None:
            render_cache = prevalidated_sphere_render_cache(
                state,
                camera,
                DYNAMIC_SET_IMAGE_SIZE,
            )
            self._render_caches[row.camera_stratum] = render_cache
        rendered = render_spheres(
            state,
            camera,
            DYNAMIC_SET_IMAGE_SIZE,
            edge_softness_pixels=1.0,
            noise_std=0.0,
            _prevalidated_cache=render_cache,
        )
        labels = make_perception_labels(state, rendered, DYNAMIC_SET_IMAGE_SIZE)
        compact_camera = _mapping(content["camera"], label="validated cache camera")
        episode_camera = {
            name: value.unsqueeze(0).expand(DYNAMIC_SET_FRAMES, *value.shape).clone()
            for name, value in compact_camera.items()
        }
        objects = self._clone_group(
            _mapping(content["objects"], label="validated cache objects")  # type: ignore[arg-type]
        )
        events = self._clone_group(
            _mapping(content["events"], label="validated cache events")  # type: ignore[arg-type]
        )
        episode: dict[str, Any] = {
            "rgb": rendered.rgb.clone(),
            "depth": rendered.depth_buffer.unsqueeze(0).clone(),
            "timestamps": content["timestamps"].clone(),
            "camera": episode_camera,
            "objects": objects,
            "events": events,
            "labels": {"segmentation_mask": labels["segmentation_mask"].clone()},
            "metadata": {
                "training_materialization_schema": TRAINING_EPISODE_SCHEMA,
                "perception_frame_index": perception_frame_index,
                "cache_content_sha256": entry["content_sha256"],
            },
        }
        certificate = _validated_certificate(entry["certificate"])
        return DynamicSetTrainingMaterialization(
            row=row,
            episode=episode,
            known_action_observed=events["known_action_observed"].clone(),
            certificate=certificate,
            accepted_seed=int(entry["accepted_seed"]),
            attempt_count=int(entry["attempt_count"]),
            rejection_reasons=tuple(entry["rejection_reasons"]),
            cache_content_sha256=str(entry["content_sha256"]),
            cache_hit=cache_hit,
        )

    def materialize_for_training(
        self,
        row: PhysicalManifestRow,
        *,
        perception_frame_index: int,
    ) -> DynamicSetTrainingMaterialization:
        """Return one exact rotating-frame view after authenticating its trace."""

        self._validate_row(row)
        if (
            isinstance(perception_frame_index, bool)
            or not isinstance(perception_frame_index, int)
            or perception_frame_index not in range(DYNAMIC_SET_FRAMES)
        ):
            raise ValueError("perception_frame_index must lie in [0,55]")
        started = time.perf_counter()
        entry, cache_hit = self._entry(row)
        result = self._training_materialization(
            row,
            entry,
            perception_frame_index=perception_frame_index,
            cache_hit=cache_hit,
        )
        elapsed = time.perf_counter() - started
        if cache_hit:
            self.warm_materialization_seconds += elapsed
        else:
            self.cold_materialization_seconds += elapsed
        return result


__all__ = [
    "TRAINING_CACHE_DIRECTORY_NAME",
    "TRAINING_CACHE_ENTRY_SCHEMA",
    "TRAINING_CACHE_NAMESPACE_SCHEMA",
    "TRAINING_CACHE_SCHEMA",
    "TRAINING_EPISODE_SCHEMA",
    "DynamicSetTrainingCache",
    "DynamicSetTrainingCacheBinding",
    "DynamicSetTrainingMaterialization",
    "validate_training_cache_directory",
]
