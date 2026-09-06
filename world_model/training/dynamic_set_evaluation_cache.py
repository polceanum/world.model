"""Authenticated full-artifact cache for repeated development evaluation.

This cache is deliberately separate from the optimizer's lean training cache.
It retains complete public development episodes and planning-task
materializations so every later candidate sees the exact evaluator input from
the first certified validation.  Protected splits are rejected before any
filesystem lookup.

Disk entries are never trusted merely because they exist.  Before the first
qualification-sealed population root, an existing entry is accepted only
after the full public materializer is rerun and its semantic content digest is
identical.  Once the qualification supplies a trusted population evidence
record, the constructor verifies the complete ordered physical/planning
Merkle roots, and every read rechecks its content-addressed payload.
"""

from __future__ import annotations

import hashlib
import io
import math
import os
import stat
import time
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor

from world_model.planning import CounterfactualCostWeights
from world_model.simulator.episode import validate_episode
from world_model.training.dynamic_set_materializer import (
    DynamicSetMaterialization,
    materialize_dynamic_set_episode,
)
from world_model.training.dynamic_set_planning import (
    PlanningCandidateDescriptor,
    PlanningTaskConfig,
    PrivatePlanningOracleEvidence,
    PublicPlanningTemplate,
)
from world_model.training.dynamic_set_planning_materializer import (
    PlanningTaskMaterialization,
    PublicPlanningHistory,
    PublicPlanningHistoryFrame,
    _validate_complete_materialization,
    materialize_planning_task,
)
from world_model.training.dynamic_set_protocol import (
    SIMULATOR_VERSION,
    PhysicalManifestRow,
    PlanningManifestRow,
    PlanningOracleCertificate,
)
from world_model.training.dynamic_set_scene import (
    DynamicSetSceneCertificate,
    preflight_dynamic_set_episode,
)
from world_model.training.dynamic_set_training_cache import (
    _atomic_write_fresh,
    _stable_regular_bytes,
    _validate_real_directory,
)
from world_model.training.qualification_core import (
    canonical_json_bytes,
    canonical_sha256,
    validated_sha256,
)

DEVELOPMENT_EVALUATION_CACHE_DIRECTORY_NAME = "development_evaluation_cache"
DEVELOPMENT_EVALUATION_CACHE_SCHEMA = "dynamic_set_development_evaluation_cache_v1"
DEVELOPMENT_EVALUATION_CACHE_NAMESPACE_SCHEMA = (
    "dynamic_set_development_evaluation_cache_namespace_v1"
)
DEVELOPMENT_EVALUATION_CACHE_ENTRY_SCHEMA = "dynamic_set_development_evaluation_cache_entry_v1"
DEVELOPMENT_EVALUATION_CACHE_EVIDENCE_SCHEMA = (
    "dynamic_set_development_evaluation_cache_evidence_v1"
)
DEVELOPMENT_EVALUATION_CACHE_MERKLE_SCHEMA = "dynamic_set_development_evaluation_cache_merkle_v1"

_NAMESPACE_NAME = "namespace.json"
_KINDS = ("physical", "planning")
_MAXIMUM_NAMESPACE_BYTES = 32 * 1024
_MAXIMUM_COMPRESSED_ENTRY_BYTES = 96 * 1024 * 1024
_MAXIMUM_UNCOMPRESSED_ENTRY_BYTES = 128 * 1024 * 1024
CacheKind = Literal["physical", "planning"]


@dataclass(frozen=True, slots=True)
class DynamicSetDevelopmentEvaluationCacheBinding:
    source_sha256: str
    config_sha256: str
    physical_manifest_sha256: str
    planning_manifest_sha256: str
    evaluator_sha256: str
    simulator_version: str = SIMULATOR_VERSION

    def validate(self) -> DynamicSetDevelopmentEvaluationCacheBinding:
        for label, value in (
            ("source", self.source_sha256),
            ("config", self.config_sha256),
            ("physical manifest", self.physical_manifest_sha256),
            ("planning manifest", self.planning_manifest_sha256),
            ("evaluator", self.evaluator_sha256),
        ):
            validated_sha256(value, label=f"development evaluation cache {label}")
        if self.simulator_version != SIMULATOR_VERSION:
            raise ValueError("development cache simulator version differs from source")
        return self

    @property
    def namespace_sha256(self) -> str:
        return canonical_sha256(
            {
                "schema": DEVELOPMENT_EVALUATION_CACHE_NAMESPACE_SCHEMA,
                **asdict(self.validate()),
            }
        )


@dataclass(frozen=True, slots=True)
class DynamicSetDevelopmentEvaluationCacheEvidence:
    namespace_sha256: str
    physical_entry_count: int
    planning_entry_count: int
    physical_merkle_sha256: str
    planning_merkle_sha256: str
    population_merkle_sha256: str
    evidence_sha256: str

    @classmethod
    def create(
        cls,
        *,
        binding: DynamicSetDevelopmentEvaluationCacheBinding,
        physical_entry_count: int,
        planning_entry_count: int,
        physical_merkle_sha256: str,
        planning_merkle_sha256: str,
    ) -> DynamicSetDevelopmentEvaluationCacheEvidence:
        binding.validate()
        body = {
            "schema": DEVELOPMENT_EVALUATION_CACHE_EVIDENCE_SCHEMA,
            "namespace_sha256": binding.namespace_sha256,
            "physical_entry_count": physical_entry_count,
            "planning_entry_count": planning_entry_count,
            "physical_merkle_sha256": physical_merkle_sha256,
            "planning_merkle_sha256": planning_merkle_sha256,
        }
        population = canonical_sha256(
            {
                "schema": "dynamic_set_development_evaluation_population_root_v1",
                **body,
            }
        )
        evidence_body = {**body, "population_merkle_sha256": population}
        return cls(
            namespace_sha256=binding.namespace_sha256,
            physical_entry_count=physical_entry_count,
            planning_entry_count=planning_entry_count,
            physical_merkle_sha256=physical_merkle_sha256,
            planning_merkle_sha256=planning_merkle_sha256,
            population_merkle_sha256=population,
            evidence_sha256=canonical_sha256(evidence_body),
        ).validate(binding=binding)

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        binding: DynamicSetDevelopmentEvaluationCacheBinding,
    ) -> DynamicSetDevelopmentEvaluationCacheEvidence:
        expected = {
            "schema",
            "namespace_sha256",
            "physical_entry_count",
            "planning_entry_count",
            "physical_merkle_sha256",
            "planning_merkle_sha256",
            "population_merkle_sha256",
            "evidence_sha256",
        }
        if type(value) is not dict or set(value) != expected:
            raise ValueError("development cache evidence schema differs")
        if value["schema"] != DEVELOPMENT_EVALUATION_CACHE_EVIDENCE_SCHEMA:
            raise ValueError("development cache evidence version differs")
        return cls(**{name: value[name] for name in expected if name != "schema"}).validate(
            binding=binding
        )

    def validate(
        self,
        *,
        binding: DynamicSetDevelopmentEvaluationCacheBinding,
        physical_entry_count: int | None = None,
        planning_entry_count: int | None = None,
    ) -> DynamicSetDevelopmentEvaluationCacheEvidence:
        binding.validate()
        for name, value in (
            ("physical_entry_count", self.physical_entry_count),
            ("planning_entry_count", self.planning_entry_count),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"development cache {name} must be positive")
        for label, value in (
            ("namespace", self.namespace_sha256),
            ("physical Merkle", self.physical_merkle_sha256),
            ("planning Merkle", self.planning_merkle_sha256),
            ("population Merkle", self.population_merkle_sha256),
            ("evidence", self.evidence_sha256),
        ):
            validated_sha256(value, label=f"development cache {label}")
        body = {
            "schema": DEVELOPMENT_EVALUATION_CACHE_EVIDENCE_SCHEMA,
            "namespace_sha256": self.namespace_sha256,
            "physical_entry_count": self.physical_entry_count,
            "planning_entry_count": self.planning_entry_count,
            "physical_merkle_sha256": self.physical_merkle_sha256,
            "planning_merkle_sha256": self.planning_merkle_sha256,
        }
        population = canonical_sha256(
            {
                "schema": "dynamic_set_development_evaluation_population_root_v1",
                **body,
            }
        )
        evidence_body = {**body, "population_merkle_sha256": population}
        if (
            self.namespace_sha256 != binding.namespace_sha256
            or self.population_merkle_sha256 != population
            or self.evidence_sha256 != canonical_sha256(evidence_body)
            or (
                physical_entry_count is not None
                and self.physical_entry_count != physical_entry_count
            )
            or (
                planning_entry_count is not None
                and self.planning_entry_count != planning_entry_count
            )
        ):
            raise ValueError("development cache evidence binding differs")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DEVELOPMENT_EVALUATION_CACHE_EVIDENCE_SCHEMA,
            **asdict(self),
        }


def _validated_population(
    rows: Sequence[PhysicalManifestRow] | Sequence[PlanningManifestRow],
    *,
    kind: CacheKind,
    manifest_sha256: str,
) -> tuple[PhysicalManifestRow, ...] | tuple[PlanningManifestRow, ...]:
    expected_type = PhysicalManifestRow if kind == "physical" else PlanningManifestRow
    values = tuple(rows)
    if not values or any(type(row) is not expected_type for row in values):
        raise TypeError(f"development cache {kind} rows have an incompatible type")
    if any(row.split != "development" for row in values):
        raise PermissionError("development cache rejects every non-development row")
    ordinals = tuple(row.ordinal for row in values)
    if any(type(ordinal) is not int or ordinal < 0 for ordinal in ordinals) or ordinals != tuple(
        sorted(set(ordinals))
    ):
        raise ValueError(f"development cache {kind} rows must use unique manifest order")
    if canonical_sha256([asdict(row) for row in values]) != manifest_sha256:
        raise ValueError(f"development cache {kind} manifest binding differs")
    return values


def _tree_digest(value: object) -> str:
    tensors: list[tuple[str, Tensor]] = []

    def structure(item: object, path: str) -> object:
        if isinstance(item, Tensor):
            if item.device.type != "cpu" or item.layout is not torch.strided or item.requires_grad:
                raise ValueError("development cache tensors must be detached strided CPU data")
            tensor = item.detach().contiguous()
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise ValueError("development cache tensors must be finite")
            tensors.append((path, tensor))
            return {
                "tensor": len(tensors) - 1,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            }
        if type(item) is dict:
            if any(type(key) is not str or not key for key in item):
                raise TypeError("development cache mappings require nonempty string keys")
            return {
                "mapping": [[key, structure(item[key], f"{path}/{key}")] for key in sorted(item)]
            }
        if type(item) is tuple:
            return {
                "tuple": [structure(value, f"{path}/{index}") for index, value in enumerate(item)]
            }
        if type(item) is list:
            return {
                "list": [structure(value, f"{path}/{index}") for index, value in enumerate(item)]
            }
        if item is None or type(item) in {str, bool, int}:
            return item
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError("development cache scalar metadata must be finite")
            return item
        raise TypeError(f"unsupported development cache value at {path}: {type(item).__name__}")

    metadata = structure(value, "root")
    digest = hashlib.sha256()
    header = canonical_json_bytes(metadata)
    digest.update(len(header).to_bytes(8, byteorder="big"))
    digest.update(header)
    for path, tensor in tensors:
        tensor_header = canonical_json_bytes(
            {
                "path": path,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            }
        )
        digest.update(len(tensor_header).to_bytes(8, byteorder="big"))
        digest.update(tensor_header)
        raw = tensor.numpy().tobytes(order="C")
        digest.update(len(raw).to_bytes(8, byteorder="big"))
        digest.update(raw)
    return digest.hexdigest()


def _owned_tree(value: object) -> object:
    if isinstance(value, Tensor):
        if value.device.type != "cpu" or value.layout is not torch.strided or value.requires_grad:
            raise ValueError("development cache tensors must be detached strided CPU data")
        tensor = value.detach().contiguous().clone()
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError("development cache tensors must be finite")
        return tensor
    if type(value) is dict:
        if any(type(key) is not str or not key for key in value):
            raise TypeError("development cache mappings require nonempty string keys")
        return {key: _owned_tree(item) for key, item in value.items()}
    if type(value) is tuple:
        return tuple(_owned_tree(item) for item in value)
    if type(value) is list:
        return [_owned_tree(item) for item in value]
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise TypeError(f"development cache contains unsupported value {type(value).__name__}")


def _compressed_payload(value: Mapping[str, Any]) -> bytes:
    stream = io.BytesIO()
    torch.save(dict(value), stream)
    raw = stream.getvalue()
    if not raw or len(raw) > _MAXIMUM_UNCOMPRESSED_ENTRY_BYTES:
        raise ValueError("development cache entry exceeds its uncompressed bound")
    compressed = zlib.compress(raw, level=1)
    if not compressed or len(compressed) > _MAXIMUM_COMPRESSED_ENTRY_BYTES:
        raise ValueError("development cache entry exceeds its compressed bound")
    return compressed


def _decompressed_payload(contents: bytes) -> dict[str, Any]:
    decoder = zlib.decompressobj()
    raw = decoder.decompress(contents, _MAXIMUM_UNCOMPRESSED_ENTRY_BYTES + 1)
    if (
        len(raw) > _MAXIMUM_UNCOMPRESSED_ENTRY_BYTES
        or decoder.unconsumed_tail
        or not decoder.eof
        or decoder.unused_data
    ):
        raise ValueError("development cache entry has invalid compressed framing")
    try:
        value = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("development cache entry is not a safe tensor payload") from error
    if type(value) is not dict:
        raise ValueError("development cache entry payload must be one mapping")
    return value


def _validate_physical_acceptance_audit(
    materialization: DynamicSetMaterialization,
) -> DynamicSetMaterialization:
    validate_episode(materialization.episode)
    if (
        materialization.episode.get("seed") != materialization.accepted_seed
        or type(materialization.accepted_seed) is not int
        or type(materialization.attempt_count) is not int
        or materialization.attempt_count <= 0
        or len(materialization.rejection_reasons) != materialization.attempt_count - 1
        or any(
            type(reason) is not str or not reason for reason in materialization.rejection_reasons
        )
    ):
        raise ValueError("development physical cache acceptance audit differs")
    return materialization


def _physical_content(materialization: DynamicSetMaterialization) -> dict[str, Any]:
    if not isinstance(materialization, DynamicSetMaterialization):
        raise TypeError("physical materializer returned an incompatible artifact")
    _validate_physical_acceptance_audit(materialization)
    return _owned_tree(
        {
            "schema": "dynamic_set_development_physical_artifact_v1",
            "row": asdict(materialization.row),
            "episode": materialization.episode,
            "known_action_observed": materialization.known_action_observed,
            "certificate": asdict(materialization.certificate),
            "accepted_seed": materialization.accepted_seed,
            "attempt_count": materialization.attempt_count,
            "rejection_reasons": materialization.rejection_reasons,
        }
    )  # type: ignore[return-value]


def _planning_content(materialization: PlanningTaskMaterialization) -> dict[str, Any]:
    if not isinstance(materialization, PlanningTaskMaterialization):
        raise TypeError("planning materializer returned an incompatible artifact")
    return _owned_tree(
        {
            "schema": "dynamic_set_development_planning_artifact_v1",
            **asdict(materialization),
        }
    )  # type: ignore[return-value]


def _physical_from_content(
    content: Mapping[str, Any],
    *,
    row: PhysicalManifestRow,
) -> DynamicSetMaterialization:
    expected = {
        "schema",
        "row",
        "episode",
        "known_action_observed",
        "certificate",
        "accepted_seed",
        "attempt_count",
        "rejection_reasons",
    }
    if (
        set(content) != expected
        or content["schema"] != "dynamic_set_development_physical_artifact_v1"
    ):
        raise ValueError("development physical cache content schema differs")
    if content["row"] != asdict(row):
        raise ValueError("development physical cache row differs")
    episode = _owned_tree(content["episode"])
    known_action = _owned_tree(content["known_action_observed"])
    certificate_raw = content["certificate"]
    if type(episode) is not dict or not isinstance(known_action, Tensor):
        raise TypeError("development physical cache episode/action payload differs")
    if type(certificate_raw) is not dict:
        raise TypeError("development physical cache certificate payload differs")
    certificate = DynamicSetSceneCertificate(**certificate_raw)
    materialization = DynamicSetMaterialization(
        row=row,
        episode=episode,
        known_action_observed=known_action,
        certificate=certificate,
        accepted_seed=content["accepted_seed"],
        attempt_count=content["attempt_count"],
        rejection_reasons=tuple(content["rejection_reasons"]),
    )
    _validate_physical_acceptance_audit(materialization)
    # The full preflight can be replayed without private paired counterfactual
    # evidence for every cell except known-action contact.  Those exceptional
    # artifacts are still protected by first-use full rematerialization and
    # the qualification-sealed population Merkle root.
    if not (row.known_action and row.contact):
        rebuilt = preflight_dynamic_set_episode(
            materialization.episode,
            row,
            known_action_observed=materialization.known_action_observed,
        )
        if rebuilt != materialization.certificate:
            raise ValueError("development physical cache certificate differs")
    return materialization


def _planning_from_content(
    content: Mapping[str, Any],
    *,
    row: PlanningManifestRow,
) -> PlanningTaskMaterialization:
    expected = {
        "schema",
        "public_history",
        "template",
        "private_oracle",
        "accepted_seed",
        "attempt_count",
        "rejection_reasons",
        "materialization_sha256",
    }
    if (
        set(content) != expected
        or content["schema"] != "dynamic_set_development_planning_artifact_v1"
    ):
        raise ValueError("development planning cache content schema differs")
    history_raw = content["public_history"]
    template_raw = content["template"]
    oracle_raw = content["private_oracle"]
    if not all(type(value) is dict for value in (history_raw, template_raw, oracle_raw)):
        raise TypeError("development planning cache nested schema differs")
    frames = tuple(PublicPlanningHistoryFrame(**frame) for frame in history_raw["frames"])
    history = PublicPlanningHistory(
        frames=frames,
        previously_dynamic=history_raw["previously_dynamic"],
        history_sha256=history_raw["history_sha256"],
    )
    if template_raw["row"] != asdict(row):
        raise ValueError("development planning cache row differs")
    config_raw = dict(template_raw["config"])
    config_raw["cost_weights"] = CounterfactualCostWeights(**config_raw["cost_weights"])
    config = PlanningTaskConfig(**config_raw)
    template = PublicPlanningTemplate(
        row=row,
        source_timestamp=template_raw["source_timestamp"],
        appearance_handle=template_raw["appearance_handle"],
        query_offsets=template_raw["query_offsets"],
        goal_position_world=template_raw["goal_position_world"],
        candidate_timestamps=template_raw["candidate_timestamps"],
        candidate_impulses_world=template_raw["candidate_impulses_world"],
        candidate_descriptors=tuple(
            PlanningCandidateDescriptor(**descriptor)
            for descriptor in template_raw["candidate_descriptors"]
        ),
        config=config,
        template_sha256=template_raw["template_sha256"],
    )
    oracle = PrivatePlanningOracleEvidence(
        template_sha256=oracle_raw["template_sha256"],
        candidate_costs=oracle_raw["candidate_costs"],
        candidate_terminal_goal_distance_m=oracle_raw["candidate_terminal_goal_distance_m"],
        candidate_goal_success=oracle_raw["candidate_goal_success"],
        candidate_induced_contact=oracle_raw["candidate_induced_contact"],
        certificate=PlanningOracleCertificate(**oracle_raw["certificate"]),
        regret_scale=oracle_raw["regret_scale"],
        evidence_sha256=oracle_raw["evidence_sha256"],
    )
    materialization = PlanningTaskMaterialization(
        public_history=history,
        template=template,
        private_oracle=oracle,
        accepted_seed=content["accepted_seed"],
        attempt_count=content["attempt_count"],
        rejection_reasons=tuple(content["rejection_reasons"]),
        materialization_sha256=content["materialization_sha256"],
    )
    return _validate_complete_materialization(materialization)


def _merkle_sha256(leaves: Sequence[Mapping[str, Any]], *, kind: CacheKind) -> str:
    if not leaves:
        raise ValueError("development cache cannot seal an empty population")
    level = [hashlib.sha256(b"\x00" + canonical_json_bytes(dict(leaf))).digest() for leaf in leaves]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [
            hashlib.sha256(b"\x01" + level[index] + level[index + 1]).digest()
            for index in range(0, len(level), 2)
        ]
    return canonical_sha256(
        {
            "schema": DEVELOPMENT_EVALUATION_CACHE_MERKLE_SCHEMA,
            "kind": kind,
            "leaf_count": len(leaves),
            "binary_root_sha256": level[0].hex(),
        }
    )


class DynamicSetDevelopmentEvaluationCache:
    """Complete development-only physical/planning materialization cache."""

    def __init__(
        self,
        root: str | Path,
        *,
        binding: DynamicSetDevelopmentEvaluationCacheBinding,
        physical_rows: Sequence[PhysicalManifestRow],
        planning_rows: Sequence[PlanningManifestRow],
        trusted_evidence: Mapping[str, Any]
        | DynamicSetDevelopmentEvaluationCacheEvidence
        | None = None,
        physical_materializer: Callable[[PhysicalManifestRow], DynamicSetMaterialization] = (
            materialize_dynamic_set_episode
        ),
        planning_materializer: Callable[[PlanningManifestRow], PlanningTaskMaterialization] = (
            materialize_planning_task
        ),
    ) -> None:
        if not callable(physical_materializer) or not callable(planning_materializer):
            raise TypeError("development cache materializers must be callable")
        self.binding = binding.validate()
        self.physical_rows = _validated_population(
            physical_rows,
            kind="physical",
            manifest_sha256=binding.physical_manifest_sha256,
        )
        self.planning_rows = _validated_population(
            planning_rows,
            kind="planning",
            manifest_sha256=binding.planning_manifest_sha256,
        )
        self._physical_by_ordinal = {row.ordinal: row for row in self.physical_rows}
        self._planning_by_ordinal = {row.ordinal: row for row in self.planning_rows}
        self.root = _validate_real_directory(Path(root), create=True)
        self._roots = {
            kind: _validate_real_directory(self.root / kind, create=True) for kind in _KINDS
        }
        self._physical_materializer = physical_materializer
        self._planning_materializer = planning_materializer
        self._entry_index: dict[tuple[str, int, str], list[Path]] = {}
        self.cold_materializations = 0
        self.recertified_materializations = 0
        self.warm_hits = 0
        self.bytes_written = 0
        self.cold_materialization_seconds = 0.0
        self.warm_materialization_seconds = 0.0
        self._bind_namespace()
        self._validate_inventory()
        if trusted_evidence is None:
            self.trusted_evidence = None
        elif isinstance(trusted_evidence, DynamicSetDevelopmentEvaluationCacheEvidence):
            self.trusted_evidence = trusted_evidence.validate(
                binding=self.binding,
                physical_entry_count=len(self.physical_rows),
                planning_entry_count=len(self.planning_rows),
            )
        else:
            self.trusted_evidence = DynamicSetDevelopmentEvaluationCacheEvidence.from_mapping(
                trusted_evidence,
                binding=self.binding,
            ).validate(
                binding=self.binding,
                physical_entry_count=len(self.physical_rows),
                planning_entry_count=len(self.planning_rows),
            )
        if self.trusted_evidence is not None:
            rebuilt = self._complete_evidence()
            if rebuilt != self.trusted_evidence:
                raise ValueError("development cache population differs from sealed evidence")

    def _namespace_payload(self) -> dict[str, Any]:
        return {
            "schema": DEVELOPMENT_EVALUATION_CACHE_NAMESPACE_SCHEMA,
            **asdict(self.binding),
            "namespace_sha256": self.binding.namespace_sha256,
        }

    def _bind_namespace(self) -> None:
        path = self.root / _NAMESPACE_NAME
        expected = canonical_json_bytes(self._namespace_payload()) + b"\n"
        if path.exists() or path.is_symlink():
            if _stable_regular_bytes(path, maximum_bytes=_MAXIMUM_NAMESPACE_BYTES) != expected:
                raise ValueError("development cache namespace binding differs")
            return
        _atomic_write_fresh(path, expected)

    def _validate_inventory(self) -> None:
        allowed = {_NAMESPACE_NAME, *_KINDS}
        for item in self.root.iterdir():
            if item.name not in allowed:
                raise OSError(f"development cache contains unsupported artifact: {item.name}")
            metadata = os.lstat(item)
            if item.name == _NAMESPACE_NAME:
                if (
                    stat.S_ISLNK(metadata.st_mode)
                    or not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_nlink != 1
                ):
                    raise OSError("development cache namespace must be single-link regular data")
            elif stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise OSError("development cache population paths must be real directories")
        entry_index: dict[tuple[str, int, str], list[Path]] = {}
        for kind, root in self._roots.items():
            for shard in root.iterdir():
                metadata = os.lstat(shard)
                if (
                    len(shard.name) != 3
                    or not shard.name.isascii()
                    or not shard.name.isdigit()
                    or stat.S_ISLNK(metadata.st_mode)
                    or not stat.S_ISDIR(metadata.st_mode)
                ):
                    raise OSError("development cache shard inventory differs")
                for entry in shard.iterdir():
                    entry_metadata = os.lstat(entry)
                    if entry.name.startswith(".") and entry.name.endswith(".tmp"):
                        raise OSError("incomplete development cache write requires review")
                    if (
                        stat.S_ISLNK(entry_metadata.st_mode)
                        or not stat.S_ISREG(entry_metadata.st_mode)
                        or entry_metadata.st_nlink != 1
                    ):
                        raise OSError("development cache entry must be single-link regular data")
                    if not entry.name.endswith(".ptz"):
                        raise OSError("development cache entry filename differs")
                    fields = entry.name.removesuffix(".ptz").split("-")
                    if len(fields) != 4 or len(fields[0]) != 6 or not fields[0].isdigit():
                        raise OSError("development cache entry filename differs")
                    ordinal = int(fields[0])
                    for label, digest in zip(
                        ("row", "content", "artifact"), fields[1:], strict=True
                    ):
                        validated_sha256(digest, label=f"development cache {label}")
                    entry_index.setdefault((kind, ordinal, fields[1]), []).append(entry)
        self._entry_index = entry_index

    def _rows(self, kind: CacheKind) -> Mapping[int, Any]:
        return self._physical_by_ordinal if kind == "physical" else self._planning_by_ordinal

    def _validate_row(self, row: object, *, kind: CacheKind) -> None:
        expected = self._rows(kind).get(getattr(row, "ordinal", None))
        if row != expected:
            if getattr(row, "split", None) != "development":
                raise PermissionError("development cache rejects every non-development row")
            raise ValueError(f"development cache {kind} row differs from its bound manifest")

    def _shard(self, kind: CacheKind, ordinal: int, *, create: bool) -> Path:
        shard = self._roots[kind] / f"{ordinal // 1000:03d}"
        if not shard.exists() and not create:
            return shard
        return _validate_real_directory(shard, create=create)

    @staticmethod
    def _row_sha256(row: PhysicalManifestRow | PlanningManifestRow) -> str:
        return canonical_sha256(asdict(row))

    def _entry_candidates(
        self,
        kind: CacheKind,
        row: PhysicalManifestRow | PlanningManifestRow,
    ) -> tuple[Path, ...]:
        candidates = self._entry_index.get((kind, row.ordinal, self._row_sha256(row)), [])
        if len(candidates) > 1:
            raise OSError("development cache row has ambiguous content-addressed entries")
        return tuple(candidates)

    def _envelope(
        self,
        *,
        kind: CacheKind,
        row: PhysicalManifestRow | PlanningManifestRow,
        content: Mapping[str, Any],
    ) -> tuple[dict[str, Any], str]:
        content_sha256 = _tree_digest(content)
        envelope = {
            "schema": DEVELOPMENT_EVALUATION_CACHE_ENTRY_SCHEMA,
            "kind": kind,
            "namespace_sha256": self.binding.namespace_sha256,
            "row_ordinal": row.ordinal,
            "row_sha256": self._row_sha256(row),
            "content_sha256": content_sha256,
            "content": dict(content),
        }
        return envelope, content_sha256

    def _read_content(
        self,
        path: Path,
        *,
        kind: CacheKind,
        row: PhysicalManifestRow | PlanningManifestRow,
    ) -> tuple[dict[str, Any], str]:
        contents = _stable_regular_bytes(path, maximum_bytes=_MAXIMUM_COMPRESSED_ENTRY_BYTES)
        raw = _decompressed_payload(contents)
        expected = {
            "schema",
            "kind",
            "namespace_sha256",
            "row_ordinal",
            "row_sha256",
            "content_sha256",
            "content",
        }
        if set(raw) != expected or raw["schema"] != DEVELOPMENT_EVALUATION_CACHE_ENTRY_SCHEMA:
            raise ValueError("development cache entry schema differs")
        content = raw["content"]
        if type(content) is not dict:
            raise TypeError("development cache content must be one mapping")
        content_sha256 = _tree_digest(content)
        filename_content_sha256, filename_artifact_sha256 = path.name.removesuffix(".ptz").rsplit(
            "-", maxsplit=2
        )[-2:]
        if (
            raw["kind"] != kind
            or raw["namespace_sha256"] != self.binding.namespace_sha256
            or raw["row_ordinal"] != row.ordinal
            or raw["row_sha256"] != self._row_sha256(row)
            or raw["content_sha256"] != content_sha256
            or filename_content_sha256 != content_sha256
            or filename_artifact_sha256 != hashlib.sha256(contents).hexdigest()
        ):
            raise ValueError("development cache entry binding differs")
        return content, content_sha256

    def _write_content(
        self,
        *,
        kind: CacheKind,
        row: PhysicalManifestRow | PlanningManifestRow,
        content: Mapping[str, Any],
    ) -> str:
        envelope, content_sha256 = self._envelope(kind=kind, row=row, content=content)
        shard = self._shard(kind, row.ordinal, create=True)
        contents = _compressed_payload(envelope)
        artifact_sha256 = hashlib.sha256(contents).hexdigest()
        path = shard / (
            f"{row.ordinal:06d}-{self._row_sha256(row)}-{content_sha256}-{artifact_sha256}.ptz"
        )
        _atomic_write_fresh(path, contents)
        self._entry_index.setdefault((kind, row.ordinal, self._row_sha256(row)), []).append(path)
        self.bytes_written += len(contents)
        return content_sha256

    def _materialize(self, row: Any, *, kind: CacheKind) -> Any:
        self._validate_row(row, kind=kind)
        started = time.monotonic()
        candidates = self._entry_candidates(kind, row)
        if self.trusted_evidence is not None:
            if len(candidates) != 1:
                raise FileNotFoundError("sealed development cache entry is absent")
            content, _digest = self._read_content(candidates[0], kind=kind, row=row)
            result = (
                _physical_from_content(content, row=row)
                if kind == "physical"
                else _planning_from_content(content, row=row)
            )
            self.warm_hits += 1
            self.warm_materialization_seconds += time.monotonic() - started
            return result

        materializer = (
            self._physical_materializer if kind == "physical" else self._planning_materializer
        )
        result = materializer(row)
        content = _physical_content(result) if kind == "physical" else _planning_content(result)
        content_sha256 = _tree_digest(content)
        if candidates:
            cached, cached_sha256 = self._read_content(candidates[0], kind=kind, row=row)
            if cached_sha256 != content_sha256 or _tree_digest(cached) != content_sha256:
                raise ValueError("uncertified development cache differs from rematerialization")
            self.recertified_materializations += 1
        else:
            self._write_content(kind=kind, row=row, content=content)
            self.cold_materializations += 1
        self.cold_materialization_seconds += time.monotonic() - started
        return result

    def physical(self, row: PhysicalManifestRow) -> DynamicSetMaterialization:
        return self._materialize(row, kind="physical")

    def planning(self, row: PlanningManifestRow) -> PlanningTaskMaterialization:
        return self._materialize(row, kind="planning")

    def _leaf(
        self,
        *,
        kind: CacheKind,
        row: PhysicalManifestRow | PlanningManifestRow,
    ) -> dict[str, Any]:
        candidates = self._entry_candidates(kind, row)
        if len(candidates) != 1:
            raise FileNotFoundError("development cache population is incomplete")
        content_sha256, artifact_sha256 = (
            candidates[0].name.removesuffix(".ptz").rsplit("-", maxsplit=2)[-2:]
        )
        validated_sha256(content_sha256, label="development cache leaf content")
        validated_sha256(artifact_sha256, label="development cache leaf artifact")
        return {
            "kind": kind,
            "ordinal": row.ordinal,
            "row_sha256": self._row_sha256(row),
            "content_sha256": content_sha256,
            "artifact_sha256": artifact_sha256,
        }

    def _complete_evidence(self) -> DynamicSetDevelopmentEvaluationCacheEvidence:
        self._validate_inventory()
        physical_leaves = [self._leaf(kind="physical", row=row) for row in self.physical_rows]
        planning_leaves = [self._leaf(kind="planning", row=row) for row in self.planning_rows]
        expected_paths = {
            kind: {
                self._entry_candidates(kind, row)[0]
                for row in (self.physical_rows if kind == "physical" else self.planning_rows)
            }
            for kind in _KINDS
        }
        for kind in _KINDS:
            actual = {
                entry
                for shard in self._roots[kind].iterdir()
                for entry in shard.iterdir()
                if not entry.name.startswith(".")
            }
            if actual != expected_paths[kind]:
                raise OSError("development cache contains an entry outside its bound manifest")
        return DynamicSetDevelopmentEvaluationCacheEvidence.create(
            binding=self.binding,
            physical_entry_count=len(physical_leaves),
            planning_entry_count=len(planning_leaves),
            physical_merkle_sha256=_merkle_sha256(physical_leaves, kind="physical"),
            planning_merkle_sha256=_merkle_sha256(planning_leaves, kind="planning"),
        )

    def seal_complete(self) -> DynamicSetDevelopmentEvaluationCacheEvidence:
        """Seal the exact complete population after first-use certification."""

        evidence = self._complete_evidence()
        self.trusted_evidence = evidence
        return evidence


def validate_development_evaluation_cache_directory(path: str | Path) -> Path:
    """Validate an existing cache inventory without accepting its contents."""

    root = _validate_real_directory(Path(path), create=False)
    allowed = {_NAMESPACE_NAME, *_KINDS}
    if {item.name for item in root.iterdir()} != allowed:
        raise OSError("development evaluation cache root inventory differs")
    for item in root.iterdir():
        metadata = os.lstat(item)
        if item.name == _NAMESPACE_NAME:
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
            ):
                raise OSError("development cache namespace must be single-link regular data")
        elif stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise OSError("development cache population path must be a real directory")
    return root


__all__ = [
    "DEVELOPMENT_EVALUATION_CACHE_DIRECTORY_NAME",
    "DEVELOPMENT_EVALUATION_CACHE_ENTRY_SCHEMA",
    "DEVELOPMENT_EVALUATION_CACHE_EVIDENCE_SCHEMA",
    "DEVELOPMENT_EVALUATION_CACHE_NAMESPACE_SCHEMA",
    "DEVELOPMENT_EVALUATION_CACHE_SCHEMA",
    "DynamicSetDevelopmentEvaluationCache",
    "DynamicSetDevelopmentEvaluationCacheBinding",
    "DynamicSetDevelopmentEvaluationCacheEvidence",
    "validate_development_evaluation_cache_directory",
]
