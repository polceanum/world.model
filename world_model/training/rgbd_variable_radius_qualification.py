"""Hardened seedless qualification for per-object metric-radius RGB-D inference.

The governed family is addressed only by an exact conceptual split and an
ordinal in 0..63. No random seed, manifest file, simulator constructor, or
public renderer participates. A live durable ledger owns single-use
four-ordinal capabilities; only that capability may synthesize independent
RGB-D packets and pass them through OnlineWorldModel.

Development produces an unchanged empty-model-state checkpoint and reviewable
report. Protected access requires exact externally reviewed SHA-256 values for
the checkpoint, report, and terminal development ledger. Selector,
confirmation, and final-test are then consumed in that order and at most once.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import inspect
import io
import json
import math
import os
import resource
import stat
import subprocess
import sys
import threading
import time
import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import torch
import yaml
from torch import Tensor

from world_model.belief import slow_packing_map
from world_model.observations import MeasurementSet, ObservationPacket
from world_model.runtime import OnlineWorldModel
from world_model.training.rgbd_online_bridge_qualification import (
    clean_source,
)
from world_model.training.rgbd_variable_radius_scene import (
    CAMERA_STRATA,
    FRAME_COUNT,
    FRAME_RATE_HZ,
    FROZEN_CERTIFICATE_SHA256,
    FROZEN_SPLIT_TRACE_SHA256,
    FROZEN_TRACE_SHA256,
    HISTORY_FRAME_COUNT,
    IMAGE_SIZE,
    PAIR_VARIANTS_PER_PRIMITIVE,
    PRIMITIVES_PER_SPLIT,
    RADIUS_ROLES_PER_PRIMITIVE,
    SCENES_PER_SPLIT,
    SPLITS,
    counterfactual_twin_ordinal,
    manual_kinematic_trajectory,
    pair_variant_twin_ordinal,
    pure_orbital_camera_frame,
    scene_metadata,
    scene_signature,
    scene_specification,
)
from world_model.utils.config import OrpheusConfig, _strict_construct
from world_model.utils.version import SIMULATOR_VERSION, SPECIFICATION_VERSION, __version__

Split = Literal["development", "selector", "confirmation", "final_test"]
Reduction = Literal["sum", "min", "max", "mean", "rmse"]
_NATIVE_PATH_TYPE = type(Path())

ARCHITECTURE_VERSION = 2
ARCHITECTURE_ATTEMPT = 2
MAX_ARCHITECTURE_ATTEMPTS = 2
OPTIMIZER_UPDATES = 0
BATCH_SIZE = 4
ORDINALS = tuple(range(SCENES_PER_SPLIT))
if SCENES_PER_SPLIT != 64 or len(ORDINALS) != 64:
    raise RuntimeError("variable-radius qualification requires exactly 64 ordinals per split")

HISTORY_FRAME_INDICES = tuple(range(HISTORY_FRAME_COUNT))
ANCHOR_FRAME_INDEX = HISTORY_FRAME_COUNT - 1
HORIZONS_SECONDS = (0.1, 0.25, 0.5, 1.0, 2.0)
TARGET_FRAME_INDICES = tuple(
    ANCHOR_FRAME_INDEX + round(horizon * FRAME_RATE_HZ) for horizon in HORIZONS_SECONDS
)
OBJECT_INDICES = (0, 1)
AXIS_NAMES = ("x", "y", "z")
VJP_STAGES = ("raw", "deployed")
VJP_MODALITIES = ("rgb", "depth", "intrinsics")
VJP_AUDIT_ORDINAL_GROUPS = (
    (0, 12, 17, 29),
    (34, 46, 51, 63),
)
VJP_AUDIT_ORDINALS = tuple(ordinal for group in VJP_AUDIT_ORDINAL_GROUPS for ordinal in group)
if (
    len(VJP_AUDIT_ORDINAL_GROUPS) != 2
    or any(len(group) != BATCH_SIZE for group in VJP_AUDIT_ORDINAL_GROUPS)
    or len(set(VJP_AUDIT_ORDINALS)) != 8
    or {ordinal // 32 for ordinal in VJP_AUDIT_ORDINALS} != set(range(PRIMITIVES_PER_SPLIT))
    or {(ordinal % 32) // 16 for ordinal in VJP_AUDIT_ORDINALS}
    != set(range(PAIR_VARIANTS_PER_PRIMITIVE))
    or {(ordinal % 16) // 8 for ordinal in VJP_AUDIT_ORDINALS}
    != set(range(RADIUS_ROLES_PER_PRIMITIVE))
    or {ordinal % 8 for ordinal in VJP_AUDIT_ORDINALS} != set(range(CAMERA_STRATA))
):
    raise RuntimeError("VJP audit ordinals must cover every governed scene axis")
RUNTIME_STREAM_KEY = "rgbd:camera0:rgbd"
EXPECTED_RADIUS_LOG_VARIANCE = math.log(1.0e-5)
MAX_CHECKPOINT_BYTES = 4 * 1024 * 1024

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUN_RELATIVE_PATH = Path("runs/rgbd_two_visible_variable_radius_v2")
DEVELOPMENT_REPORT_NAME = "development_report_v2.json"
CHECKPOINT_NAME = "development_model_v2.pt"
DEVELOPMENT_LEDGER_NAME = f"development_attempt_{ARCHITECTURE_ATTEMPT}_access.json"
QUALIFICATION_REPORT_NAME = "qualification_report_v2.json"
QUALIFICATION_LEDGER_NAME = f"qualification_attempt_{ARCHITECTURE_ATTEMPT}_access.json"
DEVELOPMENT_ARTIFACT_NAMES = frozenset(
    {DEVELOPMENT_REPORT_NAME, CHECKPOINT_NAME, DEVELOPMENT_LEDGER_NAME}
)
QUALIFICATION_ARTIFACT_NAMES = frozenset(
    {*DEVELOPMENT_ARTIFACT_NAMES, QUALIFICATION_REPORT_NAME, QUALIFICATION_LEDGER_NAME}
)
if len(QUALIFICATION_ARTIFACT_NAMES) != 5:
    raise RuntimeError("variable-radius qualification owns exactly five artifacts")

_PRIOR_ARCHITECTURE_ATTEMPT_DISCLOSURE = {
    "schema": "rgbd_variable_radius_prior_attempt_disclosure_v1",
    "architecture_version": 1,
    "architecture_attempt": 1,
    "protocol_name": "rgbd_two_visible_variable_radius_v1",
    "commit": "db669b099f4e51c18e24645ddee8c1249f86b175",
    "development_report": {
        "sha256": "7f194a41bd5e64328f0a57d8142aad8a81f01d2b449386bb05939fb3ed49b142",
        "bytes": 66758,
    },
    "development_ledger": {
        "sha256": "aec6c9500d3cd8ca6a152b8107578b2b441a544dca605fe7f6ae59a61f0d021e",
        "bytes": 10248,
    },
    "terminal_status": "terminal_error",
    "error": {
        "type": "RuntimeError",
        "message": "element 0 of tensors does not require grad and does not have a grad_fn",
    },
    "active_split": "development",
    "active_batch": [0, 1, 2, 3],
    "checkpoint_published": False,
    "protected_access_started": False,
    "protected_splits_opened": [],
    "retry_permitted": False,
}

FROZEN_CONFIG_SHA256 = "e27934fa16940c82f7bfdfb40d529fe70b6e7eddd82cbffa9cef1ff14d46eb46"
FROZEN_SOURCE_SHA256 = {
    "checkpoint_roundtrip_test": "810e75c72df98230861553f126eb90c485f3318bc7a30aeff9926b1dfe9a3f45",
    "rgbd_two_object_bridge_test": "dceb2517a8f5ec99cc0d8a67c3c0dee972e3c8f30fcd5f889c91c23f37b31d7d",
    "belief_invariants_test": "23f125f8a9fa5c3b9cd74c3c4ab30d4c82fe55151f9160d5b40e2f5d83e6fd91",
    "config_test": "4680c8c3ffff60c4c47ad88b08f8e68a4ba976396ba1eaf41dfa73fad9d4378a",
    "filter_update_test": "31a5ec0e155bfcd68b68e65727a1ee90095eff4f1ddd69ae839c78198f9253ba",
    "rgbd_observation_test": "c01d2e4d7fefdc5d5fbf0929ff58b64f44cd6fc03e9b3f7c48e82b3c3151dfce",
    "two_disc_geometry_test": "79010b2a330b5a85808130856991a0508ce19259ad13746e58d2b580f8948c28",
    "scene": "d9d4f7b9cbb22de2d2cb07db4c9e2b77aa4d57798f996c5c445fa5202f256525",
    "scene_test": "c84d98b9a68acca49d47e2fa9e4d1a3631515deb5f2197f52318d83186aaa10c",
    "profile": "e27934fa16940c82f7bfdfb40d529fe70b6e7eddd82cbffa9cef1ff14d46eb46",
    "profile_test": "9262c18ff825af098d41fd1d3c4b33f222fedf1b58a58c910ff18d0c9e87d6e0",
    "accepted_qualification_provenance": "eda45aa855359b15cfa1372eb1187fe4c827c2d0dc64adae553c6dee5b94930a",
    "two_disc_geometry": "b2b8e21de941213820fa0bf2a076469d1238a41cb9e395ce9fe907816a976220",
    "rgbd_observation": "e0310d9e76d3e7ac390499384143403d66c7ad81e63ac0e2c2ee4002fcb9b7f2",
    "measurements": "c1c696977d3c6eee2ae3eb57d9ee6d1f509c9186f4fc908fba02041d19402bed",
    "filter_correction": "54908b21e68533bb8076ea86a81d7e7fdaa83157daf6f7ec08e8547db1668e9a",
    "filter_uncertainty": "7e2ac7e09fbc969769aa4e58c2a774dcd5ad91a584e2c8e7dbf23265f3831e0b",
    "lifecycle": "1450c7bd0bb3aebc6bf2b5e9678708073e432b17471c8fd66e3e7c42c7c8ed58",
    "online_world_model": "3a49d174b91d6e2158f774974a43943670ec8dece11494aab9ebd80234b88b9a",
    "checkpointing": "91a5eb646b95d433fb0ea08400105ed8595c588e75b18cbd18458d3194781e7b",
    "config": "0ebbb6fb9dbd355c59d577762c9394b02a0c46f47a430c0027bfe51345fe0f7f",
}
FROZEN_SOURCE_PATHS = {
    "checkpoint_roundtrip_test": "tests/integration/test_checkpoint_roundtrip.py",
    "rgbd_two_object_bridge_test": "tests/integration/test_rgbd_two_object_bridge.py",
    "belief_invariants_test": "tests/unit/test_belief_invariants.py",
    "config_test": "tests/unit/test_config.py",
    "filter_update_test": "tests/unit/test_filter_update.py",
    "rgbd_observation_test": "tests/unit/test_rgbd_observation_module.py",
    "two_disc_geometry_test": "tests/unit/test_two_disc_rgbd_geometry.py",
    "scene": "world_model/training/rgbd_variable_radius_scene.py",
    "scene_test": "tests/unit/test_rgbd_variable_radius_scene.py",
    "profile": "configs/rgbd_variable_radius_cpu.yaml",
    "profile_test": "tests/unit/test_rgbd_variable_radius_config.py",
    "accepted_qualification_provenance": (
        "world_model/training/rgbd_online_bridge_qualification.py"
    ),
    "two_disc_geometry": "world_model/observations/rgbd/two_disc_geometry.py",
    "rgbd_observation": "world_model/observations/rgbd/module.py",
    "measurements": "world_model/observations/measurements.py",
    "filter_correction": "world_model/filtering/correction.py",
    "filter_uncertainty": "world_model/filtering/uncertainty.py",
    "lifecycle": "world_model/belief/lifecycle.py",
    "online_world_model": "world_model/runtime/online_world_model.py",
    "checkpointing": "world_model/training/checkpointing.py",
    "config": "world_model/utils/config.py",
}
PUBLICATION_SURFACE_PATHS = {
    "qualification": "world_model/training/rgbd_variable_radius_qualification.py",
    "runner": "scripts/run_rgbd_variable_radius_qualification.py",
    "qualification_test": "tests/unit/test_rgbd_variable_radius_qualification.py",
}

FROZEN_SCENE_DESCRIPTOR_SHA256 = "5145cc7c9f09c4189afe7ddd4da147d93c63e64683f2eaff347b488090a55532"
FROZEN_SCENE_INPUT_BINDING_SHA256 = (
    "4863b2cb58dccc5d2338f27f4afe8b12456a424266a5c89f3ef2da9b1a1cc51d"
)

_FROZEN_SCENE_CERTIFICATE_BINDING: dict[str, Any] = {
    "artifact_kind": "rgbd_variable_radius_scene_family_offline_source_freeze",
    "runtime_recomputation_permitted": False,
    "certificate_sha256": "473137981e0a6443834c806f9f8792e2fee6a556961e5d977d3c6ae69cc7f0d5",
    "descriptor_sha256": "5145cc7c9f09c4189afe7ddd4da147d93c63e64683f2eaff347b488090a55532",
    "input_binding_sha256": "4863b2cb58dccc5d2338f27f4afe8b12456a424266a5c89f3ef2da9b1a1cc51d",
    "scenes_per_split": 64,
    "splits": ["development", "selector", "confirmation", "final_test"],
    "trace_sha256": {
        "metadata": "c7439be3d453fee83b28615cdf338f750b2700f56b1ea4d165790089661acbc1",
        "balance": "ddc36a238791e567012c06953c79999944e1290ad1d7750a6776f060f8e5088f",
        "kinematic": "7b3abf198b12825c9ff548dfe747dde1d9402f4bf7b1e573e2468004a17df1fa",
        "radius_labelled_physical": "4b812318837751db3d97d3b935c20027bb4202147d80b8cbeab3d06c9a4fe960",
        "camera": "aad97eb3b84b35f1016850b80ff685c456458796a420902ec5457002b66d2b76",
        "raster": "b7990b5d424fdcf2373d9f6a9adca3d6a0182a1d39758d2655f5b650625e3e0d",
        "conic_geometry": "f5239000cf5407af72559c55f4817fa9948d57199a8e30871d57435cb0e3ff6a",
        "fit_observability": "a352b68362a502428c4e6d73f2a5f5d806ee63f80dcfcb48685df5aa841353d0",
        "combined": "a2d73c20f6d5167d35bee3b790ae6959e4dcb51889acd7f4323b027607529c79",
        "expected_lifecycle": "3cf7559ecd1606a03b7c21f0596b14f8ab690e6ec658875040ff7e5d9859b38b",
    },
    "split_trace_sha256": {
        "kinematic": {
            "development": "809e23de612aa17e1f270fc983fec4c56d6196c3cb78ec1fb7f9328a19fa0771",
            "selector": "9205618a6169b596dac863888b2375c50615f79221006acfbe47369e923e31fb",
            "confirmation": "566eace7afd803cfc06ffc7390d7989b094bb5bde43c5c8ee9d84b8b2077029b",
            "final_test": "8bbfb3eb8ddfb35be1b0f1685499179d1cde535ad9c925ceaf4b5d3a4ab31126",
        },
        "radius_labelled_physical": {
            "development": "cbde1225854f9f7b37413f927c9d3af2afb11f1e26cbe02f387c8c9994a37c54",
            "selector": "a3f84a3a5491aa0cd9fe02cf561ca6f9f872b4bc8a3a9a8e4df40becf7bdef31",
            "confirmation": "3e8800d283c5ffe3d42a8166547704dfffa38738c284e6809e476fc752944731",
            "final_test": "d6e7315fa7bf49347be29927b79b2082b80e911e4f73a8dcfd9f07224e84fa0c",
        },
        "raster": {
            "development": "2f165c0c0759fbd07da8e5bd6e88d94d12700bd59a9413a21e26ef9327b8be89",
            "selector": "7238e7d9d7f1b80758438563a1bf24e3ee6a20a75ab7eb6e37d3cfc65d9e1da5",
            "confirmation": "e9f257c62448c793f6099c727a1d2f17ac41b01e3e76bb669861e075c5bc147a",
            "final_test": "89ad46344fe8770d79ab0ebba851ea063078ff9f915f369b8386537a07c78df5",
        },
        "conic_geometry": {
            "development": "dd91ad6696032a6d5f43952fc5eaedbf4591ec91b9f2b23480be1479b668beed",
            "selector": "b5dfbdc9ffbcaae62f807b627ee19262e56786e1b7784b1e6ec2077713f7909e",
            "confirmation": "3836edccbb9b653da26d645e2938bf475db6db10c9216387c75f93e78f444b1b",
            "final_test": "139c605d0bbbb6507fcb81e955e4d94e1ad6fc95f55c2e35b94b22ca893a8107",
        },
        "fit_observability": {
            "development": "a2a3f3ecf7bf487fe2b6de7cd338569e6911729978c4772f6452ae03ad72d39d",
            "selector": "de735397a35a0dabc447082fdb1c7556413c61fe81210549b94044787ddf6e3b",
            "confirmation": "f69851687b3ec0a4447eef08220ba5fea71a24c641b1a96167ad5b3f417b6b44",
            "final_test": "4fa71919b84c3b9bc8c6e2e3cbd1235d07ed77f8baf369c487887f4e0954c50f",
        },
        "combined": {
            "development": "d891e50f0e1e9962c34508e96c0294248f43cd8a6ab14ea4cefae0010b4bbee6",
            "selector": "14e7c852fa3a56abf3d130aea4b6793549c29b3cf9aff7ae64395744d4fc68eb",
            "confirmation": "822ed7b4ff3efe05906c1bf4b1928253c1ebdb5d152cab84dee466789c36170f",
            "final_test": "adbc42f44133eb3826ce644f9c08a2a007257a8c26e791376042bbe2b6c4e12c",
        },
    },
}


def _json_native(
    value: Any,
    *,
    label: str = "JSON value",
    _active_containers: set[int] | None = None,
) -> Any:
    """Return an exact recursively JSON-native copy.

    Tuples are deliberately normalized to lists before any durable hash or
    validation. Other Python containers and scalar subclasses are rejected so
    an in-memory protocol cannot acquire a representation that changes after a
    JSON round trip.
    """

    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{label} must contain only finite floats")
        return value
    if type(value) not in {dict, list, tuple}:
        raise TypeError(f"{label} contains non-JSON-native type {type(value).__name__}")
    active = set() if _active_containers is None else _active_containers
    identity = id(value)
    if identity in active:
        raise ValueError(f"{label} contains a recursive container")
    active.add(identity)
    try:
        if type(value) is dict:
            if any(type(key) is not str for key in value):
                raise TypeError(f"{label} object keys must be exact strings")
            return {
                key: _json_native(
                    item,
                    label=f"{label}.{key}",
                    _active_containers=active,
                )
                for key, item in value.items()
            }
        return [
            _json_native(
                item,
                label=f"{label}[{index}]",
                _active_containers=active,
            )
            for index, item in enumerate(value)
        ]
    finally:
        active.remove(identity)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _json_native(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def sha256_bytes(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def validated_sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise ValueError(f"{label} must be an exact SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} must be hexadecimal") from error
    return value


def _prior_architecture_attempt_disclosure() -> dict[str, Any]:
    disclosure = _json_native(
        copy.deepcopy(_PRIOR_ARCHITECTURE_ATTEMPT_DISCLOSURE),
        label="prior architecture attempt disclosure",
    )
    if type(disclosure) is not dict:
        raise RuntimeError("prior architecture attempt disclosure must be an exact dict")
    expected_keys = {
        "schema",
        "architecture_version",
        "architecture_attempt",
        "protocol_name",
        "commit",
        "development_report",
        "development_ledger",
        "terminal_status",
        "error",
        "active_split",
        "active_batch",
        "checkpoint_published",
        "protected_access_started",
        "protected_splits_opened",
        "retry_permitted",
    }
    if set(disclosure) != expected_keys:
        raise RuntimeError("prior architecture attempt disclosure schema differs")
    expected_scalars = {
        "schema": "rgbd_variable_radius_prior_attempt_disclosure_v1",
        "architecture_version": 1,
        "architecture_attempt": 1,
        "protocol_name": "rgbd_two_visible_variable_radius_v1",
        "terminal_status": "terminal_error",
        "active_split": "development",
        "checkpoint_published": False,
        "protected_access_started": False,
        "retry_permitted": False,
    }
    for name, expected in expected_scalars.items():
        _exact_equal(disclosure[name], expected, label=f"prior attempt.{name}")
    commit = disclosure["commit"]
    if (
        type(commit) is not str
        or commit != "db669b099f4e51c18e24645ddee8c1249f86b175"
        or len(commit) != 40
    ):
        raise ValueError("prior attempt commit must be one exact SHA-1")
    try:
        int(commit, 16)
    except ValueError as error:
        raise ValueError("prior attempt commit must be hexadecimal") from error
    for name, expected_sha256, expected_bytes in (
        (
            "development_report",
            "7f194a41bd5e64328f0a57d8142aad8a81f01d2b449386bb05939fb3ed49b142",
            66758,
        ),
        (
            "development_ledger",
            "aec6c9500d3cd8ca6a152b8107578b2b441a544dca605fe7f6ae59a61f0d021e",
            10248,
        ),
    ):
        artifact = disclosure[name]
        if type(artifact) is not dict or set(artifact) != {"sha256", "bytes"}:
            raise ValueError(f"prior attempt {name} binding schema differs")
        if validated_sha256(artifact["sha256"], label=f"prior attempt {name}") != expected_sha256:
            raise ValueError(f"prior attempt {name} digest differs")
        if type(artifact["bytes"]) is not int or artifact["bytes"] != expected_bytes:
            raise ValueError(f"prior attempt {name} byte count differs")
    _exact_equal(
        disclosure["error"],
        {
            "type": "RuntimeError",
            "message": "element 0 of tensors does not require grad and does not have a grad_fn",
        },
        label="prior attempt error",
    )
    _exact_equal(disclosure["active_batch"], [0, 1, 2, 3], label="prior attempt batch")
    _exact_equal(
        disclosure["protected_splits_opened"],
        [],
        label="prior attempt protected split inventory",
    )
    return disclosure


def _frozen_scene_certificate_binding() -> dict[str, Any]:
    result = copy.deepcopy(_FROZEN_SCENE_CERTIFICATE_BINDING)
    if (
        type(result) is not dict
        or result["artifact_kind"] != "rgbd_variable_radius_scene_family_offline_source_freeze"
        or result["runtime_recomputation_permitted"] is not False
        or result["certificate_sha256"] != FROZEN_CERTIFICATE_SHA256
        or result["descriptor_sha256"] != FROZEN_SCENE_DESCRIPTOR_SHA256
        or result["input_binding_sha256"] != FROZEN_SCENE_INPUT_BINDING_SHA256
        or result["scenes_per_split"] != SCENES_PER_SPLIT
        or result["splits"] != list(SPLITS)
        or result["trace_sha256"] != FROZEN_TRACE_SHA256
        or result["split_trace_sha256"] != FROZEN_SPLIT_TRACE_SHA256
    ):
        raise RuntimeError(
            "literal variable-radius certificate binding differs from source constants"
        )
    validated_sha256(result["descriptor_sha256"], label="scene descriptor")
    validated_sha256(result["input_binding_sha256"], label="scene input binding")
    return result


def _manifest_rows(split: str) -> tuple[dict[str, Any], ...]:
    if type(split) is not str or split not in SPLITS:
        raise ValueError(f"unknown variable-radius split {split!r}")
    return tuple({"split": split, "ordinal": ordinal} for ordinal in ORDINALS)


MANIFEST_SHA256 = {split: canonical_sha256(list(_manifest_rows(split))) for split in SPLITS}


def _validate_manifest_rows(split: str, rows: Any) -> list[dict[str, Any]]:
    if type(rows) is not list or len(rows) != SCENES_PER_SPLIT:
        raise ValueError(f"{split} manifest must be an exact 64-row list")
    for ordinal, row in enumerate(rows):
        if type(row) is not dict or set(row) != {"split", "ordinal"}:
            raise ValueError(f"{split} manifest row {ordinal} has the wrong exact schema")
        if type(row["split"]) is not str or row["split"] != split:
            raise ValueError(f"{split} manifest row {ordinal} has the wrong split")
        if type(row["ordinal"]) is not int or row["ordinal"] != ordinal:
            raise ValueError(f"{split} manifest row {ordinal} has the wrong exact ordinal")
    if canonical_sha256(rows) != MANIFEST_SHA256[split]:
        raise ValueError(f"{split} manifest row hash differs")
    return rows


@dataclass(frozen=True, slots=True)
class VariableRadiusGates:
    """Predeclared behavioural, ownership, topology, and resource limits."""

    anchor_radius_rmse_m: float = 5.0e-4
    history_radius_rmse_m: float = 5.0e-4
    radius_max_abs_error_m: float = 5.0e-4
    grouped_radius_rmse_m: float = 5.0e-4
    within_track_radius_span_max_m: float = 5.0e-4
    paired_radius_delta_rmse_m: float = 5.0e-4
    paired_radius_delta_max_abs_error_m: float = 5.0e-4
    paired_radius_delta_sign_fraction: float = 1.0
    pair_variant_paired_radius_delta_rmse_m: float = 5.0e-4
    pair_variant_paired_radius_delta_max_abs_error_m: float = 5.0e-4
    pair_variant_paired_radius_delta_sign_fraction: float = 1.0
    pair_variant_estimated_anchor_position_rmse_m: float = 1.0e-3
    pair_variant_estimated_anchor_position_max_abs_m: float = 1.0e-3
    pair_variant_estimated_anchor_velocity_rmse_mps: float = 1.0e-3
    pair_variant_estimated_anchor_velocity_max_abs_mps: float = 1.0e-3
    surface_fit_relative_residual_max: float = 0.05
    surface_fit_condition_max: float = 100.0

    current_position_rmse_m: float = 0.010
    current_velocity_rmse_mps: float = 0.012
    horizon_position_rmse_m: tuple[float, ...] = (0.011, 0.013, 0.016, 0.022, 0.035)
    horizon_velocity_rmse_mps: float = 0.012
    per_object_axis_position_rmse_m: float = 0.014
    per_object_axis_velocity_rmse_mps: float = 0.016

    minimum_rgb_radius_gradient_l1: float = 1.0e-14
    minimum_depth_intrinsics_radius_gradient_l1: float = 1.0e-8
    maximum_input_gradient_l1: float = 1.0e8
    maximum_gradient_difference: float = 1.0e-5
    perception_latency_seconds: float = 3.5
    state_only_rollout_latency_seconds: float = 0.075
    vjp_latency_seconds: float = 30.0
    persistent_runtime_tensor_state_bytes: float = 65_536.0
    process_max_rss_bytes: float = 2_500_000_000.0
    process_rss_delta_bytes: float = 1_000_000_000.0


DEFAULT_GATES = VariableRadiusGates()


@dataclass(frozen=True, slots=True)
class ReducerSpec:
    """One closed scalar reduction over validated per-scene evidence."""

    output: str
    source: str
    reduction: Reduction


REDUCER_REGISTRY = (
    ReducerSpec("anchor_radius_rmse_m", "anchor_radius_error", "rmse"),
    ReducerSpec("history_radius_rmse_m", "history_radius_error", "rmse"),
    ReducerSpec("radius_max_abs_error_m", "all_radius_abs_error", "max"),
    ReducerSpec("radius_valid_fraction", "radius_valid", "mean"),
    ReducerSpec("radius_in_bounds_fraction", "radius_in_bounds", "mean"),
    ReducerSpec(
        "surface_fit_relative_residual_max",
        "surface_fit_relative_residual",
        "max",
    ),
    ReducerSpec("surface_fit_condition_max", "surface_fit_condition", "max"),
    ReducerSpec(
        "within_track_radius_span_max_m",
        "within_track_radius_span",
        "max",
    ),
    ReducerSpec("current_position_rmse_m", "current_position_error", "rmse"),
    ReducerSpec("current_velocity_rmse_mps", "current_velocity_error", "rmse"),
    ReducerSpec("identity_switch_count", "identity_switch_count", "sum"),
    ReducerSpec(
        "persistent_id_mismatch_count",
        "persistent_id_mismatch_count",
        "sum",
    ),
    ReducerSpec(
        "association_ambiguous_pair_count",
        "association_ambiguous_pair_count",
        "sum",
    ),
    ReducerSpec(
        "direct_radius_owner_max_abs_m",
        "direct_radius_owner_error",
        "max",
    ),
    ReducerSpec(
        "direct_radius_log_variance_owner_max_abs",
        "direct_radius_log_variance_owner_error",
        "max",
    ),
    ReducerSpec(
        "configured_radius_log_variance_max_abs_error",
        "configured_radius_log_variance_error",
        "max",
    ),
    ReducerSpec(
        "emitted_radius_log_variance_pairwise_max_abs_error",
        "emitted_radius_log_variance_pairwise_error",
        "max",
    ),
    ReducerSpec(
        "stored_radius_log_variance_pairwise_max_abs_error",
        "stored_radius_log_variance_pairwise_error",
        "max",
    ),
    ReducerSpec("active_fraction", "active", "mean"),
    ReducerSpec("rollout_active_fraction", "rollout_active", "mean"),
    ReducerSpec(
        "public_rollout_output_alias_count",
        "public_rollout_output_alias_count",
        "sum",
    ),
)


def _radius_group_metric_names() -> tuple[str, ...]:
    names: list[str] = []
    for object_index in OBJECT_INDICES:
        names.append(f"anchor_radius_rmse_m/object_{object_index}")
    for primitive_index in range(PRIMITIVES_PER_SPLIT):
        names.append(f"anchor_radius_rmse_m/primitive_{primitive_index}")
    for pair_variant in range(PAIR_VARIANTS_PER_PRIMITIVE):
        names.append(f"anchor_radius_rmse_m/pair_variant_{pair_variant}")
    for role in range(RADIUS_ROLES_PER_PRIMITIVE):
        names.append(f"anchor_radius_rmse_m/role_{role}")
    for stratum in range(CAMERA_STRATA):
        names.append(f"anchor_radius_rmse_m/camera_stratum_{stratum}")
    return tuple(names)


def _position_velocity_metric_names() -> tuple[str, ...]:
    names: list[str] = []
    for object_index in OBJECT_INDICES:
        for axis in AXIS_NAMES:
            names.extend(
                (
                    f"current_position_rmse_m/object_{object_index}/{axis}",
                    f"current_velocity_rmse_mps/object_{object_index}/{axis}",
                )
            )
    for horizon in HORIZONS_SECONDS:
        label = f"{horizon:.2f}"
        names.extend(
            (
                f"horizon_{label}_position_rmse_m",
                f"horizon_{label}_velocity_rmse_mps",
            )
        )
        for object_index in OBJECT_INDICES:
            for axis in AXIS_NAMES:
                names.extend(
                    (
                        f"horizon_{label}_position_rmse_m/object_{object_index}/{axis}",
                        f"horizon_{label}_velocity_rmse_mps/object_{object_index}/{axis}",
                    )
                )
    return tuple(names)


def _vjp_metric_names() -> tuple[str, ...]:
    names: list[str] = []
    for stage in VJP_STAGES:
        for object_index in OBJECT_INDICES:
            for modality in VJP_MODALITIES:
                prefix = f"{stage}/object_{object_index}/{modality}"
                names.extend(
                    (
                        f"gradient_l1/{prefix}",
                        f"gradient_max_l1/{prefix}",
                        f"gradient_anchor_history_frame_l1/{prefix}",
                        f"gradient_nonanchor_max_history_frame_l1/{prefix}",
                        f"gradient_supported_history_frames/{prefix}",
                        f"gradient_cross_scene_max_l1/{prefix}",
                    )
                )
            names.append(f"world_from_camera_gradient_l1/{stage}/object_{object_index}")
    for object_index in OBJECT_INDICES:
        for modality in VJP_MODALITIES:
            prefix = f"object_{object_index}/{modality}"
            names.extend(
                (
                    f"gradient_raw_deployed_max_abs_difference/{prefix}",
                    f"gradient_raw_deployed_support_mismatch_count/{prefix}",
                )
            )
    names.extend(
        (
            "gradient_vector_count",
            "gradient_audit_scene_count",
            "gradient_audit_unique_scene_fraction",
            "gradient_audit_primitive_coverage_fraction",
            "gradient_audit_pair_variant_coverage_fraction",
            "gradient_audit_radius_role_coverage_fraction",
            "gradient_audit_camera_stratum_coverage_fraction",
            "vjp_latency_seconds",
        )
    )
    return tuple(names)


_NONREDUCED_METRIC_NAMES = (
    "scene_count",
    "object_count",
    "identity_coverage",
    "persistent_object_id_min",
    "persistent_object_id_max",
    "association_pair_coverage",
    "history_sample_count_min",
    "history_sample_count_max",
    "history_valid_count_min",
    "history_valid_count_max",
    "history_span_max_abs_error_seconds",
    "radius_owner_count_min",
    "radius_owner_count_max",
    "larger_radius_slot_zero_fraction",
    "larger_radius_slot_one_fraction",
    "counterfactual_pair_count",
    "paired_radius_delta_rmse_m",
    "paired_radius_delta_max_abs_error_m",
    "paired_radius_delta_sign_fraction",
    "counterfactual_non_radius_certificate_mismatch_count",
    "counterfactual_unintended_truth_position_max_abs_m",
    "counterfactual_unintended_truth_velocity_max_abs_mps",
    "pair_variant_counterfactual_pair_count",
    "pair_variant_paired_radius_delta_rmse_m",
    "pair_variant_paired_radius_delta_max_abs_error_m",
    "pair_variant_paired_radius_delta_sign_fraction",
    "pair_variant_non_radius_certificate_mismatch_count",
    "pair_variant_unordered_radius_pair_match_count",
    "pair_variant_unintended_truth_position_max_abs_m",
    "pair_variant_unintended_truth_velocity_max_abs_mps",
    "pair_variant_estimated_anchor_position_rmse_m",
    "pair_variant_estimated_anchor_position_max_abs_m",
    "pair_variant_estimated_anchor_velocity_rmse_mps",
    "pair_variant_estimated_anchor_velocity_max_abs_mps",
    "prior_raw_radius_bit_mismatch_count",
    "prior_deployed_radius_bit_mismatch_count",
    "prior_complete_state_bit_mismatch_count",
    "prior_object_id_mismatch_count",
    "prior_history_bit_mismatch_count",
    "prior_prediction_bit_mismatch_count",
    "alternate_prior_count",
    "legacy_fixed_radius_max_abs_error_m",
    "legacy_radius_variance_write_count",
    "legacy_supported_radius_field_count",
    "missing_depth_radius_valid_fraction",
    "missing_depth_radius_write_count",
    "no_foreground_radius_valid_fraction",
    "no_foreground_radius_write_count",
    "ambiguous_radius_write_count",
    "malformed_radius_group_rejection_count",
    "ingested_frame_count_min",
    "ingested_frame_count_max",
    "state_ingest_count_min",
    "state_ingest_count_max",
    "public_predict_calls_per_batch_min",
    "public_predict_calls_per_batch_max",
    "model_reset_count_per_batch_min",
    "model_reset_count_per_batch_max",
    "packet_factory_call_count",
    "simulator_constructor_call_count",
    "public_renderer_call_count",
    "formal_certificate_recomputation_count",
    "optimizer_updates",
    "optimizer_state_entry_count",
    "rng_state_entry_count",
    "learned_parameter_count",
    "learned_parameter_bytes",
    "module_tensor_buffer_count",
    "persistent_module_state_key_count",
    "persistent_module_state_bytes",
    "persistent_runtime_tensor_state_bytes_max",
    "process_max_rss_bytes",
    "process_rss_delta_bytes",
    "perception_latency_seconds",
    "state_only_rollout_latency_seconds",
    "source_scene_count",
    "source_split_count",
    "source_counterfactual_pairs",
    "unique_scene_specification_fraction",
)


_GATE_SCHEMA_OWNERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("reducers", tuple(spec.output for spec in REDUCER_REGISTRY)),
    ("nonreduced", _NONREDUCED_METRIC_NAMES),
    ("radius_groups", _radius_group_metric_names()),
    ("position_velocity_groups", _position_velocity_metric_names()),
    ("vjp", _vjp_metric_names()),
)
_gate_names_in_owner_order = tuple(name for _, names in _GATE_SCHEMA_OWNERS for name in names)
_gate_duplicate_names = tuple(
    sorted(
        name
        for name in set(_gate_names_in_owner_order)
        if _gate_names_in_owner_order.count(name) != 1
    )
)
if _gate_duplicate_names:
    raise RuntimeError(f"variable-radius gate ownership collides: {_gate_duplicate_names!r}")
GATE_METRIC_SCHEMA = tuple(sorted(_gate_names_in_owner_order))


def _gate_surface(
    metrics: Mapping[str, Any],
    *,
    schema_only: bool,
) -> tuple[list[str], set[str]]:
    gates = DEFAULT_GATES
    failures: list[str] = []
    required = set(GATE_METRIC_SCHEMA)
    if schema_only:
        return failures, required

    def scalar(key: str) -> float | None:
        value = metrics.get(key)
        if type(value) is not float or not math.isfinite(value):
            failures.append(f"{key}:missing_nonfinite_or_nonfloat")
            return None
        return value

    def maximum(key: str, limit: float) -> None:
        value = scalar(key)
        if value is not None and value > limit:
            failures.append(f"{key}:{value:.9g}>{limit:.9g}")

    def minimum(key: str, limit: float) -> None:
        value = scalar(key)
        if value is not None and value < limit:
            failures.append(f"{key}:{value:.9g}<{limit:.9g}")

    def equal(key: str, expected: float) -> None:
        value = scalar(key)
        if value is not None and value != expected:
            failures.append(f"{key}:{value:.9g}!={expected:.9g}")

    maximum("anchor_radius_rmse_m", gates.anchor_radius_rmse_m)
    maximum("history_radius_rmse_m", gates.history_radius_rmse_m)
    maximum("radius_max_abs_error_m", gates.radius_max_abs_error_m)
    maximum("within_track_radius_span_max_m", gates.within_track_radius_span_max_m)
    maximum(
        "surface_fit_relative_residual_max",
        gates.surface_fit_relative_residual_max,
    )
    maximum("surface_fit_condition_max", gates.surface_fit_condition_max)
    for key in _radius_group_metric_names():
        maximum(key, gates.grouped_radius_rmse_m)
    equal("radius_valid_fraction", 1.0)
    equal("radius_in_bounds_fraction", 1.0)

    maximum("paired_radius_delta_rmse_m", gates.paired_radius_delta_rmse_m)
    maximum(
        "paired_radius_delta_max_abs_error_m",
        gates.paired_radius_delta_max_abs_error_m,
    )
    minimum(
        "paired_radius_delta_sign_fraction",
        gates.paired_radius_delta_sign_fraction,
    )
    equal("counterfactual_pair_count", 32.0)
    equal("counterfactual_non_radius_certificate_mismatch_count", 0.0)
    equal("counterfactual_unintended_truth_position_max_abs_m", 0.0)
    equal("counterfactual_unintended_truth_velocity_max_abs_mps", 0.0)
    maximum(
        "pair_variant_paired_radius_delta_rmse_m",
        gates.pair_variant_paired_radius_delta_rmse_m,
    )
    maximum(
        "pair_variant_paired_radius_delta_max_abs_error_m",
        gates.pair_variant_paired_radius_delta_max_abs_error_m,
    )
    minimum(
        "pair_variant_paired_radius_delta_sign_fraction",
        gates.pair_variant_paired_radius_delta_sign_fraction,
    )
    equal("pair_variant_counterfactual_pair_count", 32.0)
    equal("pair_variant_non_radius_certificate_mismatch_count", 0.0)
    equal("pair_variant_unordered_radius_pair_match_count", 0.0)
    equal("pair_variant_unintended_truth_position_max_abs_m", 0.0)
    equal("pair_variant_unintended_truth_velocity_max_abs_mps", 0.0)
    maximum(
        "pair_variant_estimated_anchor_position_rmse_m",
        gates.pair_variant_estimated_anchor_position_rmse_m,
    )
    maximum(
        "pair_variant_estimated_anchor_position_max_abs_m",
        gates.pair_variant_estimated_anchor_position_max_abs_m,
    )
    maximum(
        "pair_variant_estimated_anchor_velocity_rmse_mps",
        gates.pair_variant_estimated_anchor_velocity_rmse_mps,
    )
    maximum(
        "pair_variant_estimated_anchor_velocity_max_abs_mps",
        gates.pair_variant_estimated_anchor_velocity_max_abs_mps,
    )

    maximum("current_position_rmse_m", gates.current_position_rmse_m)
    maximum("current_velocity_rmse_mps", gates.current_velocity_rmse_mps)
    for object_index in OBJECT_INDICES:
        for axis in AXIS_NAMES:
            maximum(
                f"current_position_rmse_m/object_{object_index}/{axis}",
                gates.per_object_axis_position_rmse_m,
            )
            maximum(
                f"current_velocity_rmse_mps/object_{object_index}/{axis}",
                gates.per_object_axis_velocity_rmse_mps,
            )
    for horizon_index, horizon in enumerate(HORIZONS_SECONDS):
        label = f"{horizon:.2f}"
        maximum(
            f"horizon_{label}_position_rmse_m",
            gates.horizon_position_rmse_m[horizon_index],
        )
        maximum(
            f"horizon_{label}_velocity_rmse_mps",
            gates.horizon_velocity_rmse_mps,
        )
        for object_index in OBJECT_INDICES:
            for axis in AXIS_NAMES:
                maximum(
                    f"horizon_{label}_position_rmse_m/object_{object_index}/{axis}",
                    gates.per_object_axis_position_rmse_m
                    + gates.horizon_position_rmse_m[horizon_index],
                )
                maximum(
                    f"horizon_{label}_velocity_rmse_mps/object_{object_index}/{axis}",
                    gates.per_object_axis_velocity_rmse_mps,
                )

    for key, expected in {
        "scene_count": 64.0,
        "object_count": 128.0,
        "identity_switch_count": 0.0,
        "persistent_id_mismatch_count": 0.0,
        "identity_coverage": 1.0,
        "persistent_object_id_min": 0.0,
        "persistent_object_id_max": 1.0,
        "association_pair_coverage": 1.0,
        "association_ambiguous_pair_count": 0.0,
        "history_sample_count_min": 16.0,
        "history_sample_count_max": 16.0,
        "history_valid_count_min": 16.0,
        "history_valid_count_max": 16.0,
        "radius_owner_count_min": 1.0,
        "radius_owner_count_max": 1.0,
        "larger_radius_slot_zero_fraction": 0.5,
        "larger_radius_slot_one_fraction": 0.5,
        "alternate_prior_count": 2.0,
        "prior_raw_radius_bit_mismatch_count": 0.0,
        "prior_deployed_radius_bit_mismatch_count": 0.0,
        "prior_complete_state_bit_mismatch_count": 0.0,
        "prior_object_id_mismatch_count": 0.0,
        "prior_history_bit_mismatch_count": 0.0,
        "prior_prediction_bit_mismatch_count": 0.0,
        "legacy_radius_variance_write_count": 0.0,
        "legacy_supported_radius_field_count": 0.0,
        "missing_depth_radius_valid_fraction": 0.0,
        "missing_depth_radius_write_count": 0.0,
        "no_foreground_radius_valid_fraction": 0.0,
        "no_foreground_radius_write_count": 0.0,
        "ambiguous_radius_write_count": 0.0,
        "malformed_radius_group_rejection_count": 1.0,
        "ingested_frame_count_min": 16.0,
        "ingested_frame_count_max": 16.0,
        "state_ingest_count_min": 16.0,
        "state_ingest_count_max": 16.0,
        "public_predict_calls_per_batch_min": 1.0,
        "public_predict_calls_per_batch_max": 1.0,
        "model_reset_count_per_batch_min": 1.0,
        "model_reset_count_per_batch_max": 1.0,
        "packet_factory_call_count": 0.0,
        "simulator_constructor_call_count": 0.0,
        "public_renderer_call_count": 0.0,
        "formal_certificate_recomputation_count": 0.0,
        "optimizer_updates": 0.0,
        "optimizer_state_entry_count": 0.0,
        "rng_state_entry_count": 0.0,
        "learned_parameter_count": 0.0,
        "learned_parameter_bytes": 0.0,
        "module_tensor_buffer_count": 0.0,
        "persistent_module_state_key_count": 0.0,
        "persistent_module_state_bytes": 0.0,
        "source_scene_count": 256.0,
        "source_split_count": 4.0,
        "source_counterfactual_pairs": 128.0,
        "unique_scene_specification_fraction": 1.0,
    }.items():
        equal(key, expected)

    for key in (
        "direct_radius_owner_max_abs_m",
        "direct_radius_log_variance_owner_max_abs",
        "configured_radius_log_variance_max_abs_error",
        "emitted_radius_log_variance_pairwise_max_abs_error",
        "stored_radius_log_variance_pairwise_max_abs_error",
        "history_span_max_abs_error_seconds",
        "public_rollout_output_alias_count",
    ):
        equal(key, 0.0)
    maximum("legacy_fixed_radius_max_abs_error_m", 1.0e-7)
    equal("active_fraction", 1.0)
    equal("rollout_active_fraction", 1.0)

    for stage in VJP_STAGES:
        for object_index in OBJECT_INDICES:
            for modality in VJP_MODALITIES:
                prefix = f"{stage}/object_{object_index}/{modality}"
                gradient_floor = (
                    gates.minimum_rgb_radius_gradient_l1
                    if modality == "rgb"
                    else gates.minimum_depth_intrinsics_radius_gradient_l1
                )
                minimum(f"gradient_l1/{prefix}", gradient_floor)
                maximum(f"gradient_max_l1/{prefix}", gates.maximum_input_gradient_l1)
                minimum(
                    f"gradient_anchor_history_frame_l1/{prefix}",
                    gradient_floor,
                )
                equal(f"gradient_nonanchor_max_history_frame_l1/{prefix}", 0.0)
                equal(f"gradient_supported_history_frames/{prefix}", 1.0)
                equal(f"gradient_cross_scene_max_l1/{prefix}", 0.0)
            equal(f"world_from_camera_gradient_l1/{stage}/object_{object_index}", 0.0)
    for object_index in OBJECT_INDICES:
        for modality in VJP_MODALITIES:
            prefix = f"object_{object_index}/{modality}"
            maximum(
                f"gradient_raw_deployed_max_abs_difference/{prefix}",
                gates.maximum_gradient_difference,
            )
            equal(f"gradient_raw_deployed_support_mismatch_count/{prefix}", 0.0)
    equal("gradient_vector_count", 12.0)
    equal("gradient_audit_scene_count", 8.0)
    equal("gradient_audit_unique_scene_fraction", 1.0)
    equal("gradient_audit_primitive_coverage_fraction", 1.0)
    equal("gradient_audit_pair_variant_coverage_fraction", 1.0)
    equal("gradient_audit_radius_role_coverage_fraction", 1.0)
    equal("gradient_audit_camera_stratum_coverage_fraction", 1.0)

    maximum("perception_latency_seconds", gates.perception_latency_seconds)
    maximum("state_only_rollout_latency_seconds", gates.state_only_rollout_latency_seconds)
    maximum("vjp_latency_seconds", gates.vjp_latency_seconds)
    maximum(
        "persistent_runtime_tensor_state_bytes_max",
        gates.persistent_runtime_tensor_state_bytes,
    )
    maximum("process_max_rss_bytes", gates.process_max_rss_bytes)
    maximum("process_rss_delta_bytes", gates.process_rss_delta_bytes)
    return failures, required


def gate_failures(metrics: Mapping[str, Any]) -> list[str]:
    if not isinstance(metrics, Mapping):
        return ["metric_schema:not_a_mapping"]
    failures, required = _gate_surface(metrics, schema_only=False)
    actual = set(metrics)
    if actual != required:
        failures.append(
            "metric_schema:"
            f"missing={sorted(required - actual)!r}:extra={sorted(actual - required)!r}"
        )
    return failures


def _exact_equal(actual: Any, expected: Any, *, label: str) -> None:
    if type(actual) is not type(expected):
        raise ValueError(
            f"{label} has type {type(actual).__name__}, expected {type(expected).__name__}"
        )
    if isinstance(expected, tuple):
        if len(actual) != len(expected):
            raise ValueError(f"{label} length differs")
        for index, (item, target) in enumerate(zip(actual, expected, strict=True)):
            _exact_equal(item, target, label=f"{label}[{index}]")
        return
    if isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"{label} length differs")
        for index, (item, target) in enumerate(zip(actual, expected, strict=True)):
            _exact_equal(item, target, label=f"{label}[{index}]")
        return
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError(f"{label} key set differs")
        for key in expected:
            _exact_equal(actual[key], expected[key], label=f"{label}.{key}")
        return
    if actual != expected:
        raise ValueError(f"{label} must equal {expected!r}, got {actual!r}")


def assert_rgbd_variable_radius_config(config: OrpheusConfig) -> None:
    """Reject every semantic profile outside the reviewed qualification."""

    if not isinstance(config, OrpheusConfig):
        raise TypeError("variable-radius qualification requires OrpheusConfig")
    expected = {
        "project.deterministic": (config.project.deterministic, True),
        "device.preference": (config.device.preference, "cpu"),
        "device.closed_loop_preference": (
            config.device.closed_loop_preference,
            "cpu",
        ),
        "device.cuda_amp": (config.device.cuda_amp, False),
        "device.compile": (config.device.compile, False),
        "simulator.image_size": (config.simulator.image_size, (64, 64)),
        "simulator.frame_rate": (config.simulator.frame_rate, 20),
        "simulator.physics_rate": (config.simulator.physics_rate, 120),
        "simulator.sequence_frames": (config.simulator.sequence_frames, 56),
        "simulator.min_objects": (config.simulator.min_objects, 2),
        "simulator.max_objects": (config.simulator.max_objects, 2),
        "simulator.gravity": (config.simulator.gravity, (0.0, 0.0, 0.0)),
        "simulator.radius_range": (config.simulator.radius_range, (0.19, 0.25)),
        "simulator.drag_range": (config.simulator.drag_range, (0.05, 0.05)),
        "simulator.render_noise_std": (config.simulator.render_noise_std, 0.0),
        "simulator.ensure_collision": (config.simulator.ensure_collision, False),
        "simulator.external_impulse_probability": (
            config.simulator.external_impulse_probability,
            0.0,
        ),
        "model.max_objects": (config.model.max_objects, 2),
        "model.state.geometry_dim": (config.model.state.geometry_dim, 1),
        "model.state.appearance_dim": (config.model.state.appearance_dim, 3),
        "model.state.modal_count": (config.model.state.modal_count, 0),
        "model.rgb.enabled": (config.model.rgb.enabled, False),
        "model.rgbd.enabled": (config.model.rgbd.enabled, True),
        "model.rgbd.proposal_count": (config.model.rgbd.proposal_count, 2),
        "model.rgbd.world_radius": (config.model.rgbd.world_radius, 0.21),
        "model.rgbd.metric_radius_estimation_enabled": (
            config.model.rgbd.metric_radius_estimation_enabled,
            True,
        ),
        "model.rgbd.minimum_world_radius": (
            config.model.rgbd.minimum_world_radius,
            0.19,
        ),
        "model.rgbd.maximum_world_radius": (
            config.model.rgbd.maximum_world_radius,
            0.25,
        ),
        "model.rgbd.measurement_radius_variance": (
            config.model.rgbd.measurement_radius_variance,
            1.0e-5,
        ),
        "model.rgbd.linear_drag": (config.model.rgbd.linear_drag, 0.05),
        "model.rgbd.temporal_history_size": (
            config.model.rgbd.temporal_history_size,
            16,
        ),
        "model.rgbd.temporal_min_samples": (
            config.model.rgbd.temporal_min_samples,
            16,
        ),
        "model.rgbd.fit_conditioning_limit": (
            config.model.rgbd.fit_conditioning_limit,
            100.0,
        ),
        "model.filter.enable_learned_corrector": (
            config.model.filter.enable_learned_corrector,
            False,
        ),
        "model.filter.direct_metric_position_update": (
            config.model.filter.direct_metric_position_update,
            True,
        ),
        "model.filter.innovation_anchored_correction": (
            config.model.filter.innovation_anchored_correction,
            True,
        ),
        "model.dynamics.attention_residual_enabled": (
            config.model.dynamics.attention_residual_enabled,
            False,
        ),
        "model.dynamics.analytic_free_motion_only": (
            config.model.dynamics.analytic_free_motion_only,
            True,
        ),
        "model.identification.enabled": (config.model.identification.enabled, False),
        "runtime.modality": (config.runtime.modality, "rgbd"),
        "runtime.enable_debug_oracle": (config.runtime.enable_debug_oracle, False),
        "runtime.strict_timestamps": (config.runtime.strict_timestamps, True),
        "runtime.hypothesis_pool_enabled": (
            config.runtime.hypothesis_pool_enabled,
            False,
        ),
        "training.batch_size": (config.training.batch_size, 4),
        "training.steps": (config.training.steps, 1),
        "training.train_episodes": (config.training.train_episodes, 64),
        "training.validation_episodes": (config.training.validation_episodes, 64),
        "training.fixed_dataset": (config.training.fixed_dataset, True),
        "evaluation.horizons_seconds": (
            config.evaluation.horizons_seconds,
            HORIZONS_SECONDS,
        ),
        "evaluation.episodes": (config.evaluation.episodes, 64),
    }
    for label, (actual, target) in expected.items():
        _exact_equal(actual, target, label=label)
    if config.runtime.modality_order != ("debug_oracle", "rgbd"):
        raise ValueError("runtime modality order differs from the reviewed packet boundary")
    if config.simulator.scenario_mixture != ("baseline",):
        raise ValueError("scenario mixture differs from the reviewed no-event control")
    if config.training.horizon_weights != (1.0, 1.0, 1.0, 1.0, 1.0):
        raise ValueError("training horizon weights differ from the reviewed profile")


def new_public_model(config: OrpheusConfig) -> OnlineWorldModel:
    assert_rgbd_variable_radius_config(config)
    model = OnlineWorldModel.from_config(config)
    state = model.state_dict()
    if type(state) is not dict and state.__class__.__name__ != "OrderedDict":
        raise RuntimeError("public model state must be a mapping")
    if len(state) != 0:
        raise RuntimeError("variable-radius public model must have empty persistent module state")
    parameters = tuple(model.parameters())
    buffers = tuple(model.buffers())
    if parameters or buffers:
        raise RuntimeError("variable-radius public model must own no parameters or buffers")
    return model


def bridge_protocol() -> dict[str, Any]:
    """Return the canonical self-hashed ordinal-only contract."""

    protocol: dict[str, Any] = {
        "name": "rgbd_two_visible_variable_radius_v2",
        "architecture_version": ARCHITECTURE_VERSION,
        "architecture_attempt": ARCHITECTURE_ATTEMPT,
        "maximum_architecture_attempts": MAX_ARCHITECTURE_ATTEMPTS,
        "terminal_after_attempt": True,
        "prior_architecture_attempt": _prior_architecture_attempt_disclosure(),
        "optimizer": None,
        "optimizer_updates": OPTIMIZER_UPDATES,
        "batch_size": BATCH_SIZE,
        "resolved_config_sha256": FROZEN_CONFIG_SHA256,
        "frozen_source_sha256": dict(FROZEN_SOURCE_SHA256),
        "scene_certificate": _frozen_scene_certificate_binding(),
        "manifests": {
            split: {
                "rows": list(_manifest_rows(split)),
                "sha256": MANIFEST_SHA256[split],
            }
            for split in SPLITS
        },
        "scene_family": {
            "address": "exact conceptual split plus ordinal",
            "ordinals": list(ORDINALS),
            "split_order": list(SPLITS),
            "scenes_per_split": 64,
            "objects_per_scene": 2,
            "primitives_per_split": PRIMITIVES_PER_SPLIT,
            "pair_variants_per_primitive": PAIR_VARIANTS_PER_PRIMITIVE,
            "radius_roles_per_primitive": RADIUS_ROLES_PER_PRIMITIVE,
            "camera_strata": CAMERA_STRATA,
            "counterfactual_twin": "ordinal xor 8",
            "unordered_pair_variant_twin": "ordinal xor 16",
            "fully_visible": True,
            "image_separated": True,
            "non_contact": True,
            "known_orbital_extrinsics": True,
            "unknown_metric_radius_per_object": True,
            "fixed_drag": 0.05,
        },
        "runtime": {
            "input_boundary": "ObservationPacket_only",
            "packet_synthesis": "independent_stable_metric_ray_sphere_near_root",
            "runtime": "OnlineWorldModel",
            "ingested_frame_indices": list(HISTORY_FRAME_INDICES),
            "anchor_frame_index": ANCHOR_FRAME_INDEX,
            "horizons_seconds": list(HORIZONS_SECONDS),
            "target_frame_indices": list(TARGET_FRAME_INDICES),
            "vjp_audit_ordinal_groups": [list(group) for group in VJP_AUDIT_ORDINAL_GROUPS],
            "vjp_audit_scene_denominator": len(VJP_AUDIT_ORDINALS),
            "learned_parameters": 0,
            "persistent_module_state_bytes": 0,
            "public_simulator_calls": 0,
            "public_renderer_calls": 0,
            "packet_factory_calls": 0,
        },
        "evaluator_provenance": {
            "model_visible": False,
            "receipt_schema": "variable_radius_evaluator_provenance_receipt_v3",
            "scene_evidence_field": "provenance_sha256",
            "split_result_field": "provenance_sha256",
            "ordered_split_digest": True,
            "bindings": [
                "episode identity and digest",
                "evidence truth digest",
                "ordered 16 packet identities and digests",
                "ordinal token and batch receipts",
                "durable ledger generation, record, bindings, and inode receipt",
            ],
        },
        "access": {
            "development": {
                "fixed_exclusive_durable_ledger": True,
                "receipt_before_materialisation": True,
                "single_manifest_pass": True,
                "atomic_batch_size": 4,
                "error_ledger_binds_intended_report_before_report_write": True,
            },
            "protected": {
                "order": ["selector", "confirmation", "final_test"],
                "receipt_before_materialisation": True,
                "later_split_unopened_after_any_failure": True,
                "external_review_sha256s": [
                    "development checkpoint",
                    "development report",
                    "development terminal ledger",
                ],
                "reviewed_empty_state_strictly_loaded_for_every_batch": True,
                "error_ledger_binds_intended_report_before_report_write": True,
            },
        },
        "reducers": [asdict(spec) for spec in REDUCER_REGISTRY],
        "gates": asdict(DEFAULT_GATES),
        "gate_metric_schema": list(GATE_METRIC_SCHEMA),
        "execution": {
            "device": "cpu_float32",
            "torch_intraop_threads": 1,
            "evaluation_seconds_max": DEFAULT_GATES.perception_latency_seconds,
            "vjp_seconds_max": DEFAULT_GATES.vjp_latency_seconds,
            "rss_bytes_max": DEFAULT_GATES.process_max_rss_bytes,
        },
        "threshold_design_only_public_evidence": {
            "governed_split_evidence": False,
            "formal_metric_schema_expanded": False,
            "family_sha256": "af790f40c0f3f1fad4151157e119a94ea43d49281741deeb79b7beabc7726cf5",
            "packet_trace_sha256": "02615d8bd5ccca94a14a64fe00d681dd48e628793d207a659f2fedb4f940656f",
            "radius_target_vjp_floor_l1": {
                "rgb": 1.0e-14,
                "depth": 1.0e-8,
                "intrinsics": 1.0e-8,
            },
            "combined_position_radius_vjp_floor_l1": 1.0e-8,
            "combined_position_radius_is_formal_metric": False,
            "pair_variant_estimated_anchor_position_public_upper_bound_m": 5.331e-4,
            "pair_variant_estimated_anchor_velocity_public_upper_bound_mps": 4.189e-4,
        },
        "scientific_limitations": [
            "The designed ordinal family is not iid.",
            "The fixed radius variance is explicit uncalibrated evidence.",
            "The rung isolates radius under exactly two separated spheres, zero gravity, known orbital extrinsics, fixed drag, and no contacts.",
            "Public feasibility informed thresholds but is not governed split evidence.",
            "The public combined-position-plus-radius VJP diagnostic is threshold-design-only and is not a formal metric or claim.",
            "The estimator bounds are an interval prior; twin swaps and alternate valid fixed priors test narrower shortcut classes.",
        ],
    }
    forbidden = {"seed", "dataset", "manifest_path"}
    if any(key in forbidden for key in protocol):
        raise RuntimeError("ordinal protocol acquired a forbidden selector field")
    native = _json_native(protocol, label="bridge protocol")
    if type(native) is not dict:
        raise RuntimeError("bridge protocol must be one exact JSON object")
    native["protocol_sha256"] = canonical_sha256(native)
    return native


def _frozen_config_path() -> Path:
    return REPOSITORY_ROOT / "configs" / "rgbd_variable_radius_cpu.yaml"


def require_frozen_config(path: str | Path) -> OrpheusConfig:
    source = Path(path)
    contents = stable_read_bytes(source, label="frozen variable-radius config")
    actual = sha256_bytes(contents)
    if actual != FROZEN_CONFIG_SHA256:
        raise ValueError(
            "variable-radius qualification requires exact frozen config bytes: "
            f"expected {FROZEN_CONFIG_SHA256}, got {actual}"
        )
    try:
        text = contents.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("frozen variable-radius config must be UTF-8") from error
    loaded = yaml.safe_load(text)
    if type(loaded) is not dict:
        raise TypeError("frozen variable-radius config must be an exact mapping")
    loaded = copy.deepcopy(loaded)
    loaded["source_path"] = str(source.expanduser().resolve())
    config = _strict_construct(OrpheusConfig, loaded, "config")
    config.validate()
    assert_rgbd_variable_radius_config(config)
    return config


def _require_config_matches_frozen_path(config: OrpheusConfig, path: Path) -> None:
    if path.resolve() != _frozen_config_path().resolve():
        raise ValueError("qualification config path must be the canonical frozen path")
    loaded = require_frozen_config(path)
    _exact_equal(config.to_dict(), loaded.to_dict(), label="in-memory config")


_GIT_OBJECT_FORMAT = "sha1"
_GIT_OID_BYTES = 20
_GIT_MAX_OUTPUT_BYTES = 4 * 1024 * 1024


def _git_repository_paths(root: Path) -> tuple[Path, Path, Path]:
    if type(root) is not _NATIVE_PATH_TYPE or not root.is_absolute() or root.resolve() != root:
        raise PermissionError("Git work tree must be one exact canonical native Path")
    git_dir = root / ".git"
    git_stat = git_dir.lstat()
    if not stat.S_ISDIR(git_stat.st_mode) or git_dir.resolve() != git_dir:
        raise PermissionError("Git directory must be the exact canonical .git directory")
    common_dir = git_dir
    return git_dir, common_dir, root


def _git_environment(root: Path) -> dict[str, str]:
    git_dir, common_dir, work_tree = _git_repository_paths(root)
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


def _git_command(root: Path, arguments: Sequence[str]) -> list[str]:
    git_dir, _, work_tree = _git_repository_paths(root)
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


def _validate_repository_identity() -> None:
    if REPOSITORY_ROOT.resolve() != Path(__file__).resolve().parents[2]:
        raise RuntimeError("qualification repository root changed")
    git_dir, common_dir, work_tree = _git_repository_paths(REPOSITORY_ROOT)
    if (
        Path(
            _git_text(
                REPOSITORY_ROOT,
                ["rev-parse", "--show-toplevel"],
                label="repository root",
            )
        ).resolve()
        != work_tree
        or Path(
            _git_text(
                REPOSITORY_ROOT,
                ["rev-parse", "--absolute-git-dir"],
                label="Git directory",
            )
        ).resolve()
        != git_dir
        or Path(
            _git_text(
                REPOSITORY_ROOT,
                ["rev-parse", "--path-format=absolute", "--git-common-dir"],
                label="Git common directory",
            )
        ).resolve()
        != common_dir
        or _git_text(
            REPOSITORY_ROOT,
            ["rev-parse", "--show-object-format"],
            label="Git object format",
        )
        != _GIT_OBJECT_FORMAT
    ):
        raise RuntimeError("qualification Git repository identity differs")


def _validate_frozen_critical_sources() -> None:
    if set(FROZEN_SOURCE_SHA256) != set(FROZEN_SOURCE_PATHS):
        raise RuntimeError("frozen source name/path schemas differ")
    for name, relative in FROZEN_SOURCE_PATHS.items():
        contents = stable_read_bytes(REPOSITORY_ROOT / relative, label=f"frozen source {name}")
        actual = sha256_bytes(contents)
        expected = FROZEN_SOURCE_SHA256[name]
        if actual != expected:
            raise RuntimeError(f"frozen source {name} differs: expected {expected}, got {actual}")


def _validated_git_oid(value: Any, *, label: str) -> str:
    if type(value) is not str or len(value) != 2 * _GIT_OID_BYTES:
        raise ValueError(f"{label} must be an exact Git object ID")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} must be hexadecimal") from error
    if value != value.lower():
        raise ValueError(f"{label} must use lowercase hexadecimal")
    return value


def _git_text(root: Path, arguments: Sequence[str], *, label: str) -> str:
    raw = _git_bytes(root, arguments, label=label)
    try:
        value = raw.decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError as error:
        raise RuntimeError(f"git returned non-UTF-8 {label}") from error
    if not value:
        raise RuntimeError(f"git returned empty {label}")
    return value


def _git_bytes(root: Path, arguments: Sequence[str], *, label: str) -> bytes:
    result = subprocess.run(
        _git_command(root, arguments),
        cwd=root,
        env=_git_environment(root),
        check=True,
        capture_output=True,
        text=False,
    )
    if type(result.stdout) is not bytes or len(result.stdout) > _GIT_MAX_OUTPUT_BYTES:
        raise TypeError(f"git returned invalid or oversized {label}")
    return result.stdout


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
    root: Path,
    oid: str,
    *,
    object_type: Literal["blob", "commit", "tree"],
    label: str,
    object_cache: dict[tuple[str, str], bytes] | None = None,
) -> bytes:
    oid = _validated_git_oid(oid, label=f"{label} object")
    cache_key = (object_type, oid)
    cached = None if object_cache is None else object_cache.get(cache_key)
    if cached is not None:
        if type(cached) is not bytes or _git_object_oid(object_type, cached) != oid:
            raise PermissionError(f"{label} cached Git object framing differs")
        return cached
    reported_type = _git_text(
        root,
        ["cat-file", "-t", oid],
        label=f"{label} object type",
    )
    size_text = _git_text(
        root,
        ["cat-file", "-s", oid],
        label=f"{label} object size",
    )
    if not size_text.isascii() or not size_text.isdecimal():
        raise PermissionError(f"{label} Git object size is not canonical")
    reported_size = int(size_text)
    if str(reported_size) != size_text:
        raise PermissionError(f"{label} Git object size is not canonical")
    contents = _git_bytes(
        root,
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
    root: Path,
    commit: str,
    *,
    label: str,
    object_cache: dict[tuple[str, str], bytes] | None = None,
) -> str:
    contents = _git_verified_object(
        root,
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
    root: Path,
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
        root,
        commit,
        label=f"{label} commit",
        object_cache=object_cache,
    )
    parts = relative.encode("utf-8", errors="strict").split(b"/")
    for index, part in enumerate(parts):
        tree = _git_verified_object(
            root,
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


def _capture_git_publication_state(
    root: Path,
    *,
    object_cache: dict[tuple[str, str], bytes] | None = None,
) -> dict[str, Any]:
    status = _git_text_allow_empty(
        root,
        ["status", "--porcelain=v1", "--untracked-files=all"],
        label="publication status",
    )
    if status:
        raise RuntimeError("qualification requires clean committed published source")
    object_format = _git_text(
        root,
        ["rev-parse", "--show-object-format"],
        label="Git object format",
    )
    if object_format != _GIT_OBJECT_FORMAT:
        raise RuntimeError("qualification Git object format differs")
    commit = _validated_git_oid(
        _git_text(root, ["rev-parse", "--verify", "HEAD"], label="HEAD commit"),
        label="HEAD commit",
    )
    tree_oid = _git_commit_tree_oid(
        root,
        commit,
        label="HEAD commit",
        object_cache=object_cache,
    )
    _git_verified_object(
        root,
        tree_oid,
        object_type="tree",
        label="HEAD tree",
        object_cache=object_cache,
    )
    upstream_ref = _git_text(
        root,
        ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"],
        label="upstream ref",
    )
    if upstream_ref == "HEAD":
        raise RuntimeError("qualification requires one configured branch upstream")
    upstream_commit = _validated_git_oid(
        _git_text(
            root,
            ["rev-parse", "--verify", "@{upstream}"],
            label="upstream commit",
        ),
        label="upstream commit",
    )
    _git_verified_object(
        root,
        upstream_commit,
        object_type="commit",
        label="upstream commit",
        object_cache=object_cache,
    )
    counts = _git_text(
        root,
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


def _git_text_allow_empty(root: Path, arguments: Sequence[str], *, label: str) -> str:
    raw = _git_bytes(root, arguments, label=label)
    try:
        return raw.decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError as error:
        raise RuntimeError(f"git returned non-UTF-8 {label}") from error


def _git_blob_binding(
    root: Path,
    *,
    commit: str,
    relative: str,
    label: str,
    object_cache: dict[tuple[str, str], bytes] | None = None,
) -> tuple[dict[str, str], bytes]:
    commit = _validated_git_oid(commit, label=f"{label} commit")
    entry = _git_tree_entry(
        root,
        commit=commit,
        relative=relative,
        label=label,
        object_cache=object_cache,
    )
    assert entry is not None
    mode, blob_oid = entry
    if mode not in {"100644", "100755"}:
        raise PermissionError(f"{label} exact Git blob mode differs")
    blob_contents = _git_verified_object(
        root,
        blob_oid,
        object_type="blob",
        label=f"{label} blob",
        object_cache=object_cache,
    )
    return (
        {
            "path": relative,
            "mode": mode,
            "blob_oid": blob_oid,
            "blob_sha256": sha256_bytes(blob_contents),
            "worktree_sha256": sha256_bytes(blob_contents),
        },
        blob_contents,
    )


def capture_published_source(root: Path) -> dict[str, Any]:
    if type(root) is not _NATIVE_PATH_TYPE:
        raise TypeError("publication root must be a native Path")
    if root != REPOSITORY_ROOT or root.resolve() != REPOSITORY_ROOT.resolve():
        raise ValueError("publication root must be the exact canonical repository root")
    _validate_repository_identity()
    _validate_frozen_critical_sources()
    object_cache: dict[tuple[str, str], bytes] = {}
    publication_git = _capture_git_publication_state(root, object_cache=object_cache)
    commit = publication_git["commit"]
    metadata = clean_source(
        {
            "commit": commit,
            "dirty": False,
            "worktree_fingerprint": sha256_bytes(commit.encode("ascii")),
            "runtime_source_fingerprint": canonical_sha256(FROZEN_SOURCE_SHA256),
        },
        label="published",
    )
    publication_surface_sha256: dict[str, str] = {}
    publication_surface_blobs: dict[str, dict[str, str]] = {}
    for name, relative in PUBLICATION_SURFACE_PATHS.items():
        blob_binding, blob_contents = _git_blob_binding(
            root,
            commit=commit,
            relative=relative,
            label=f"published surface {name}",
            object_cache=object_cache,
        )
        before = stable_read_bytes(root / relative, label=f"published surface {name}")
        after = stable_read_bytes(root / relative, label=f"published surface {name}")
        if before != blob_contents or after != blob_contents:
            raise RuntimeError(f"published surface {name} differs from its exact HEAD blob")
        publication_surface_sha256[name] = sha256_bytes(blob_contents)
        publication_surface_blobs[name] = blob_binding
    _validate_frozen_critical_sources()
    final_git = _capture_git_publication_state(root)
    if final_git != publication_git:
        raise RuntimeError("published source changed while clean provenance was captured")
    return {
        **metadata,
        "frozen_source_sha256": dict(FROZEN_SOURCE_SHA256),
        "resolved_config_sha256": FROZEN_CONFIG_SHA256,
        "scene_certificate": _frozen_scene_certificate_binding(),
        "publication_git": publication_git,
        "publication_surface_sha256": publication_surface_sha256,
        "publication_surface_blobs": publication_surface_blobs,
    }


def _validated_published_source(
    value: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{label} must be an exact dict")
    expected = {
        "commit",
        "dirty",
        "worktree_fingerprint",
        "runtime_source_fingerprint",
        "frozen_source_sha256",
        "resolved_config_sha256",
        "scene_certificate",
        "publication_git",
        "publication_surface_sha256",
        "publication_surface_blobs",
    }
    if set(value) != expected:
        raise ValueError(f"{label} has the wrong exact schema")
    cleaned = clean_source(
        {
            key: value[key]
            for key in (
                "commit",
                "dirty",
                "worktree_fingerprint",
                "runtime_source_fingerprint",
            )
        },
        label=label,
    )
    if value["frozen_source_sha256"] != FROZEN_SOURCE_SHA256:
        raise ValueError(f"{label} frozen source binding differs")
    if value["resolved_config_sha256"] != FROZEN_CONFIG_SHA256:
        raise ValueError(f"{label} frozen config binding differs")
    if value["scene_certificate"] != _frozen_scene_certificate_binding():
        raise ValueError(f"{label} scene certificate binding differs")
    publication_git = value["publication_git"]
    if type(publication_git) is not dict or set(publication_git) != {
        "commit",
        "tree_oid",
        "object_format",
        "upstream_ref",
        "upstream_commit",
        "ahead",
        "behind",
    }:
        raise ValueError(f"{label} publication Git schema differs")
    if (
        _validated_git_oid(publication_git["commit"], label=f"{label} Git commit")
        != cleaned["commit"]
        or publication_git["object_format"] != _GIT_OBJECT_FORMAT
        or _validated_git_oid(
            publication_git["tree_oid"],
            label=f"{label} Git tree",
        )
        != publication_git["tree_oid"]
        or type(publication_git["upstream_ref"]) is not str
        or not publication_git["upstream_ref"]
        or publication_git["upstream_ref"] == "HEAD"
        or _validated_git_oid(
            publication_git["upstream_commit"],
            label=f"{label} upstream commit",
        )
        != cleaned["commit"]
        or type(publication_git["ahead"]) is not int
        or type(publication_git["behind"]) is not int
        or publication_git["ahead"] != 0
        or publication_git["behind"] != 0
    ):
        raise ValueError(f"{label} publication Git binding differs")
    publication = value["publication_surface_sha256"]
    if type(publication) is not dict or set(publication) != set(PUBLICATION_SURFACE_PATHS):
        raise ValueError(f"{label} publication surface schema differs")
    for name in PUBLICATION_SURFACE_PATHS:
        digest = publication[name]
        validated_sha256(digest, label=f"{label} publication surface {name}")
    blobs = value["publication_surface_blobs"]
    if type(blobs) is not dict or set(blobs) != set(PUBLICATION_SURFACE_PATHS):
        raise ValueError(f"{label} publication blob schema differs")
    validated_blobs: dict[str, dict[str, str]] = {}
    for name, relative in PUBLICATION_SURFACE_PATHS.items():
        binding = blobs[name]
        if type(binding) is not dict or set(binding) != {
            "path",
            "mode",
            "blob_oid",
            "blob_sha256",
            "worktree_sha256",
        }:
            raise ValueError(f"{label} publication blob {name} schema differs")
        if (
            type(binding["path"]) is not str
            or binding["path"] != relative
            or type(binding["mode"]) is not str
            or len(binding["mode"]) != 6
            or not binding["mode"].isdigit()
        ):
            raise ValueError(f"{label} publication blob {name} path or mode differs")
        blob_oid = _validated_git_oid(
            binding["blob_oid"],
            label=f"{label} publication blob {name}",
        )
        blob_sha256 = validated_sha256(
            binding["blob_sha256"],
            label=f"{label} publication blob {name} SHA-256",
        )
        worktree_sha256 = validated_sha256(
            binding["worktree_sha256"],
            label=f"{label} publication worktree {name} SHA-256",
        )
        if blob_sha256 != worktree_sha256 or blob_sha256 != publication[name]:
            raise ValueError(f"{label} publication blob {name} digest binding differs")
        validated_blobs[name] = {
            "path": relative,
            "mode": binding["mode"],
            "blob_oid": blob_oid,
            "blob_sha256": blob_sha256,
            "worktree_sha256": worktree_sha256,
        }
    return {
        **cleaned,
        "frozen_source_sha256": dict(FROZEN_SOURCE_SHA256),
        "resolved_config_sha256": FROZEN_CONFIG_SHA256,
        "scene_certificate": _frozen_scene_certificate_binding(),
        "publication_git": copy.deepcopy(publication_git),
        "publication_surface_sha256": dict(publication),
        "publication_surface_blobs": validated_blobs,
    }


def _validate_publication_surface(value: Mapping[str, Any]) -> None:
    source = _validated_published_source(dict(value), label="publication boundary source")
    publication = source["publication_surface_sha256"]
    blobs = source["publication_surface_blobs"]
    current_git = _capture_git_publication_state(REPOSITORY_ROOT)
    _exact_equal(
        current_git,
        source["publication_git"],
        label="publication boundary Git state",
    )
    for name, relative in PUBLICATION_SURFACE_PATHS.items():
        expected = validated_sha256(publication[name], label=f"publication surface {name}")
        current_binding, blob_contents = _git_blob_binding(
            REPOSITORY_ROOT,
            commit=current_git["commit"],
            relative=relative,
            label=f"current publication surface {name}",
        )
        _exact_equal(
            current_binding,
            blobs[name],
            label=f"publication surface {name} HEAD blob",
        )
        before = stable_read_bytes(
            REPOSITORY_ROOT / relative,
            label=f"current publication surface {name}",
        )
        after = stable_read_bytes(
            REPOSITORY_ROOT / relative,
            label=f"current publication surface {name}",
        )
        if (
            before != blob_contents
            or after != blob_contents
            or sha256_bytes(before) != expected
            or sha256_bytes(after) != expected
        ):
            raise PermissionError(
                f"published qualification surface {name} differs from exact HEAD blob"
            )
    final_git = _capture_git_publication_state(REPOSITORY_ROOT)
    _exact_equal(final_git, current_git, label="publication boundary final Git state")


def _canonical_run_directory() -> Path:
    return REPOSITORY_ROOT / RUN_RELATIVE_PATH


def canonical_development_report_path() -> Path:
    return _canonical_run_directory() / DEVELOPMENT_REPORT_NAME


def canonical_checkpoint_path() -> Path:
    return _canonical_run_directory() / CHECKPOINT_NAME


def canonical_qualification_report_path() -> Path:
    return _canonical_run_directory() / QUALIFICATION_REPORT_NAME


def development_ledger_path() -> Path:
    return _canonical_run_directory() / DEVELOPMENT_LEDGER_NAME


def qualification_ledger_path() -> Path:
    return _canonical_run_directory() / QUALIFICATION_LEDGER_NAME


def _is_owned_run_artifact_path(path: Path) -> bool:
    return (
        type(path) is _NATIVE_PATH_TYPE
        and path.parent == _canonical_run_directory()
        and (
            path.name in QUALIFICATION_ARTIFACT_NAMES
            or (
                path.name.endswith(".tmp")
                and path.name[: -len(".tmp")] in QUALIFICATION_ARTIFACT_NAMES
            )
        )
    )


def _reject_generic_run_artifact_path(path: Path, *, label: str) -> None:
    if _is_owned_run_artifact_path(path):
        raise PermissionError(f"{label} requires the pinned run-directory capability")


def _lexists(path: Path) -> bool:
    _reject_generic_run_artifact_path(path, label="lexists")
    return os.path.lexists(os.fspath(path))


def _atomic_temporary(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".tmp")


def _require_canonical_path(
    actual: Path,
    expected: Path,
    *,
    label: str,
    directory_pin: _PinnedDirectory | None = None,
) -> None:
    if type(actual) is not _NATIVE_PATH_TYPE:
        raise TypeError(f"{label} must be a native Path")
    if actual != expected:
        raise ValueError(f"{label} must equal canonical path {expected}")
    if expected.parent == _canonical_run_directory():
        if directory_pin is None:
            raise PermissionError(f"{label} requires the pinned run-directory capability")
        _pinned_artifact_name(directory_pin, actual, label=label)
        return
    if actual.resolve(strict=False) != expected.resolve(strict=False):
        raise ValueError(f"{label} must equal canonical path {expected}")


def _require_nonlink_directory(path: Path, *, label: str) -> os.stat_result:
    if path == _canonical_run_directory():
        raise PermissionError(f"{label} requires the pinned run-directory capability")
    before = os.lstat(path)
    if stat.S_ISLNK(before.st_mode):
        raise ValueError(f"{label} must not be a symbolic link")
    if not stat.S_ISDIR(before.st_mode):
        raise ValueError(f"{label} must be a directory")
    after = os.lstat(path)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after:
        raise RuntimeError(f"{label} changed during lstat validation")
    return after


def _require_single_link_regular(path: Path, *, label: str) -> os.stat_result:
    _reject_generic_run_artifact_path(path, label=label)
    before = os.lstat(path)
    if stat.S_ISLNK(before.st_mode):
        raise ValueError(f"{label} must not be a symbolic link")
    metadata = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular file")
    if metadata.st_nlink != 1:
        raise ValueError(f"{label} must have exactly one hard link")
    after = os.lstat(path)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
        before.st_nlink,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
        after.st_nlink,
    )
    if identity_before != identity_after:
        raise RuntimeError(f"{label} changed during lstat validation")
    return metadata


def stable_read_bytes(path: str | Path, *, label: str) -> bytes:
    """Read one exact single-link inode through a no-follow descriptor."""

    source = Path(path)
    _reject_generic_run_artifact_path(source, label=label)
    before_path = _require_single_link_regular(source, label=label)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    try:
        before_open = os.fstat(descriptor)
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            contents = handle.read()
        after_open = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = _require_single_link_regular(source, label=label)

    def identity(metadata: os.stat_result) -> tuple[int, ...]:
        return (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
            metadata.st_nlink,
        )

    expected = identity(before_path)
    if (
        identity(before_open) != expected
        or identity(after_open) != expected
        or identity(after_path) != expected
        or len(contents) != before_open.st_size
    ):
        raise RuntimeError(f"{label} changed while its pinned inode was being read")
    return contents


@dataclass(frozen=True, slots=True)
class _PinnedDirectory:
    path: Path
    parent_path: Path
    child_name: str
    parent_fd: int
    directory_fd: int
    parent_identity: tuple[int, int, int]
    directory_identity: tuple[int, int, int]
    canonical: bool
    owner_thread: int
    nonce: object


@dataclass(frozen=True, slots=True)
class _PinnedDirectoryRegistration:
    pin: _PinnedDirectory
    owner_thread: int
    parent_identity: tuple[int, int, int]
    directory_identity: tuple[int, int, int]
    status: Literal["live"]


def _build_pinned_directory_vault() -> tuple[
    Callable[[_PinnedDirectory], None],
    Callable[[_PinnedDirectory], _PinnedDirectoryRegistration],
    Callable[[_PinnedDirectory, tuple[int, int, int]], None],
    Callable[[_PinnedDirectory], None],
    Callable[[], None],
]:
    records: dict[int, _PinnedDirectoryRegistration] = {}

    def issue(pin: _PinnedDirectory) -> None:
        caller = inspect.currentframe().f_back
        if (
            caller is None
            or caller.f_code is not _acquire_pinned_directory.__code__
            or id(pin) in records
        ):
            raise PermissionError("directory pin issue is not acquisition-owned")
        records[id(pin)] = _PinnedDirectoryRegistration(
            pin=pin,
            owner_thread=pin.owner_thread,
            parent_identity=pin.parent_identity,
            directory_identity=pin.directory_identity,
            status="live",
        )

    def registration(pin: _PinnedDirectory) -> _PinnedDirectoryRegistration:
        record = records.get(id(pin))
        if (
            not isinstance(pin, _PinnedDirectory)
            or not isinstance(record, _PinnedDirectoryRegistration)
            or record.pin is not pin
            or record.owner_thread != threading.get_ident()
            or record.status != "live"
        ):
            raise PermissionError("directory pin is forged, closed, or thread-rebound")
        return record

    def refresh(
        pin: _PinnedDirectory,
        directory_identity: tuple[int, int, int],
    ) -> None:
        caller = inspect.currentframe().f_back
        record = records.get(id(pin))
        if (
            caller is None
            or caller.f_code is not _refresh_pinned_directory_after_owned_mutation.__code__
            or not isinstance(record, _PinnedDirectoryRegistration)
            or record.pin is not pin
            or record.owner_thread != threading.get_ident()
            or record.status != "live"
            or type(directory_identity) is not tuple
            or len(directory_identity) != 3
            or any(type(value) is not int for value in directory_identity)
        ):
            raise PermissionError("directory pin refresh is not owned by a durable mutation")
        records[id(pin)] = _PinnedDirectoryRegistration(
            pin=pin,
            owner_thread=record.owner_thread,
            parent_identity=record.parent_identity,
            directory_identity=directory_identity,
            status="live",
        )

    def retire(pin: _PinnedDirectory) -> None:
        record = records.pop(id(pin), None)
        if (
            not isinstance(record, _PinnedDirectoryRegistration)
            or record.pin is not pin
            or record.owner_thread != threading.get_ident()
        ):
            raise PermissionError("directory pin cannot be closed twice or by another thread")
        close_errors: list[OSError] = []
        for descriptor in (pin.directory_fd, pin.parent_fd):
            try:
                os.close(descriptor)
            except OSError as error:
                close_errors.append(error)
        if close_errors:
            raise RuntimeError("directory pin descriptor close failed") from close_errors[0]

    def clear() -> None:
        for record in tuple(records.values()):
            for descriptor in (record.pin.directory_fd, record.pin.parent_fd):
                with contextlib.suppress(OSError):
                    os.close(descriptor)
        records.clear()

    return issue, registration, refresh, retire, clear


(
    _issue_pinned_directory,
    _pinned_directory_registration,
    _refresh_pinned_directory_registration,
    _retire_pinned_directory,
    _clear_pinned_directory_vault_for_tests,
) = _build_pinned_directory_vault()


def _directory_identity(metadata: os.stat_result, *, label: str) -> tuple[int, int, int]:
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} must be a directory")
    if metadata.st_nlink < 1:
        raise ValueError(f"{label} must have at least one link")
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
    )


def _directory_path_identity(path: Path, *, label: str) -> tuple[int, int, int]:
    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"{label} must not be a symbolic link")
    return _directory_identity(metadata, label=label)


def _safe_basename(name: str, *, label: str) -> str:
    if (
        type(name) is not str
        or not name
        or name in {".", ".."}
        or os.path.sep in name
        or (os.path.altsep is not None and os.path.altsep in name)
    ):
        raise ValueError(f"{label} must be one exact relative basename")
    return name


def _acquire_pinned_directory(
    path: Path,
    *,
    create: bool,
    canonical: bool,
) -> _PinnedDirectory:
    if type(path) is not _NATIVE_PATH_TYPE:
        raise TypeError("pinned directory path must be a native Path")
    if canonical and path != _canonical_run_directory():
        raise ValueError("formal directory pin must name the exact canonical run directory")
    parent_path = path.parent
    child_name = _safe_basename(path.name, label="pinned directory child")
    parent_before = _directory_path_identity(parent_path, label="pinned parent path")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    parent_fd = os.open(parent_path, directory_flags)
    directory_fd: int | None = None
    issued_pin: _PinnedDirectory | None = None
    try:
        parent_open = _directory_identity(os.fstat(parent_fd), label="pinned parent descriptor")
        if parent_open != parent_before:
            raise RuntimeError("pinned parent changed while its descriptor was opened")
        created = False
        if create:
            try:
                os.mkdir(child_name, mode=0o700, dir_fd=parent_fd)
                created = True
            except FileExistsError:
                pass
            if created:
                os.fsync(parent_fd)
        child_entry = os.stat(child_name, dir_fd=parent_fd, follow_symlinks=False)
        if stat.S_ISLNK(child_entry.st_mode):
            raise ValueError("pinned directory child must not be a symbolic link")
        child_identity = _directory_identity(child_entry, label="pinned directory child")
        directory_fd = os.open(child_name, directory_flags, dir_fd=parent_fd)
        directory_open = _directory_identity(
            os.fstat(directory_fd),
            label="pinned directory descriptor",
        )
        parent_identity = _directory_identity(
            os.fstat(parent_fd),
            label="pinned parent descriptor",
        )
        if directory_open != child_identity:
            raise RuntimeError("pinned directory changed while its descriptor was opened")
        if _directory_path_identity(parent_path, label="pinned parent path") != parent_identity:
            raise RuntimeError("pinned parent namespace changed during acquisition")
        if _directory_path_identity(path, label="pinned directory path") != directory_open:
            raise RuntimeError("pinned directory namespace changed during acquisition")
        pin = _PinnedDirectory(
            path=path,
            parent_path=parent_path,
            child_name=child_name,
            parent_fd=parent_fd,
            directory_fd=directory_fd,
            parent_identity=parent_identity,
            directory_identity=directory_open,
            canonical=canonical,
            owner_thread=threading.get_ident(),
            nonce=object(),
        )
        _issue_pinned_directory(pin)
        issued_pin = pin
        _validate_pinned_directory(pin)
        return pin
    except BaseException:
        if issued_pin is not None:
            _retire_pinned_directory(issued_pin)
        else:
            if directory_fd is not None:
                os.close(directory_fd)
            os.close(parent_fd)
        raise


def _validate_pinned_directory(pin: _PinnedDirectory) -> None:
    registration = _pinned_directory_registration(pin)
    if (
        pin.owner_thread != threading.get_ident()
        or registration.parent_identity != pin.parent_identity
        or registration.directory_identity != pin.directory_identity
    ):
        raise PermissionError("directory pin registry binding changed")
    parent_open = _directory_identity(os.fstat(pin.parent_fd), label="pinned parent descriptor")
    directory_open = _directory_identity(
        os.fstat(pin.directory_fd),
        label="pinned directory descriptor",
    )
    child_entry = os.stat(pin.child_name, dir_fd=pin.parent_fd, follow_symlinks=False)
    if stat.S_ISLNK(child_entry.st_mode):
        raise PermissionError("canonical run-directory entry became a symbolic link")
    if (
        parent_open != pin.parent_identity
        or directory_open != registration.directory_identity
        or _directory_identity(child_entry, label="pinned directory child")
        != registration.directory_identity
        or _directory_path_identity(pin.parent_path, label="pinned parent path")
        != pin.parent_identity
        or _directory_path_identity(pin.path, label="pinned directory path")
        != registration.directory_identity
    ):
        raise PermissionError("pinned directory or canonical namespace binding changed")


def _raw_pinned_inventory(pin: _PinnedDirectory) -> frozenset[str]:
    return frozenset(entry.name for entry in os.scandir(pin.directory_fd))


def _refresh_pinned_directory_after_owned_mutation(
    pin: _PinnedDirectory,
    *,
    before_names: frozenset[str],
    after_names: frozenset[str],
) -> None:
    registration = _pinned_directory_registration(pin)
    if type(before_names) is not frozenset or type(after_names) is not frozenset:
        raise TypeError("directory mutation inventories must be exact frozensets")
    if any(type(name) is not str for name in (*before_names, *after_names)):
        raise TypeError("directory mutation inventory names must be exact strings")
    parent_open = _directory_identity(os.fstat(pin.parent_fd), label="pinned parent descriptor")
    directory_open = _directory_identity(
        os.fstat(pin.directory_fd),
        label="pinned directory descriptor",
    )
    child_entry = _directory_identity(
        os.stat(pin.child_name, dir_fd=pin.parent_fd, follow_symlinks=False),
        label="pinned directory child",
    )
    if (
        parent_open != registration.parent_identity
        or _directory_path_identity(pin.parent_path, label="pinned parent path")
        != registration.parent_identity
        or directory_open != pin.directory_identity
        or child_entry != directory_open
        or _directory_path_identity(pin.path, label="pinned directory path") != directory_open
        or _raw_pinned_inventory(pin) != after_names
    ):
        raise PermissionError("owned directory mutation changed its pinned namespace")
    _refresh_pinned_directory_registration(pin, directory_open)
    _validate_pinned_directory(pin)


def _release_pinned_directory(pin: _PinnedDirectory) -> None:
    validation_error: BaseException | None = None
    try:
        _validate_pinned_directory(pin)
    except BaseException as error:
        validation_error = error
    _retire_pinned_directory(pin)
    if validation_error is not None:
        raise PermissionError("directory pin closed after namespace validation failed") from (
            validation_error
        )


def _pinned_directory_binding(pin: _PinnedDirectory) -> dict[str, Any]:
    _validate_pinned_directory(pin)
    return {
        "schema": "rgbd_variable_radius_run_directory_v2",
        "path": os.fspath(pin.path),
        "parent_identity": list(pin.parent_identity),
        "directory_identity": list(pin.directory_identity),
        "canonical": pin.canonical,
    }


def _pinned_directory_capability_sha256(pin: _PinnedDirectory) -> str:
    return canonical_sha256(
        {
            **_pinned_directory_binding(pin),
            "parent_fd": pin.parent_fd,
            "directory_fd": pin.directory_fd,
            "owner_thread": pin.owner_thread,
        }
    )


def _pinned_artifact_name(pin: _PinnedDirectory, path: Path, *, label: str) -> str:
    _validate_pinned_directory(pin)
    if type(path) is not _NATIVE_PATH_TYPE or path.parent != pin.path:
        raise ValueError(f"{label} must be directly inside the pinned directory")
    name = _safe_basename(path.name, label=label)
    if pin.canonical and not _is_owned_run_artifact_path(path):
        raise PermissionError(f"{label} is not an owned qualification artifact")
    return name


def _pinned_lexists(pin: _PinnedDirectory, path: Path, *, label: str) -> bool:
    name = _pinned_artifact_name(pin, path, label=label)
    try:
        os.stat(name, dir_fd=pin.directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        _validate_pinned_directory(pin)
        return False
    _validate_pinned_directory(pin)
    return True


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        metadata.st_nlink,
    )


def _pinned_require_single_link_regular(
    pin: _PinnedDirectory,
    path: Path,
    *,
    label: str,
) -> os.stat_result:
    name = _pinned_artifact_name(pin, path, label=label)
    before = os.stat(name, dir_fd=pin.directory_fd, follow_symlinks=False)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} must be a non-link regular file")
    if before.st_nlink != 1:
        raise ValueError(f"{label} must have exactly one hard link")
    after = os.stat(name, dir_fd=pin.directory_fd, follow_symlinks=False)
    _validate_pinned_directory(pin)
    if _file_identity(before) != _file_identity(after):
        raise RuntimeError(f"{label} changed during pinned stat validation")
    return after


def _pinned_stable_read_bytes(
    pin: _PinnedDirectory,
    path: Path,
    *,
    label: str,
) -> bytes:
    name = _pinned_artifact_name(pin, path, label=label)
    before = _pinned_require_single_link_regular(pin, path, label=label)
    descriptor = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=pin.directory_fd,
    )
    try:
        before_open = os.fstat(descriptor)
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            contents = handle.read()
        after_open = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = _pinned_require_single_link_regular(pin, path, label=label)
    expected = _file_identity(before)
    if (
        _file_identity(before_open) != expected
        or _file_identity(after_open) != expected
        or _file_identity(after) != expected
        or len(contents) != before.st_size
    ):
        raise RuntimeError(f"{label} changed while its pinned inode was read")
    _validate_pinned_directory(pin)
    return contents


def _pinned_artifact_identity(
    pin: _PinnedDirectory,
    path: Path,
) -> tuple[int, int, int, int, int]:
    metadata = _pinned_require_single_link_regular(
        pin,
        path,
        label=f"artifact identity {path.name}",
    )
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_nlink,
    )


def _validate_distinct_canonical_paths(
    pin: _PinnedDirectory,
    paths: Mapping[str, Path],
) -> None:
    if type(paths) is not dict or not paths:
        raise TypeError("artifact path inventory must be a nonempty exact dict")
    expanded: dict[str, tuple[Path, str]] = {}
    for name, path in paths.items():
        if type(name) is not str or not isinstance(path, _NATIVE_PATH_TYPE):
            raise TypeError("artifact path inventory has invalid key or path")
        expanded[name] = (path, _pinned_artifact_name(pin, path, label=name))
        temporary = _atomic_temporary(path)
        expanded[f"{name}_temporary"] = (
            temporary,
            _pinned_artifact_name(pin, temporary, label=f"{name} temporary"),
        )
    names = [item[1] for item in expanded.values()]
    if len(names) != len(set(names)):
        raise ValueError("qualification artifact basenames alias")
    identities: dict[tuple[int, int], str] = {}
    for label, (path, _) in expanded.items():
        if not _pinned_lexists(pin, path, label=label):
            continue
        metadata = _pinned_require_single_link_regular(pin, path, label=label)
        identity = (metadata.st_dev, metadata.st_ino)
        if identity in identities:
            raise ValueError(
                f"qualification artifacts hard-link alias: {identities[identity]}, {label}"
            )
        identities[identity] = label
    _validate_pinned_directory(pin)


def _validate_run_tree(
    pin: _PinnedDirectory,
    expected_names: frozenset[str],
    *,
    stage: str,
) -> None:
    _validate_pinned_directory(pin)
    actual = frozenset(entry.name for entry in os.scandir(pin.directory_fd))
    _validate_pinned_directory(pin)
    if actual != expected_names:
        raise RuntimeError(
            f"{stage} run tree differs: expected {sorted(expected_names)}, got {sorted(actual)}"
        )
    for name in actual:
        _pinned_require_single_link_regular(
            pin,
            pin.path / name,
            label=f"{stage} artifact {name}",
        )
    for owned in QUALIFICATION_ARTIFACT_NAMES:
        temporary = _atomic_temporary(pin.path / owned)
        if _pinned_lexists(pin, temporary, label=f"{stage} temporary {owned}"):
            raise RuntimeError(f"{stage} found unresolved atomic temporary {temporary.name}")
    _validate_pinned_directory(pin)


def _write_descriptor(descriptor: int, contents: bytes) -> os.stat_result:
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(contents)
        handle.flush()
        os.fsync(handle.fileno())
        metadata = os.fstat(handle.fileno())
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size != len(contents)
        ):
            raise RuntimeError("durable writer descriptor identity differs")
        return metadata


def _pinned_durable_create(
    pin: _PinnedDirectory,
    path: Path,
    contents: bytes,
    *,
    mode: int = 0o600,
) -> None:
    if not isinstance(contents, bytes):
        raise TypeError("durable contents must be exact bytes")
    name = _pinned_artifact_name(pin, path, label="durable create artifact")
    before_names = _raw_pinned_inventory(pin)
    if name in before_names:
        raise FileExistsError(f"durable create artifact must be fresh: {name}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, mode, dir_fd=pin.directory_fd)
    written = _write_descriptor(descriptor, contents)
    _refresh_pinned_directory_after_owned_mutation(
        pin,
        before_names=before_names,
        after_names=before_names | frozenset({name}),
    )
    os.fsync(pin.directory_fd)
    _validate_pinned_directory(pin)
    published = _pinned_require_single_link_regular(pin, path, label="new durable artifact")
    if (published.st_dev, published.st_ino, published.st_size) != (
        written.st_dev,
        written.st_ino,
        written.st_size,
    ):
        raise RuntimeError("new durable artifact inode differs from writer")
    if _pinned_stable_read_bytes(pin, path, label="new durable artifact") != contents:
        raise RuntimeError("new durable artifact bytes differ from writer")


def _pinned_durable_replace(
    pin: _PinnedDirectory,
    path: Path,
    contents: bytes,
    *,
    mode: int = 0o600,
) -> None:
    target = _pinned_artifact_identity(pin, path)
    temporary = _atomic_temporary(path)
    temporary_name = _pinned_artifact_name(
        pin,
        temporary,
        label="durable replacement temporary",
    )
    target_name = _pinned_artifact_name(pin, path, label="durable replacement target")
    if _pinned_lexists(pin, temporary, label="durable replacement temporary"):
        raise FileExistsError(f"atomic temporary must be fresh: {temporary}")
    before_temporary_names = _raw_pinned_inventory(pin)
    if target_name not in before_temporary_names or temporary_name in before_temporary_names:
        raise PermissionError("durable replacement inventory changed before temporary create")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary_name, flags, mode, dir_fd=pin.directory_fd)
    try:
        written = _write_descriptor(descriptor, contents)
        _refresh_pinned_directory_after_owned_mutation(
            pin,
            before_names=before_temporary_names,
            after_names=before_temporary_names | frozenset({temporary_name}),
        )
        temporary_metadata = _pinned_require_single_link_regular(
            pin,
            temporary,
            label="durable replacement temporary",
        )
        if (temporary_metadata.st_dev, temporary_metadata.st_ino) != (
            written.st_dev,
            written.st_ino,
        ):
            raise RuntimeError("durable replacement temporary inode differs")
        if _pinned_artifact_identity(pin, path) != target:
            raise PermissionError("durable replacement target changed before replace")
        before_replace_names = _raw_pinned_inventory(pin)
        if before_replace_names != before_temporary_names | frozenset({temporary_name}):
            raise PermissionError("durable replacement inventory changed before replace")
        os.replace(
            temporary_name,
            target_name,
            src_dir_fd=pin.directory_fd,
            dst_dir_fd=pin.directory_fd,
        )
        _refresh_pinned_directory_after_owned_mutation(
            pin,
            before_names=before_replace_names,
            after_names=before_temporary_names,
        )
        os.fsync(pin.directory_fd)
        _validate_pinned_directory(pin)
        published = _pinned_require_single_link_regular(
            pin,
            path,
            label="replaced durable artifact",
        )
        if (published.st_dev, published.st_ino, published.st_size) != (
            written.st_dev,
            written.st_ino,
            written.st_size,
        ):
            raise RuntimeError("replaced durable artifact inode differs from writer")
        if _pinned_stable_read_bytes(pin, path, label="replaced durable artifact") != contents:
            raise RuntimeError("replaced durable artifact bytes differ from writer")
    except BaseException:
        # Preserve ambiguous bytes as terminal evidence.
        raise


def _report_bytes(report: Mapping[str, Any]) -> bytes:
    if type(report) is not dict:
        raise TypeError("report must be an exact dict")
    native = _json_native(report, label="report")
    if type(native) is not dict:
        raise TypeError("report must normalize to one exact JSON object")
    return (
        json.dumps(
            native,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _write_report_fresh(
    pin: _PinnedDirectory,
    path: Path,
    report: Mapping[str, Any],
) -> None:
    _pinned_durable_create(pin, path, _report_bytes(report))


def sha256_file(path: str | Path) -> str:
    return sha256_bytes(stable_read_bytes(path, label=f"SHA-256 source {path}"))


_LEDGER_LOCK = threading.RLock()
_CAPABILITY_REGISTRY: dict[int, tuple[object, ...]] = {}


@dataclass(frozen=True, slots=True)
class _LedgerRegistration:
    ledger: _AccessLedger
    directory_pin: _PinnedDirectory
    directory_binding_sha256: str
    directory_capability_sha256: str
    owner_thread: int
    stage: str
    path: Path
    bindings_bytes: bytes
    bindings_sha256: str
    record_sha256: str
    record_bytes: bytes
    artifact_identity: tuple[int, int, int, int, int]
    generation: int


_LEDGER_REGISTRY: dict[int, _LedgerRegistration] = {}


@dataclass(frozen=True, slots=True)
class _RunnerInvocationSeal:
    stage: Literal["development", "qualification"]
    context_sha256: str
    nonce: object


_RUNNER_INVOCATION_REGISTRY: dict[int, tuple[object, ...]] = {}


def _exact_code_object_equal(actual: types.CodeType, expected: types.CodeType) -> bool:
    return (
        actual == expected
        and actual.co_stacksize == expected.co_stacksize
        and actual.co_linetable == expected.co_linetable
    )


def _require_frozen_cli_caller(*, consume_depth: int) -> None:
    frame = inspect.currentframe()
    try:
        caller = frame
        for _ in range(consume_depth):
            if caller is None:
                raise PermissionError("runner invocation has no trusted caller frame")
            caller = caller.f_back
        expected = REPOSITORY_ROOT / PUBLICATION_SURFACE_PATHS["runner"]
        runner_contents = stable_read_bytes(expected, label="frozen CLI caller source")
        compiled = compile(
            runner_contents,
            os.fspath(expected) if caller is None else caller.f_code.co_filename,
            "exec",
        )
        expected_main = next(
            (
                constant
                for constant in compiled.co_consts
                if isinstance(constant, types.CodeType) and constant.co_name == "main"
            ),
            None,
        )
        if (
            caller is None
            or caller.f_code.co_name != "main"
            or Path(caller.f_code.co_filename).resolve() != expected.resolve()
            or caller.f_globals.get("__name__") != "__main__"
            or Path(caller.f_globals.get("__file__", "")).resolve() != expected.resolve()
            or not callable(caller.f_globals.get("main"))
            or caller.f_globals["main"].__code__ is not caller.f_code
            or expected_main is None
            or not _exact_code_object_equal(caller.f_code, expected_main)
            or not sys.argv
            or Path(sys.argv[0]).resolve() != expected.resolve()
        ):
            raise PermissionError("formal run requires the frozen CLI main boundary")
    finally:
        del frame


def _runner_invocation_context(
    *,
    stage: Literal["development", "qualification"],
    directory_pin: _PinnedDirectory,
    config: OrpheusConfig,
    config_path: Path,
    report_path: Path,
    checkpoint_path: Path,
    development_report_path: Path,
    source_provenance: Mapping[str, Any],
    reviewed_development: Mapping[str, str] | None,
) -> dict[str, Any]:
    expected_paths = {
        "config": _frozen_config_path(),
        "report": (
            canonical_development_report_path()
            if stage == "development"
            else canonical_qualification_report_path()
        ),
        "checkpoint": canonical_checkpoint_path(),
        "development_report": canonical_development_report_path(),
        "ledger": (
            development_ledger_path() if stage == "development" else qualification_ledger_path()
        ),
    }
    actual_paths = {
        "config": config_path,
        "report": report_path,
        "checkpoint": checkpoint_path,
        "development_report": development_report_path,
        "ledger": (
            development_ledger_path() if stage == "development" else qualification_ledger_path()
        ),
    }
    for name, expected in expected_paths.items():
        _require_canonical_path(
            actual_paths[name],
            expected,
            label=f"{stage} runner seal {name} path",
            directory_pin=(directory_pin if expected.parent == directory_pin.path else None),
        )
    _validate_pinned_directory(directory_pin)
    if not directory_pin.canonical or directory_pin.path != _canonical_run_directory():
        raise PermissionError("runner seal requires the canonical run-directory pin")
    assert_rgbd_variable_radius_config(config)
    source = _validated_published_source(
        dict(source_provenance),
        label=f"{stage} runner seal source",
    )
    runner_sha256 = source["publication_surface_sha256"]["runner"]
    if (
        sha256_bytes(
            stable_read_bytes(
                REPOSITORY_ROOT / PUBLICATION_SURFACE_PATHS["runner"],
                label="frozen CLI runner",
            )
        )
        != runner_sha256
    ):
        raise PermissionError("frozen CLI runner bytes differ before seal mint")
    _guard_frozen_inputs(
        config,
        config_path=config_path,
        published_source=source,
        label=f"{stage} runner invocation preflight",
    )
    _validate_distinct_canonical_paths(
        directory_pin,
        {
            "development_report": canonical_development_report_path(),
            "checkpoint": checkpoint_path,
            "development_ledger": development_ledger_path(),
            "qualification_report": canonical_qualification_report_path(),
            "qualification_ledger": qualification_ledger_path(),
        },
    )
    if stage == "development":
        if reviewed_development is not None:
            raise ValueError("development runner seal cannot bind reviewed hashes")
        reviewed: dict[str, str] | None = None
        _validate_run_tree(
            directory_pin,
            frozenset(),
            stage="development CLI seal preflight",
        )
        run_tree_names: list[str] = []
        reviewed_identities: dict[str, list[int]] | None = None
    else:
        if type(reviewed_development) is not dict or set(reviewed_development) != {
            "checkpoint_sha256",
            "report_sha256",
            "ledger_sha256",
        }:
            raise ValueError("qualification runner seal requires three reviewed hashes")
        reviewed = {
            name: validated_sha256(value, label=f"runner seal reviewed {name}")
            for name, value in reviewed_development.items()
        }
        _, checked_reviewed, _, pin = _review_development_bundle(
            directory_pin=directory_pin,
            reviewed_checkpoint_sha256=reviewed["checkpoint_sha256"],
            reviewed_report_sha256=reviewed["report_sha256"],
            reviewed_development_ledger_sha256=reviewed["ledger_sha256"],
            expected_source=source,
        )
        _exact_equal(checked_reviewed, reviewed, label="CLI seal reviewed development")
        if _pinned_lexists(
            directory_pin,
            report_path,
            label="qualification report preflight",
        ) or _pinned_lexists(
            directory_pin,
            qualification_ledger_path(),
            label="qualification ledger preflight",
        ):
            raise FileExistsError("qualification report and ledger must both be fresh")
        run_tree_names = sorted(DEVELOPMENT_ARTIFACT_NAMES)
        reviewed_identities = {
            "checkpoint": list(pin.checkpoint_identity),
            "report": list(pin.report_identity),
            "ledger": list(pin.ledger_identity),
        }
    ledger_bindings = _ledger_bindings(
        stage=stage,
        directory_pin=directory_pin,
        source_provenance=source,
        reviewed_development=reviewed,
    )
    return {
        "stage": stage,
        "owner_thread": threading.get_ident(),
        "runner_path": PUBLICATION_SURFACE_PATHS["runner"],
        "runner_sha256": runner_sha256,
        "paths": {name: os.fspath(path) for name, path in actual_paths.items()},
        "resolved_config_sha256": FROZEN_CONFIG_SHA256,
        "resolved_config_value_sha256": canonical_sha256(config.to_dict()),
        "source_provenance_sha256": canonical_sha256(source),
        "scene_certificate": _frozen_scene_certificate_binding(),
        "reviewed_development": reviewed,
        "reviewed_artifact_identities": reviewed_identities,
        "run_tree_names": run_tree_names,
        "ledger_bindings_sha256": canonical_sha256(ledger_bindings),
        "run_directory": _pinned_directory_binding(directory_pin),
        "run_directory_capability_sha256": _pinned_directory_capability_sha256(directory_pin),
        "preflight_complete": True,
    }


def _mint_runner_invocation_seal(
    *,
    stage: Literal["development", "qualification"],
    config: OrpheusConfig,
    config_path: Path,
    report_path: Path,
    checkpoint_path: Path,
    development_report_path: Path,
    source_provenance: Mapping[str, Any],
    reviewed_development: Mapping[str, str] | None,
) -> _RunnerInvocationSeal:
    _require_frozen_cli_caller(consume_depth=2)
    if any(registration[1] == stage for registration in _RUNNER_INVOCATION_REGISTRY.values()):
        raise PermissionError("frozen CLI invocation seal cannot mint twice")
    directory_pin = _acquire_pinned_directory(
        _canonical_run_directory(),
        create=stage == "development",
        canonical=True,
    )
    try:
        context = _runner_invocation_context(
            stage=stage,
            directory_pin=directory_pin,
            config=config,
            config_path=config_path,
            report_path=report_path,
            checkpoint_path=checkpoint_path,
            development_report_path=development_report_path,
            source_provenance=source_provenance,
            reviewed_development=reviewed_development,
        )
        context_sha256 = canonical_sha256(context)
        seal = _RunnerInvocationSeal(stage=stage, context_sha256=context_sha256, nonce=object())
        _RUNNER_INVOCATION_REGISTRY[id(seal)] = (
            seal,
            stage,
            context_sha256,
            threading.get_ident(),
            "issued",
            directory_pin,
        )
        return seal
    except BaseException:
        _release_pinned_directory(directory_pin)
        raise


def _consume_runner_invocation_seal(
    seal: _RunnerInvocationSeal,
    *,
    context: Mapping[str, Any],
) -> None:
    _require_frozen_cli_caller(consume_depth=3)
    registration = _RUNNER_INVOCATION_REGISTRY.get(id(seal))
    context_sha256 = canonical_sha256(context)
    if (
        not isinstance(seal, _RunnerInvocationSeal)
        or registration is None
        or registration[0] is not seal
        or registration[1] != seal.stage
        or registration[2] != context_sha256
        or registration[3] != threading.get_ident()
        or registration[4] != "issued"
        or not isinstance(registration[5], _PinnedDirectory)
        or seal.context_sha256 != context_sha256
    ):
        raise PermissionError("runner invocation seal is forged, replayed, or rebound")
    _validate_pinned_directory(registration[5])
    _RUNNER_INVOCATION_REGISTRY[id(seal)] = (
        *registration[:4],
        "consumed",
        registration[5],
    )


def _runner_invocation_pin(
    seal: _RunnerInvocationSeal,
    *,
    statuses: frozenset[str],
) -> _PinnedDirectory:
    registration = _RUNNER_INVOCATION_REGISTRY.get(id(seal))
    if (
        not isinstance(seal, _RunnerInvocationSeal)
        or registration is None
        or len(registration) < 6
        or registration[0] is not seal
        or registration[1] != seal.stage
        or registration[2] != seal.context_sha256
        or registration[3] != threading.get_ident()
        or registration[4] not in statuses
        or not isinstance(registration[5], _PinnedDirectory)
    ):
        raise PermissionError("runner invocation lost its pinned directory owner")
    pin = registration[5]
    _validate_pinned_directory(pin)
    return pin


def _release_runner_invocation_seal(seal: _RunnerInvocationSeal) -> None:
    """Close the invocation-owned directory capability after final validation."""

    _require_frozen_cli_caller(consume_depth=2)
    registration = _RUNNER_INVOCATION_REGISTRY.pop(id(seal), None)
    if (
        not isinstance(seal, _RunnerInvocationSeal)
        or registration is None
        or len(registration) < 6
        or registration[0] is not seal
        or registration[1] != seal.stage
        or registration[2] != seal.context_sha256
        or registration[3] != threading.get_ident()
        or registration[4] not in {"issued", "consumed", "authorization_owned", "terminal"}
        or not isinstance(registration[5], _PinnedDirectory)
    ):
        raise PermissionError("runner invocation seal cannot be released")
    _release_pinned_directory(registration[5])


@dataclass(frozen=True, slots=True)
class _RunAuthorization:
    stage: Literal["development", "qualification"]
    ledger_path: str
    ledger_bindings_sha256: str
    context_sha256: str
    run_directory_capability_sha256: str
    nonce: object


def _build_run_authorization_vault() -> tuple[
    Callable[[_RunAuthorization, _RunnerInvocationSeal], None],
    Callable[[_RunAuthorization, object], _RunnerInvocationSeal],
    Callable[[_RunAuthorization, object], bool],
    Callable[[_RunAuthorization, object], _RunnerInvocationSeal],
    Callable[[], None],
]:
    records: dict[int, tuple[object, ...]] = {}

    def issue(token: _RunAuthorization, seal: _RunnerInvocationSeal) -> None:
        caller = inspect.currentframe().f_back
        if (
            caller is None
            or caller.f_code is not _mint_run_authorization.__code__
            or id(token) in records
        ):
            raise PermissionError("run authorization vault issue is not mint-owned")
        records[id(token)] = (
            token,
            seal,
            threading.get_ident(),
            "issued",
            None,
        )

    def claim(token: _RunAuthorization, ledger: object) -> _RunnerInvocationSeal:
        caller = inspect.currentframe().f_back
        record = records.get(id(token))
        if (
            caller is None
            or caller.f_code is not _consume_run_authorization.__code__
            or record is None
            or record[0] is not token
            or not isinstance(record[1], _RunnerInvocationSeal)
            or record[2] != threading.get_ident()
            or record[3] != "issued"
            or record[4] is not None
        ):
            raise PermissionError("run authorization vault claim is forged or replayed")
        records[id(token)] = (*record[:3], "ledger_owned", ledger)
        return record[1]

    def verify_owned(token: _RunAuthorization, ledger: object) -> bool:
        caller = inspect.currentframe().f_back
        record = records.get(id(token))
        return bool(
            caller is not None
            and caller.f_code is _AccessLedger._assert_formal_access.__code__
            and record is not None
            and record[0] is token
            and isinstance(record[1], _RunnerInvocationSeal)
            and record[2] == threading.get_ident()
            and record[3] == "ledger_owned"
            and record[4] is ledger
        )

    def retire(token: _RunAuthorization, ledger: object) -> _RunnerInvocationSeal:
        caller = inspect.currentframe().f_back
        record = records.pop(id(token), None)
        if (
            caller is None
            or caller.f_code is not _AccessLedger._replace.__code__
            or record is None
            or record[0] is not token
            or not isinstance(record[1], _RunnerInvocationSeal)
            or record[2] != threading.get_ident()
            or record[3] != "ledger_owned"
            or record[4] is not ledger
        ):
            raise PermissionError("terminal ledger lost its private authorization vault record")
        return record[1]

    def clear() -> None:
        records.clear()

    return issue, claim, verify_owned, retire, clear


(
    _issue_run_authorization,
    _claim_run_authorization,
    _verify_owned_run_authorization,
    _retire_run_authorization,
    _clear_run_authorization_vault_for_tests,
) = _build_run_authorization_vault()


def _run_authorization_context(
    *,
    stage: Literal["development", "qualification"],
    config: OrpheusConfig,
    config_path: Path,
    report_path: Path,
    checkpoint_path: Path,
    development_report_path: Path,
    ledger_path: Path,
    ledger_bindings: Mapping[str, Any],
    source_provenance: Mapping[str, Any],
    reviewed_development: Mapping[str, str] | None,
    invocation_seal: _RunnerInvocationSeal,
) -> dict[str, Any]:
    if stage not in {"development", "qualification"}:
        raise ValueError("run authorization context stage differs")
    expected_paths = {
        "config": _frozen_config_path(),
        "report": (
            canonical_development_report_path()
            if stage == "development"
            else canonical_qualification_report_path()
        ),
        "checkpoint": canonical_checkpoint_path(),
        "development_report": canonical_development_report_path(),
        "ledger": (
            development_ledger_path() if stage == "development" else qualification_ledger_path()
        ),
    }
    actual_paths = {
        "config": config_path,
        "report": report_path,
        "checkpoint": checkpoint_path,
        "development_report": development_report_path,
        "ledger": ledger_path,
    }
    seal_registration = _RUNNER_INVOCATION_REGISTRY.get(id(invocation_seal))
    if (
        not isinstance(invocation_seal, _RunnerInvocationSeal)
        or seal_registration is None
        or seal_registration[0] is not invocation_seal
        or seal_registration[1] != stage
        or seal_registration[2] != invocation_seal.context_sha256
        or seal_registration[3] != threading.get_ident()
        or seal_registration[4] != "consumed"
        or len(seal_registration) != 6
        or not isinstance(seal_registration[5], _PinnedDirectory)
    ):
        raise PermissionError("authorization context requires one consumed CLI seal")
    directory_pin = seal_registration[5]
    _validate_pinned_directory(directory_pin)
    for name, expected in expected_paths.items():
        _require_canonical_path(
            actual_paths[name],
            expected,
            label=f"{stage} authorization {name} path",
            directory_pin=(directory_pin if expected.parent == directory_pin.path else None),
        )
    assert_rgbd_variable_radius_config(config)
    source = _validated_published_source(
        dict(source_provenance),
        label=f"{stage} authorization source",
    )
    if type(ledger_bindings) is not dict:
        raise TypeError("run authorization ledger bindings must be an exact dict")
    preflight_context = _runner_invocation_context(
        stage=stage,
        directory_pin=directory_pin,
        config=config,
        config_path=config_path,
        report_path=report_path,
        checkpoint_path=checkpoint_path,
        development_report_path=development_report_path,
        source_provenance=source,
        reviewed_development=reviewed_development,
    )
    if canonical_sha256(preflight_context) != invocation_seal.context_sha256 or preflight_context[
        "ledger_bindings_sha256"
    ] != canonical_sha256(ledger_bindings):
        raise PermissionError("authorization context differs from its full CLI preflight")
    if stage == "development":
        if reviewed_development is not None:
            raise ValueError("development authorization cannot bind reviewed hashes")
        reviewed: dict[str, str] | None = None
    else:
        if type(reviewed_development) is not dict or set(reviewed_development) != {
            "checkpoint_sha256",
            "report_sha256",
            "ledger_sha256",
        }:
            raise ValueError("qualification authorization requires three reviewed hashes")
        reviewed = {
            name: validated_sha256(value, label=f"authorization reviewed {name}")
            for name, value in reviewed_development.items()
        }
    return {
        "stage": stage,
        "owner_thread": threading.get_ident(),
        "paths": {name: os.fspath(path) for name, path in actual_paths.items()},
        "resolved_config_sha256": FROZEN_CONFIG_SHA256,
        "resolved_config_value_sha256": canonical_sha256(config.to_dict()),
        "scene_certificate": _frozen_scene_certificate_binding(),
        "source_provenance_sha256": canonical_sha256(source),
        "ledger_bindings_sha256": canonical_sha256(ledger_bindings),
        "runner_invocation_seal_sha256": invocation_seal.context_sha256,
        "run_directory": _pinned_directory_binding(directory_pin),
        "run_directory_capability_sha256": _pinned_directory_capability_sha256(directory_pin),
        "reviewed_development": reviewed,
    }


def _mint_run_authorization(
    *,
    invocation_seal: _RunnerInvocationSeal,
    context: Mapping[str, Any],
) -> _RunAuthorization:
    _require_frozen_cli_caller(consume_depth=3)
    if not isinstance(invocation_seal, _RunnerInvocationSeal) or type(context) is not dict:
        raise PermissionError("run authorization requires one consumed frozen CLI seal")
    registration = _RUNNER_INVOCATION_REGISTRY.get(id(invocation_seal))
    context_sha256 = canonical_sha256(context)
    if (
        registration is None
        or registration[0] is not invocation_seal
        or registration[1] != invocation_seal.stage
        or registration[2] != invocation_seal.context_sha256
        or registration[3] != threading.get_ident()
        or registration[4] != "consumed"
        or len(registration) != 6
        or not isinstance(registration[5], _PinnedDirectory)
        or context.get("runner_invocation_seal_sha256") != invocation_seal.context_sha256
        or context.get("stage") != invocation_seal.stage
        or context.get("run_directory_capability_sha256")
        != _pinned_directory_capability_sha256(registration[5])
    ):
        raise PermissionError("runner invocation seal is forged, replayed, or not preflight-owned")
    stage = invocation_seal.stage
    ledger_path = context["paths"]["ledger"]
    ledger_bindings_sha256 = context["ledger_bindings_sha256"]
    token = _RunAuthorization(
        stage=stage,
        ledger_path=ledger_path,
        ledger_bindings_sha256=ledger_bindings_sha256,
        context_sha256=context_sha256,
        run_directory_capability_sha256=context["run_directory_capability_sha256"],
        nonce=object(),
    )
    _issue_run_authorization(token, invocation_seal)
    _RUNNER_INVOCATION_REGISTRY[id(invocation_seal)] = (
        *registration[:4],
        "authorization_owned",
        registration[5],
        token,
    )
    return token


def _consume_run_authorization(
    token: _RunAuthorization,
    *,
    stage: Literal["development", "qualification"],
    ledger_path: Path,
    bindings: Mapping[str, Any],
    directory_pin: _PinnedDirectory,
    ledger: object,
) -> None:
    caller = inspect.currentframe().f_back
    if (
        not isinstance(token, _RunAuthorization)
        or caller is None
        or caller.f_code is not _AccessLedger.__init__.__code__
        or caller.f_locals.get("self") is not ledger
        or token.stage != stage
        or token.ledger_path != os.fspath(ledger_path)
        or token.ledger_bindings_sha256 != canonical_sha256(bindings)
        or token.run_directory_capability_sha256
        != _pinned_directory_capability_sha256(directory_pin)
    ):
        raise PermissionError("run authorization is forged, replayed, or rebound")
    seal = _claim_run_authorization(token, ledger)
    seal_registration = _RUNNER_INVOCATION_REGISTRY.get(id(seal))
    if seal_registration != (
        seal,
        stage,
        seal.context_sha256,
        threading.get_ident(),
        "authorization_owned",
        directory_pin,
        token,
    ):
        raise PermissionError("run authorization lost its frozen CLI seal binding")


@dataclass(frozen=True, slots=True)
class _OrdinalCapability:
    split: str
    ordinal: int
    nonce: object


@dataclass(frozen=True, slots=True)
class _BatchCapability:
    split: str
    ordinals: tuple[int, int, int, int]
    tokens: tuple[_OrdinalCapability, ...]
    nonce: object


@dataclass(frozen=True, slots=True)
class _BatchRegistration:
    batch: _BatchCapability
    manifest: _ManifestCapability
    ledger: _AccessLedger
    split: str
    ordinals: tuple[int, int, int, int]
    batch_sha256: str
    token_sha256: tuple[str, ...]
    consumed_ordinals: frozenset[int]
    ledger_generation: int
    ledger_record_sha256: str
    status: Literal["issued", "complete"]


_BATCH_REGISTRY: dict[int, _BatchRegistration] = {}


@dataclass(frozen=True, slots=True)
class _BatchCommit:
    split: str
    ordinals: tuple[int, int, int, int]
    result_sha256: str
    nonce: object


@dataclass(frozen=True, slots=True)
class _BatchCommitRegistration:
    commit: _BatchCommit
    ledger: _AccessLedger
    manifest: _ManifestCapability
    batch: _BatchCapability
    batch_registration: _BatchRegistration
    result_sha256: str
    owner_thread: int
    ledger_generation: int
    ledger_record_sha256: str
    status: Literal["issued"]


_BATCH_COMMIT_REGISTRY: dict[int, _BatchCommitRegistration] = {}


class _AccessLedger:
    """Durable exact-once split/batch receipt written before scene access."""

    def __init__(
        self,
        path: Path,
        *,
        stage: Literal["development", "qualification"],
        bindings: Mapping[str, Any],
        directory_pin: _PinnedDirectory,
        authorization: _RunAuthorization | None = None,
    ) -> None:
        if stage not in {"development", "qualification"}:
            raise ValueError("unknown ledger stage")
        self.path = path
        self.stage = stage
        self._directory_pin = directory_pin
        self._directory_binding_sha256 = canonical_sha256(_pinned_directory_binding(directory_pin))
        self._directory_capability_sha256 = _pinned_directory_capability_sha256(directory_pin)
        _pinned_artifact_name(directory_pin, path, label=f"{stage} ledger path")
        self._last_bytes = b""
        self._last_identity = (0, 0, 0, 0, 0)
        self._owner_thread = threading.get_ident()
        self._terminal = False
        self._authorization = authorization
        self._formal_authorized = authorization is not None
        self._bindings_bytes = _canonical_json(copy.deepcopy(dict(bindings)))
        self._bindings_sha256 = canonical_sha256(bindings)
        if authorization is not None:
            _consume_run_authorization(
                authorization,
                stage=stage,
                ledger_path=path,
                bindings=bindings,
                directory_pin=directory_pin,
                ledger=self,
            )
        order = (
            ["development"]
            if stage == "development"
            else [
                "selector",
                "confirmation",
                "final_test",
            ]
        )
        self._record: dict[str, Any] = {
            "artifact_kind": "rgbd_variable_radius_exactly_once_access_ledger",
            "architecture_attempt": ARCHITECTURE_ATTEMPT,
            "stage": stage,
            "order": order,
            "bindings": copy.deepcopy(dict(bindings)),
            "batch_size": BATCH_SIZE,
            "scenes_per_split": SCENES_PER_SPLIT,
            "splits": {
                split: {
                    "status": "unopened",
                    "access_started": False,
                    "next_ordinal": 0,
                    "active_batch": None,
                    "batch_result_sha256": [],
                    "split_result_sha256": None,
                }
                for split in order
            },
            "attempt_reserved": True,
            "status": "reserved_before_access",
            "generation": 0,
        }
        contents = self._serialized()
        _pinned_durable_create(self._directory_pin, self.path, contents)
        self._last_bytes = contents
        self._last_identity = _pinned_artifact_identity(self._directory_pin, self.path)
        if id(self) in _LEDGER_REGISTRY:
            raise RuntimeError("ledger identity was already registered")
        _LEDGER_REGISTRY[id(self)] = _LedgerRegistration(
            ledger=self,
            directory_pin=self._directory_pin,
            directory_binding_sha256=self._directory_binding_sha256,
            directory_capability_sha256=self._directory_capability_sha256,
            owner_thread=self._owner_thread,
            stage=self.stage,
            path=self.path,
            bindings_bytes=self._bindings_bytes,
            bindings_sha256=self._bindings_sha256,
            record_sha256=sha256_bytes(contents),
            record_bytes=contents,
            artifact_identity=self._last_identity,
            generation=0,
        )
        self._verify_disk()

    @property
    def record(self) -> Mapping[str, Any]:
        """Return a recursively immutable view copy of current durable state."""

        def freeze(value: Any) -> Any:
            if type(value) is dict:
                return MappingProxyType({key: freeze(item) for key, item in value.items()})
            if type(value) is list:
                return tuple(freeze(item) for item in value)
            return copy.deepcopy(value)

        return freeze(self._record)

    def _serialized(self, record: Mapping[str, Any] | None = None) -> bytes:
        source = self._record if record is None else record
        if type(source) is not dict:
            raise TypeError("ledger record must be an exact dict")
        payload = copy.deepcopy(source)
        payload["record_sha256"] = canonical_sha256(payload)
        return _report_bytes(payload)

    def _verify_disk(self) -> None:
        self._assert_live()
        registration = _LEDGER_REGISTRY.get(id(self))
        current_bytes = self._serialized()
        if (
            not isinstance(registration, _LedgerRegistration)
            or registration.ledger is not self
            or registration.directory_pin is not self._directory_pin
            or registration.directory_binding_sha256 != self._directory_binding_sha256
            or registration.directory_capability_sha256 != self._directory_capability_sha256
            or registration.owner_thread != self._owner_thread
            or registration.stage != self.stage
            or registration.path != self.path
            or registration.bindings_bytes != self._bindings_bytes
            or registration.bindings_sha256 != self._bindings_sha256
            or _canonical_json(self._record.get("bindings")) != self._bindings_bytes
            or registration.record_bytes != self._last_bytes
            or registration.record_bytes != current_bytes
            or registration.record_sha256 != sha256_bytes(current_bytes)
            or registration.artifact_identity != self._last_identity
            or type(self._record.get("generation")) is not int
            or registration.generation != self._record["generation"]
        ):
            raise PermissionError(f"{self.stage} ledger in-memory registry state changed")
        _validate_pinned_directory(self._directory_pin)
        identity = _pinned_artifact_identity(self._directory_pin, self.path)
        if identity != self._last_identity:
            raise PermissionError(f"{self.stage} ledger inode metadata changed")
        contents = _pinned_stable_read_bytes(
            self._directory_pin,
            self.path,
            label=f"{self.stage} live ledger",
        )
        if contents != self._last_bytes:
            raise PermissionError(f"{self.stage} ledger bytes changed outside live owner")
        if sha256_bytes(contents) != sha256_bytes(self._last_bytes):
            raise PermissionError(f"{self.stage} ledger digest changed outside live owner")
        _pinned_require_single_link_regular(
            self._directory_pin,
            self.path,
            label=f"{self.stage} live ledger",
        )

    def _assert_live(self) -> None:
        if self._terminal:
            raise PermissionError(f"{self.stage} ledger is already terminal")
        if threading.get_ident() != self._owner_thread:
            raise PermissionError(f"{self.stage} ledger is owned by another thread")
        registration = _LEDGER_REGISTRY.get(id(self))
        if not isinstance(registration, _LedgerRegistration) or registration.ledger is not self:
            raise PermissionError(f"{self.stage} ledger is not the registered live owner")

    def _replace(self, candidate: dict[str, Any], *, terminal: bool = False) -> None:
        self._verify_disk()
        if type(candidate) is not dict:
            raise TypeError("ledger transition candidate must be an exact dict")
        expected_generation = self._record["generation"] + 1
        if candidate.get("generation") != self._record["generation"]:
            raise PermissionError("ledger transition candidate generation was pre-mutated")
        candidate = copy.deepcopy(candidate)
        candidate["generation"] = expected_generation
        contents = self._serialized(candidate)
        _pinned_durable_replace(self._directory_pin, self.path, contents)
        self._record = candidate
        self._last_bytes = contents
        self._last_identity = _pinned_artifact_identity(self._directory_pin, self.path)
        _LEDGER_REGISTRY[id(self)] = _LedgerRegistration(
            ledger=self,
            directory_pin=self._directory_pin,
            directory_binding_sha256=self._directory_binding_sha256,
            directory_capability_sha256=self._directory_capability_sha256,
            owner_thread=self._owner_thread,
            stage=self.stage,
            path=self.path,
            bindings_bytes=self._bindings_bytes,
            bindings_sha256=self._bindings_sha256,
            record_sha256=sha256_bytes(contents),
            record_bytes=contents,
            artifact_identity=self._last_identity,
            generation=expected_generation,
        )
        self._verify_disk()
        if terminal:
            self._terminal = True
            terminal_registration = _LEDGER_REGISTRY.pop(id(self), None)
            if (
                not isinstance(terminal_registration, _LedgerRegistration)
                or terminal_registration.ledger is not self
            ):
                raise PermissionError("terminal ledger lost its trusted live registration")
            if self._authorization is not None:
                seal = _retire_run_authorization(self._authorization, self)
                seal_registration = _RUNNER_INVOCATION_REGISTRY.get(id(seal))
                if not isinstance(seal, _RunnerInvocationSeal) or seal_registration != (
                    seal,
                    self.stage,
                    seal.context_sha256,
                    self._owner_thread,
                    "authorization_owned",
                    self._directory_pin,
                    self._authorization,
                ):
                    raise PermissionError("terminal ledger lost its frozen CLI seal owner")
                _RUNNER_INVOCATION_REGISTRY[id(seal)] = (
                    *seal_registration[:4],
                    "terminal",
                    self._directory_pin,
                    self._authorization,
                )

    def _assert_formal_access(self) -> None:
        self._verify_disk()
        if not self._formal_authorized or self._authorization is None:
            raise PermissionError("formal scene access requires a runner-minted authorization")
        if not _verify_owned_run_authorization(self._authorization, self):
            raise PermissionError("formal scene run authorization is not live and ledger-owned")

    def begin_split(self, split: str) -> None:
        with _LEDGER_LOCK:
            self._verify_disk()
            current = self._record
            if split not in current["order"]:
                raise ValueError(f"split {split!r} is not owned by {self.stage} ledger")
            index = current["order"].index(split)
            expected_overall = (
                "reserved_before_access" if index == 0 else f"{current['order'][index - 1]}_passed"
            )
            if current["status"] != expected_overall:
                raise RuntimeError("split open does not follow the exact ledger transition")
            for predecessor in current["order"][:index]:
                if current["splits"][predecessor]["status"] != "passed":
                    raise RuntimeError(f"{split} must remain unopened until {predecessor} passes")
            state = current["splits"][split]
            if state["status"] != "unopened" or state["access_started"] is not False:
                raise RuntimeError(f"split {split!r} cannot be opened twice")
            candidate = copy.deepcopy(current)
            state = candidate["splits"][split]
            state["access_started"] = True
            state["status"] = "access_started"
            candidate["status"] = f"{split}_access_started"
            self._replace(candidate)

    def begin_batch(self, split: str, ordinals: tuple[int, int, int, int]) -> None:
        with _LEDGER_LOCK:
            self._verify_disk()
            current = self._record
            state = current["splits"][split]
            if state["status"] not in {"access_started", "evaluating"}:
                raise RuntimeError("batch access requires a live split")
            expected = tuple(range(state["next_ordinal"], state["next_ordinal"] + BATCH_SIZE))
            if (
                type(ordinals) is not tuple
                or len(ordinals) != BATCH_SIZE
                or any(type(value) is not int for value in ordinals)
                or ordinals != expected
                or ordinals[-1] >= SCENES_PER_SPLIT
            ):
                raise RuntimeError(
                    "batch ordinals are reordered, repeated, partial, or out of range"
                )
            batch_index = ordinals[0] // BATCH_SIZE
            expected_overall = (
                f"{split}_access_started"
                if batch_index == 0
                else f"{split}_batch_{batch_index - 1}_complete"
            )
            if current["status"] != expected_overall:
                raise RuntimeError("batch reserve does not follow the exact ledger transition")
            if state["active_batch"] is not None:
                raise RuntimeError("a prior batch remains active")
            candidate = copy.deepcopy(current)
            state = candidate["splits"][split]
            state["active_batch"] = list(ordinals)
            state["status"] = "evaluating"
            candidate["status"] = f"{split}_batch_{ordinals[0] // BATCH_SIZE}_reserved"
            self._replace(candidate)

    def complete_batch(
        self,
        split: str,
        ordinals: tuple[int, int, int, int],
        *,
        result_sha256: str,
        commit: _BatchCommit,
    ) -> None:
        with _LEDGER_LOCK:
            validated_sha256(result_sha256, label="batch result")
            self._verify_disk()
            commit_registration = _BATCH_COMMIT_REGISTRY.get(id(commit))
            frame = inspect.currentframe()
            caller = None if frame is None else frame.f_back
            if (
                not isinstance(commit, _BatchCommit)
                or not isinstance(commit_registration, _BatchCommitRegistration)
                or commit_registration.commit is not commit
                or commit_registration.ledger is not self
                or caller is None
                or caller.f_code is not _ManifestCapability.complete_batch.__code__
                or caller.f_locals.get("self") is not commit_registration.manifest
                or caller.f_locals.get("batch") is not commit_registration.batch
                or caller.f_locals.get("batch_registration")
                is not commit_registration.batch_registration
                or commit_registration.batch_registration.consumed_ordinals != frozenset(ordinals)
                or commit_registration.result_sha256 != result_sha256
                or commit_registration.owner_thread != threading.get_ident()
                or commit_registration.ledger_generation != self._record["generation"]
                or commit_registration.ledger_record_sha256 != sha256_bytes(self._last_bytes)
                or commit_registration.status != "issued"
                or commit.split != split
                or commit.ordinals != ordinals
                or commit.result_sha256 != result_sha256
            ):
                raise PermissionError("durable batch completion requires its one-shot owner commit")
            del frame, caller
            consumed_commit = _BATCH_COMMIT_REGISTRY.pop(id(commit), None)
            if consumed_commit is not commit_registration:
                raise PermissionError("durable batch owner commit changed before consumption")
            current = self._record
            state = current["splits"][split]
            batch_index = ordinals[0] // BATCH_SIZE
            if current["status"] != f"{split}_batch_{batch_index}_reserved":
                raise RuntimeError("batch complete does not follow the exact ledger transition")
            if state["active_batch"] != list(ordinals):
                raise RuntimeError("completed batch differs from durable active receipt")
            candidate = copy.deepcopy(current)
            state = candidate["splits"][split]
            state["active_batch"] = None
            state["next_ordinal"] = ordinals[-1] + 1
            state["batch_result_sha256"].append(result_sha256)
            candidate["status"] = f"{split}_batch_{ordinals[0] // BATCH_SIZE}_complete"
            self._replace(candidate)

    def complete_split(self, split: str, result: Mapping[str, Any]) -> None:
        with _LEDGER_LOCK:
            self._verify_disk()
            current = self._record
            state = current["splits"][split]
            if current["status"] != f"{split}_batch_{SCENES_PER_SPLIT // BATCH_SIZE - 1}_complete":
                raise RuntimeError("split complete does not follow the exact ledger transition")
            if (
                state["next_ordinal"] != SCENES_PER_SPLIT
                or state["active_batch"] is not None
                or len(state["batch_result_sha256"]) != SCENES_PER_SPLIT // BATCH_SIZE
            ):
                raise RuntimeError("split cannot complete before all exact batches")
            if type(result) is not dict or type(result.get("passed")) is not bool:
                raise TypeError("split result must be an exact dict with exact bool passed")
            result_hash = canonical_sha256(result)
            candidate = copy.deepcopy(current)
            state = candidate["splits"][split]
            state["split_result_sha256"] = result_hash
            state["status"] = "passed" if result["passed"] else "failed"
            candidate["status"] = f"{split}_{state['status']}"
            self._replace(candidate)

    def fail(
        self,
        *,
        error_type: str,
        error_message: str,
        report_sha256: str,
    ) -> None:
        with _LEDGER_LOCK:
            report_sha256 = validated_sha256(
                report_sha256,
                label=f"{self.stage} intended error report",
            )
            self._verify_disk()
            current = self._record
            if "error" in current:
                raise RuntimeError("ledger error transition cannot be repeated")
            if type(error_type) is not str or type(error_message) is not str:
                raise TypeError("ledger error fields must be exact strings")
            candidate = copy.deepcopy(current)
            candidate["status"] = "terminal_error"
            candidate["error"] = {
                "type": str(error_type),
                "message": str(error_message),
                "report_sha256": report_sha256,
            }
            self._replace(candidate, terminal=True)

    def finish(self) -> str:
        with _LEDGER_LOCK:
            self._verify_disk()
            current = self._record
            expected = (
                {"development"}
                if self.stage == "development"
                else {"selector", "confirmation", "final_test"}
            )
            if set(current["splits"]) != expected:
                raise RuntimeError("ledger split inventory changed")
            statuses = {value["status"] for value in current["splits"].values()}
            if statuses == {"passed"}:
                expected_status = f"{current['order'][-1]}_passed"
                terminal_status = "complete_passed"
            elif "failed" in statuses:
                failed_split = next(
                    split
                    for split in current["order"]
                    if current["splits"][split]["status"] == "failed"
                )
                expected_status = f"{failed_split}_failed"
                terminal_status = "complete_failed"
            else:
                raise RuntimeError("ledger cannot finish with nonterminal split states")
            if current["status"] != expected_status:
                raise RuntimeError("ledger finish does not follow the exact terminal transition")
            candidate = copy.deepcopy(current)
            candidate["status"] = terminal_status
            self._replace(candidate, terminal=True)
            return sha256_bytes(self._last_bytes)


class _ManifestCapability:
    """In-memory owner for one exact durable manifest pass."""

    def __init__(self, *, split: str, ledger: _AccessLedger) -> None:
        if type(split) is not str or split not in SPLITS:
            raise ValueError("manifest split is invalid")
        self.split = split
        self._ledger = ledger
        self._next_ordinal = 0
        self._active: _BatchCapability | None = None
        self._consumed_tokens: set[int] = set()
        self._closed = False
        ledger.begin_split(split)

    def _validated_batch_registration(
        self,
        batch: _BatchCapability,
    ) -> _BatchRegistration:
        self._ledger._verify_disk()
        registration = _BATCH_REGISTRY.get(id(batch))
        capability = _CAPABILITY_REGISTRY.get(id(batch))
        token_sha256 = tuple(_ordinal_capability_sha256(token) for token in batch.tokens)
        token_registrations = tuple(_CAPABILITY_REGISTRY.get(id(token)) for token in batch.tokens)
        current = self._ledger._record
        state = current["splits"][self.split]
        if (
            not isinstance(registration, _BatchRegistration)
            or registration.batch is not batch
            or registration.manifest is not self
            or registration.ledger is not self._ledger
            or registration.split != self.split
            or registration.ordinals != batch.ordinals
            or registration.batch_sha256 != _batch_capability_sha256(batch)
            or registration.token_sha256 != token_sha256
            or registration.ledger_generation != current["generation"]
            or registration.ledger_record_sha256 != sha256_bytes(self._ledger._last_bytes)
            or registration.status != "issued"
            or capability != (batch, self, "issued", _batch_capability_sha256(batch))
            or self._active is not batch
            or current["status"] != f"{self.split}_batch_{batch.ordinals[0] // BATCH_SIZE}_reserved"
            or state["status"] != "evaluating"
            or state["active_batch"] != list(batch.ordinals)
            or state["next_ordinal"] != self._next_ordinal
            or any(
                token_registration
                != (
                    token,
                    self,
                    batch,
                    ("consumed" if token.ordinal in registration.consumed_ordinals else "issued"),
                    token_digest,
                )
                for token, token_digest, token_registration in zip(
                    batch.tokens,
                    token_sha256,
                    token_registrations,
                    strict=True,
                )
            )
            or any(
                (id(token) in self._consumed_tokens)
                is not (token.ordinal in registration.consumed_ordinals)
                for token in batch.tokens
            )
        ):
            raise PermissionError("batch capability registry binding differs")
        return registration

    def begin_batch(self, ordinals: tuple[int, int, int, int]) -> _BatchCapability:
        with _LEDGER_LOCK:
            if self._closed or self._active is not None:
                raise PermissionError("manifest capability is closed or already active")
            expected = tuple(range(self._next_ordinal, self._next_ordinal + BATCH_SIZE))
            if (
                type(ordinals) is not tuple
                or len(ordinals) != BATCH_SIZE
                or any(type(value) is not int for value in ordinals)
                or ordinals != expected
            ):
                raise PermissionError("manifest batch must be the next exact four ordinals")
            tokens = tuple(
                _OrdinalCapability(self.split, ordinal, object()) for ordinal in ordinals
            )
            batch = _BatchCapability(self.split, ordinals, tokens, object())
            token_sha256 = tuple(_ordinal_capability_sha256(token) for token in tokens)
            batch_sha256 = _batch_capability_sha256(batch)
            if id(batch) in _BATCH_REGISTRY or id(batch) in _CAPABILITY_REGISTRY:
                raise PermissionError("batch identity was already registered")
            for token in tokens:
                if id(token) in _CAPABILITY_REGISTRY:
                    raise PermissionError("ordinal token identity was already registered")
            self._ledger.begin_batch(self.split, ordinals)
            registration = _BatchRegistration(
                batch=batch,
                manifest=self,
                ledger=self._ledger,
                split=self.split,
                ordinals=ordinals,
                batch_sha256=batch_sha256,
                token_sha256=token_sha256,
                consumed_ordinals=frozenset(),
                ledger_generation=self._ledger._record["generation"],
                ledger_record_sha256=sha256_bytes(self._ledger._last_bytes),
                status="issued",
            )
            for index, token in enumerate(tokens):
                _CAPABILITY_REGISTRY[id(token)] = (
                    token,
                    self,
                    batch,
                    "issued",
                    token_sha256[index],
                )
            _CAPABILITY_REGISTRY[id(batch)] = (batch, self, "issued", batch_sha256)
            _BATCH_REGISTRY[id(batch)] = registration
            self._active = batch
            return batch

    def consume_ordinal(
        self,
        batch: _BatchCapability,
        token: _OrdinalCapability,
        *,
        ordinal: int,
    ) -> None:
        with _LEDGER_LOCK:
            batch_registration = self._validated_batch_registration(batch)
            if (
                type(ordinal) is not int
                or type(batch.split) is not str
                or type(token.split) is not str
                or batch.split != self.split
                or token.split != self.split
                or token.ordinal != ordinal
            ):
                raise PermissionError("ordinal capability split/address binding differs")
            if self._active is not batch:
                raise PermissionError("batch is not the live manifest batch")
            if batch.ordinals != tuple(range(self._next_ordinal, self._next_ordinal + BATCH_SIZE)):
                raise PermissionError("batch sequence differs from live manifest cursor")
            index = ordinal - self._next_ordinal
            if not 0 <= index < BATCH_SIZE or batch.tokens[index] is not token:
                raise PermissionError("ordinal token is forged, reordered, or cross-bound")
            registration = _CAPABILITY_REGISTRY.get(id(token))
            if (
                registration is None
                or registration[0] is not token
                or registration[1] is not self
                or registration[2] is not batch
                or registration[3] != "issued"
                or registration[4] != _ordinal_capability_sha256(token)
                or id(token) in self._consumed_tokens
                or ordinal in batch_registration.consumed_ordinals
            ):
                raise PermissionError("ordinal token is forged or replayed")
            _CAPABILITY_REGISTRY[id(token)] = (
                *registration[:3],
                "consumed",
                registration[4],
            )
            _BATCH_REGISTRY[id(batch)] = replace(
                batch_registration,
                consumed_ordinals=(batch_registration.consumed_ordinals | frozenset({ordinal})),
            )
            self._consumed_tokens.add(id(token))

    def complete_batch(
        self,
        batch: _BatchCapability,
        *,
        result_sha256: str,
    ) -> None:
        with _LEDGER_LOCK:
            validated_sha256(result_sha256, label="batch result")
            batch_registration = self._validated_batch_registration(batch)
            token_registrations = tuple(
                _CAPABILITY_REGISTRY.get(id(token)) for token in batch.tokens
            )
            if (
                batch_registration.consumed_ordinals != frozenset(batch.ordinals)
                or any(
                    registration
                    != (
                        token,
                        self,
                        batch,
                        "consumed",
                        _ordinal_capability_sha256(token),
                    )
                    for token, registration in zip(
                        batch.tokens,
                        token_registrations,
                        strict=True,
                    )
                )
                or {id(token) for token in batch.tokens} - self._consumed_tokens
            ):
                raise PermissionError("batch cannot complete with unconsumed ordinals")
            commit = _BatchCommit(
                split=self.split,
                ordinals=batch.ordinals,
                result_sha256=result_sha256,
                nonce=object(),
            )
            if id(commit) in _BATCH_COMMIT_REGISTRY:
                raise PermissionError("batch commit identity was already registered")
            _BATCH_COMMIT_REGISTRY[id(commit)] = _BatchCommitRegistration(
                commit=commit,
                ledger=self._ledger,
                manifest=self,
                batch=batch,
                batch_registration=batch_registration,
                result_sha256=result_sha256,
                owner_thread=threading.get_ident(),
                ledger_generation=self._ledger._record["generation"],
                ledger_record_sha256=sha256_bytes(self._ledger._last_bytes),
                status="issued",
            )
            expected_batch_registration = _BATCH_REGISTRY.pop(id(batch), None)
            expected_batch_capability = _CAPABILITY_REGISTRY.pop(id(batch), None)
            expected_token_registrations = tuple(
                _CAPABILITY_REGISTRY.pop(id(token), None) for token in batch.tokens
            )
            if (
                expected_batch_registration != batch_registration
                or expected_batch_capability
                != (batch, self, "issued", batch_registration.batch_sha256)
                or any(
                    token_registration
                    != (
                        token,
                        self,
                        batch,
                        "consumed",
                        token_digest,
                    )
                    for token, token_digest, token_registration in zip(
                        batch.tokens,
                        batch_registration.token_sha256,
                        expected_token_registrations,
                        strict=True,
                    )
                )
            ):
                _BATCH_COMMIT_REGISTRY.pop(id(commit), None)
                raise PermissionError("batch registries changed before atomic retirement")
            self._ledger.complete_batch(
                self.split,
                batch.ordinals,
                result_sha256=result_sha256,
                commit=commit,
            )
            poisoned = (
                _BATCH_REGISTRY.pop(id(batch), None),
                _CAPABILITY_REGISTRY.pop(id(batch), None),
                _BATCH_COMMIT_REGISTRY.pop(id(commit), None),
                tuple(_CAPABILITY_REGISTRY.pop(id(token), None) for token in batch.tokens),
            )
            if any(item is not None for item in poisoned[:3]) or any(
                item is not None for item in poisoned[3]
            ):
                raise PermissionError("retired batch authority was reintroduced during commit")
            for token in batch.tokens:
                self._consumed_tokens.remove(id(token))
            self._next_ordinal += BATCH_SIZE
            self._active = None

    def close(self, result: Mapping[str, Any]) -> None:
        with _LEDGER_LOCK:
            if self._closed or self._active is not None:
                raise PermissionError("manifest cannot close in its current state")
            if self._next_ordinal != SCENES_PER_SPLIT:
                raise PermissionError("manifest cannot close before ordinal 63")
            self._ledger.complete_split(self.split, result)
            self._closed = True


def _guard_frozen_inputs(
    config: OrpheusConfig,
    *,
    config_path: Path,
    published_source: Mapping[str, Any],
    label: str,
) -> None:
    _require_config_matches_frozen_path(config, config_path)
    _validate_repository_identity()
    _validate_frozen_critical_sources()
    validated_source = _validated_published_source(
        dict(published_source),
        label=f"{label} source",
    )
    _validate_publication_surface(validated_source)
    if torch.get_num_threads() != 1:
        raise RuntimeError("qualification requires exactly one PyTorch intraop thread")


@dataclass(frozen=True, slots=True)
class _PacketEpisode:
    split: str
    ordinal: int
    scene_sha256: str
    non_radius_scene_sha256: str
    primitive_index: int
    pair_variant: int
    radius_role: int
    camera_stratum: int
    twin_ordinal: int
    pair_variant_twin_ordinal: int
    rgb: Tensor
    depth: Tensor
    timestamps: Tensor
    world_from_camera: Tensor
    intrinsics: Tensor
    position_truth: Tensor
    velocity_truth: Tensor
    radius_truth: Tensor
    albedo_truth: Tensor


@dataclass(frozen=True, slots=True)
class SceneEvidence:
    """Bounded sufficient evidence retained after one governed scene."""

    split: str
    ordinal: int
    scene_sha256: str
    non_radius_scene_sha256: str
    primitive_index: int
    pair_variant: int
    radius_role: int
    camera_stratum: int
    twin_ordinal: int
    pair_variant_twin_ordinal: int
    provenance_sha256: str
    radius_truth: Tensor
    anchor_raw_radius: Tensor
    anchor_deployed_radius: Tensor
    history_raw_radius: Tensor
    history_deployed_radius: Tensor
    radius_valid: Tensor
    radius_in_bounds: Tensor
    surface_fit_relative_residual: Tensor
    surface_fit_condition: Tensor
    current_position_truth: Tensor
    current_position_mean: Tensor
    current_velocity_truth: Tensor
    current_velocity_mean: Tensor
    future_position_truth: Tensor
    future_position_mean: Tensor
    future_velocity_truth: Tensor
    future_velocity_mean: Tensor
    object_ids: Tensor
    active: Tensor
    rollout_active: Tensor
    diagnostics: tuple[tuple[str, float], ...]


_EVIDENCE_TENSOR_SHAPES = {
    "radius_truth": (2,),
    "anchor_raw_radius": (2,),
    "anchor_deployed_radius": (2,),
    "history_raw_radius": (16, 2),
    "history_deployed_radius": (16, 2),
    "radius_valid": (16, 2),
    "radius_in_bounds": (16, 2),
    "surface_fit_relative_residual": (16, 2),
    "surface_fit_condition": (16, 2),
    "current_position_truth": (2, 3),
    "current_position_mean": (2, 3),
    "current_velocity_truth": (2, 3),
    "current_velocity_mean": (2, 3),
    "future_position_truth": (5, 2, 3),
    "future_position_mean": (5, 2, 3),
    "future_velocity_truth": (5, 2, 3),
    "future_velocity_mean": (5, 2, 3),
    "object_ids": (16, 2),
    "active": (16, 2),
    "rollout_active": (5, 2),
}


_EPISODE_REGISTRY: dict[int, tuple[object, ...]] = {}
_PACKET_REGISTRY: dict[int, tuple[object, ...]] = {}


@dataclass(frozen=True, slots=True)
class _PacketReceipt:
    packet: ObservationPacket
    packet_id: int
    frame_index: int
    packet_sha256: str


@dataclass(frozen=True, slots=True)
class _EvaluatorProvenanceReceipt:
    split: str
    ordinal: int
    episode_identity: int
    episode_sha256: str
    episode_truth_sha256: str
    evidence_truth_sha256: str
    packet_receipts: tuple[tuple[int, int, str], ...]
    token_identity: int
    token_sha256: str
    batch_identity: int
    batch_sha256: str
    batch_ordinals: tuple[int, ...]
    consumed_ordinals: tuple[int, ...]
    ledger_identity: int
    ledger_stage: str
    ledger_path: str
    ledger_generation: int
    ledger_record_sha256: str
    ledger_bindings_sha256: str
    ledger_artifact_identity: tuple[int, int, int, int, int]
    run_directory_binding_sha256: str
    run_directory_identity: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class _LiveEvidenceRegistration:
    evidence: SceneEvidence
    manifest: _ManifestCapability
    batch: _BatchCapability
    token: _OrdinalCapability
    episode: _PacketEpisode
    split: str
    ordinal: int
    batch_sha256: str
    token_sha256: str
    episode_sha256: str
    episode_truth_sha256: str
    evidence_truth_sha256: str
    packet_receipts: tuple[_PacketReceipt, ...]
    provenance_receipt: _EvaluatorProvenanceReceipt
    provenance_sha256: str
    evidence_sha256: str


@dataclass(frozen=True, slots=True)
class _FinalEvidenceRegistration:
    evidence: SceneEvidence
    split: str
    ordinal: int
    batch_id: int
    token_id: int
    episode_id: int
    batch_sha256: str
    token_sha256: str
    episode_sha256: str
    episode_truth_sha256: str
    evidence_truth_sha256: str
    packet_receipts: tuple[tuple[int, int, str], ...]
    provenance_receipt: _EvaluatorProvenanceReceipt
    provenance_sha256: str
    evidence_sha256: str


_EVIDENCE_REGISTRY: dict[
    int,
    _LiveEvidenceRegistration | _FinalEvidenceRegistration,
] = {}


def _length_framed_digest_update(digest: Any, payload: bytes) -> None:
    if type(payload) is not bytes:
        raise TypeError("digest frame payload must be exact bytes")
    digest.update(len(payload).to_bytes(8, byteorder="big", signed=False))
    digest.update(payload)


def _update_tensor_digest(digest: Any, label: str, tensor: Tensor) -> None:
    if type(label) is not str or not label:
        raise TypeError("tensor digest label must be one nonempty exact string")
    if not isinstance(tensor, Tensor):
        raise TypeError("tensor digest value must be a Tensor")
    value = tensor.detach().cpu().contiguous()
    array = value.numpy()
    dtype = array.dtype
    if dtype.itemsize > 1:
        if dtype.byteorder == ">" or (dtype.byteorder == "=" and sys.byteorder == "big"):
            array = array.byteswap().view(dtype.newbyteorder("<"))
        elif dtype.byteorder == "=":
            array = array.view(dtype.newbyteorder("<"))
        elif dtype.byteorder != "<":
            raise TypeError("tensor digest dtype has no canonical little-endian representation")
    elif dtype.byteorder not in {"|", "=", "<"}:
        raise TypeError("tensor digest single-byte dtype representation differs")
    dtype_descriptor = {
        "torch_dtype": str(value.dtype),
        "numpy_dtype": array.dtype.str,
        "byte_order": "little",
    }
    shape_descriptor = {
        "rank": value.ndim,
        "shape": list(value.shape),
        "order": "C",
    }
    for payload in (
        b"rgbd_variable_radius_tensor_digest_v2",
        label.encode("utf-8", errors="strict"),
        _canonical_json(dtype_descriptor),
        _canonical_json(shape_descriptor),
        array.tobytes(order="C"),
    ):
        _length_framed_digest_update(digest, payload)


def _ordinal_capability_sha256(token: _OrdinalCapability) -> str:
    return canonical_sha256(
        {
            "split": token.split,
            "ordinal": token.ordinal,
            "nonce_identity": id(token.nonce),
        }
    )


def _batch_capability_sha256(batch: _BatchCapability) -> str:
    return canonical_sha256(
        {
            "split": batch.split,
            "ordinals": list(batch.ordinals),
            "tokens": [
                {
                    "identity": id(token),
                    "sha256": _ordinal_capability_sha256(token),
                }
                for token in batch.tokens
            ],
            "nonce_identity": id(batch.nonce),
        }
    )


def _episode_digest(episode: _PacketEpisode) -> str:
    digest = hashlib.sha256()
    digest.update(
        _canonical_json(
            {
                "split": episode.split,
                "ordinal": episode.ordinal,
                "scene_sha256": episode.scene_sha256,
                "non_radius_scene_sha256": episode.non_radius_scene_sha256,
                "primitive_index": episode.primitive_index,
                "pair_variant": episode.pair_variant,
                "radius_role": episode.radius_role,
                "camera_stratum": episode.camera_stratum,
                "twin_ordinal": episode.twin_ordinal,
                "pair_variant_twin_ordinal": episode.pair_variant_twin_ordinal,
            }
        )
    )
    for name in (
        "rgb",
        "depth",
        "timestamps",
        "world_from_camera",
        "intrinsics",
        "position_truth",
        "velocity_truth",
        "radius_truth",
        "albedo_truth",
    ):
        _update_tensor_digest(digest, name, getattr(episode, name))
    return digest.hexdigest()


def _episode_truth_digest(episode: _PacketEpisode) -> str:
    digest = hashlib.sha256()
    for name in ("position_truth", "velocity_truth", "radius_truth"):
        _update_tensor_digest(digest, name, getattr(episode, name))
    return digest.hexdigest()


def _validate_episode_registration(
    episode: _PacketEpisode,
) -> tuple[_ManifestCapability, _BatchCapability]:
    if (
        not isinstance(episode, _PacketEpisode)
        or type(episode.split) is not str
        or episode.split not in SPLITS
        or type(episode.ordinal) is not int
        or not 0 <= episode.ordinal < SCENES_PER_SPLIT
        or any(
            type(axis) is not int
            for axis in (
                episode.primitive_index,
                episode.pair_variant,
                episode.radius_role,
                episode.camera_stratum,
                episode.twin_ordinal,
                episode.pair_variant_twin_ordinal,
            )
        )
    ):
        raise TypeError("packet episode address axes must use exact builtin scalar types")
    registration = _EPISODE_REGISTRY.get(id(episode))
    if (
        registration is None
        or len(registration) != 7
        or registration[0] is not episode
        or not isinstance(registration[1], _ManifestCapability)
        or not isinstance(registration[2], _BatchCapability)
        or not isinstance(registration[3], _OrdinalCapability)
        or registration[4] != episode.split
        or registration[5] != episode.ordinal
        or registration[6] != _episode_digest(episode)
    ):
        raise PermissionError("packet episode is substituted, mutated, or not receipt-owned")
    manifest = registration[1]
    batch = registration[2]
    token = registration[3]
    batch_registration = manifest._validated_batch_registration(batch)
    token_registration = _CAPABILITY_REGISTRY.get(id(token))
    if (
        manifest.split != episode.split
        or batch.split != episode.split
        or token.split != episode.split
        or token.ordinal != episode.ordinal
        or episode.ordinal not in batch_registration.consumed_ordinals
        or manifest._active is not batch
        or token_registration
        != (
            token,
            manifest,
            batch,
            "consumed",
            _ordinal_capability_sha256(token),
        )
    ):
        raise PermissionError("packet episode live capability binding differs")
    return manifest, batch


def _validate_episode_batch(
    episodes: Sequence[_PacketEpisode],
) -> tuple[_ManifestCapability, _BatchCapability]:
    if type(episodes) not in {list, tuple} or len(episodes) != BATCH_SIZE:
        raise ValueError("governed packet batch requires exactly four episodes")
    bindings = [_validate_episode_registration(episode) for episode in episodes]
    manifest, batch = bindings[0]
    if any(item != (manifest, batch) for item in bindings):
        raise PermissionError("episode batch mixes manifest or batch owners")
    if tuple(episode.ordinal for episode in episodes) != batch.ordinals:
        raise PermissionError("episode batch is reordered or substituted")
    return manifest, batch


def _live_batch_episodes(
    manifest: _ManifestCapability,
    batch: _BatchCapability,
) -> tuple[_PacketEpisode, ...]:
    registrations = sorted(
        (
            registration
            for registration in _EPISODE_REGISTRY.values()
            if len(registration) == 7 and registration[1] is manifest and registration[2] is batch
        ),
        key=lambda registration: int(registration[5]),
    )
    episodes = tuple(registration[0] for registration in registrations)
    if any(not isinstance(episode, _PacketEpisode) for episode in episodes):
        raise PermissionError("live packet batch contains a non-episode identity")
    _validate_episode_batch(episodes)
    return episodes


def _packet_digest(packet: ObservationPacket) -> str:
    digest = hashlib.sha256()
    digest.update(
        _canonical_json(
            {
                "modality": packet.modality,
                "sensor_id": packet.sensor_id,
                "timestamp": packet.timestamp,
                "frame_id": packet.frame_id,
                "confidence": packet.confidence,
                "metadata": dict(packet.metadata),
            }
        )
    )
    if type(packet.payload) is not dict or set(packet.payload) != {"rgb", "depth"}:
        raise ValueError("model-visible packet payload schema differs")
    if type(packet.calibration) is not dict or set(packet.calibration) != {
        "world_from_camera",
        "intrinsics",
    }:
        raise ValueError("model-visible packet camera schema differs")
    if set(packet.metadata) != {"image_size", "depth_semantics"}:
        raise ValueError("model-visible packet metadata acquired governed or truth labels")
    for name, tensor in (
        ("rgb", packet.payload["rgb"]),
        ("depth", packet.payload["depth"]),
        ("world_from_camera", packet.calibration["world_from_camera"]),
        ("intrinsics", packet.calibration["intrinsics"]),
    ):
        if not isinstance(tensor, Tensor):
            raise TypeError(f"packet field {name} must be tensor")
        _update_tensor_digest(digest, name, tensor)
    return digest.hexdigest()


def _register_packet(
    packet: ObservationPacket,
    *,
    episodes: Sequence[_PacketEpisode],
    frame_index: int,
    provenance: str,
) -> None:
    manifest, batch = _validate_episode_batch(episodes)
    if (
        type(frame_index) is not int
        or frame_index not in HISTORY_FRAME_INDICES
        or provenance != "nominal_independent_packet"
    ):
        raise PermissionError("nominal packet receipt address or provenance differs")
    if id(packet) in _PACKET_REGISTRY:
        raise PermissionError("packet identity was already registered")
    _PACKET_REGISTRY[id(packet)] = (
        packet,
        manifest,
        batch,
        tuple(id(episode) for episode in episodes),
        frame_index,
        provenance,
        _packet_digest(packet),
    )


def _validate_packet_registration(packet: ObservationPacket) -> tuple[object, ...]:
    registration = _PACKET_REGISTRY.get(id(packet))
    if (
        registration is None
        or len(registration) != 7
        or registration[0] is not packet
        or not isinstance(registration[1], _ManifestCapability)
        or not isinstance(registration[2], _BatchCapability)
        or type(registration[3]) is not tuple
        or type(registration[4]) is not int
        or type(registration[5]) is not str
        or registration[6] != _packet_digest(packet)
    ):
        raise PermissionError("model-visible packet is substituted, mutated, or unregistered")
    manifest = registration[1]
    batch = registration[2]
    episodes = _live_batch_episodes(manifest, batch)
    if (
        manifest._active is not batch
        or registration[3] != tuple(id(episode) for episode in episodes)
        or registration[4] not in HISTORY_FRAME_INDICES
        or not registration[5]
    ):
        raise PermissionError("model-visible packet outlived its live batch receipt")
    return registration


def _register_derived_packet(
    parent: ObservationPacket,
    child: ObservationPacket,
    *,
    provenance: str,
) -> ObservationPacket:
    _validate_packet_registration(parent)
    parent_registration = _PACKET_REGISTRY[id(parent)]
    if (
        child is parent
        or id(child) in _PACKET_REGISTRY
        or type(provenance) is not str
        or not provenance
        or provenance == "nominal_independent_packet"
    ):
        raise PermissionError("derived packet must have one fresh identity")
    _PACKET_REGISTRY[id(child)] = (
        child,
        parent_registration[1],
        parent_registration[2],
        parent_registration[3],
        parent_registration[4],
        provenance,
        _packet_digest(child),
    )
    return child


def _scene_evidence_digest(value: SceneEvidence) -> str:
    digest = hashlib.sha256()
    digest.update(
        _canonical_json(
            {
                "split": value.split,
                "ordinal": value.ordinal,
                "scene_sha256": value.scene_sha256,
                "non_radius_scene_sha256": value.non_radius_scene_sha256,
                "primitive_index": value.primitive_index,
                "pair_variant": value.pair_variant,
                "radius_role": value.radius_role,
                "camera_stratum": value.camera_stratum,
                "twin_ordinal": value.twin_ordinal,
                "pair_variant_twin_ordinal": value.pair_variant_twin_ordinal,
                "provenance_sha256": value.provenance_sha256,
                "diagnostics": list(value.diagnostics),
            }
        )
    )
    for name in _EVIDENCE_TENSOR_SHAPES:
        _update_tensor_digest(digest, name, getattr(value, name))
    return digest.hexdigest()


def _scene_evidence_truth_digest(value: SceneEvidence) -> str:
    digest = hashlib.sha256()
    for name in (
        "radius_truth",
        "current_position_truth",
        "current_velocity_truth",
        "future_position_truth",
        "future_velocity_truth",
    ):
        _update_tensor_digest(digest, name, getattr(value, name))
    return digest.hexdigest()


def _expected_scene_evidence_truth_digest(episode: _PacketEpisode) -> str:
    digest = hashlib.sha256()
    expected = {
        "radius_truth": episode.radius_truth,
        "current_position_truth": episode.position_truth[ANCHOR_FRAME_INDEX],
        "current_velocity_truth": episode.velocity_truth[ANCHOR_FRAME_INDEX],
        "future_position_truth": episode.position_truth[list(TARGET_FRAME_INDICES)],
        "future_velocity_truth": episode.velocity_truth[list(TARGET_FRAME_INDICES)],
    }
    for name, tensor in expected.items():
        _update_tensor_digest(digest, name, tensor)
    return digest.hexdigest()


def _evidence_matches_episode_truth(value: SceneEvidence, episode: _PacketEpisode) -> bool:
    expected = {
        "radius_truth": episode.radius_truth,
        "current_position_truth": episode.position_truth[ANCHOR_FRAME_INDEX],
        "current_velocity_truth": episode.velocity_truth[ANCHOR_FRAME_INDEX],
        "future_position_truth": episode.position_truth[list(TARGET_FRAME_INDICES)],
        "future_velocity_truth": episode.velocity_truth[list(TARGET_FRAME_INDICES)],
    }
    return all(
        torch.equal(
            getattr(value, name),
            tensor.detach().cpu().contiguous(),
        )
        for name, tensor in expected.items()
    )


def _nominal_packet_receipts(
    packets: Sequence[ObservationPacket],
    episodes: Sequence[_PacketEpisode],
) -> tuple[_PacketReceipt, ...]:
    manifest, batch = _validate_episode_batch(episodes)
    if type(packets) not in {list, tuple} or len(packets) != HISTORY_FRAME_COUNT:
        raise PermissionError("nominal packet inventory must contain exactly 16 frames")
    expected_episode_ids = tuple(id(episode) for episode in episodes)
    receipts: list[_PacketReceipt] = []
    for frame_index, packet in enumerate(packets):
        registration = _validate_packet_registration(packet)
        if (
            registration[1] is not manifest
            or registration[2] is not batch
            or registration[3] != expected_episode_ids
            or registration[4] != frame_index
            or registration[5] != "nominal_independent_packet"
        ):
            raise PermissionError("nominal packet frame/order/episode receipt differs")
        receipts.append(
            _PacketReceipt(
                packet=packet,
                packet_id=id(packet),
                frame_index=frame_index,
                packet_sha256=registration[6],
            )
        )
    nominal_owned = tuple(
        registration[0]
        for registration in _PACKET_REGISTRY.values()
        if registration[1] is manifest
        and registration[2] is batch
        and registration[5] == "nominal_independent_packet"
    )
    if nominal_owned != tuple(packets):
        raise PermissionError("registered nominal packet set/order is not exact")
    return tuple(receipts)


def _validated_evaluator_provenance_receipt(
    receipt: _EvaluatorProvenanceReceipt,
) -> _EvaluatorProvenanceReceipt:
    if not isinstance(receipt, _EvaluatorProvenanceReceipt):
        raise TypeError("evaluator provenance receipt has the wrong nominal type")
    expected_stage = "development" if receipt.split == "development" else "qualification"
    packet_receipts_valid = type(receipt.packet_receipts) is tuple and all(
        type(item) is tuple
        and len(item) == 3
        and type(item[0]) is int
        and type(item[1]) is int
        and type(item[2]) is str
        for item in receipt.packet_receipts
    )
    scalar_sha256s = (
        receipt.episode_sha256,
        receipt.episode_truth_sha256,
        receipt.evidence_truth_sha256,
        receipt.token_sha256,
        receipt.batch_sha256,
        receipt.ledger_record_sha256,
        receipt.ledger_bindings_sha256,
        receipt.run_directory_binding_sha256,
    )
    if (
        type(receipt.split) is not str
        or receipt.split not in SPLITS
        or type(receipt.ordinal) is not int
        or not 0 <= receipt.ordinal < SCENES_PER_SPLIT
        or any(
            type(identity) is not int
            for identity in (
                receipt.episode_identity,
                receipt.token_identity,
                receipt.batch_identity,
                receipt.ledger_identity,
            )
        )
        or type(receipt.batch_ordinals) is not tuple
        or len(receipt.batch_ordinals) != BATCH_SIZE
        or any(type(ordinal) is not int for ordinal in receipt.batch_ordinals)
        or receipt.ordinal not in receipt.batch_ordinals
        or receipt.consumed_ordinals != receipt.batch_ordinals
        or type(receipt.packet_receipts) is not tuple
        or len(receipt.packet_receipts) != HISTORY_FRAME_COUNT
        or not packet_receipts_valid
        or tuple(item[1] for item in receipt.packet_receipts) != HISTORY_FRAME_INDICES
        or len({item[0] for item in receipt.packet_receipts}) != HISTORY_FRAME_COUNT
        or type(receipt.ledger_stage) is not str
        or receipt.ledger_stage != expected_stage
        or type(receipt.ledger_path) is not str
        or not receipt.ledger_path
        or type(receipt.ledger_generation) is not int
        or receipt.ledger_generation < 1
        or type(receipt.ledger_artifact_identity) is not tuple
        or len(receipt.ledger_artifact_identity) != 5
        or any(type(value) is not int for value in receipt.ledger_artifact_identity)
        or receipt.ledger_artifact_identity[-1] != 1
        or type(receipt.run_directory_identity) is not tuple
        or len(receipt.run_directory_identity) != 3
        or any(type(value) is not int for value in receipt.run_directory_identity)
        or receipt.run_directory_identity[0] != receipt.ledger_artifact_identity[0]
        or not stat.S_ISDIR(receipt.run_directory_identity[2])
    ):
        raise PermissionError("evaluator provenance receipt exact schema differs")
    for index, value in enumerate(scalar_sha256s):
        validated_sha256(value, label=f"evaluator provenance scalar digest {index}")
    for _, _, value in receipt.packet_receipts:
        validated_sha256(value, label="evaluator provenance packet digest")
    return receipt


def _provenance_receipt_sha256(receipt: _EvaluatorProvenanceReceipt) -> str:
    receipt = _validated_evaluator_provenance_receipt(receipt)
    return canonical_sha256(
        {
            "schema": "variable_radius_evaluator_provenance_receipt_v3",
            "receipt": asdict(receipt),
        }
    )


def _evaluator_provenance_receipt(
    episode: _PacketEpisode,
    packets: Sequence[ObservationPacket],
    *,
    evidence_truth_sha256: str,
) -> _EvaluatorProvenanceReceipt:
    evidence_truth_sha256 = validated_sha256(
        evidence_truth_sha256,
        label="evaluator evidence truth receipt",
    )
    manifest, batch = _validate_episode_registration(episode)
    episode_registration = _EPISODE_REGISTRY[id(episode)]
    token = episode_registration[3]
    if not isinstance(token, _OrdinalCapability):
        raise PermissionError("episode provenance token has the wrong nominal type")
    batch_registration = manifest._validated_batch_registration(batch)
    if batch_registration.consumed_ordinals != frozenset(batch.ordinals):
        raise PermissionError("evaluator provenance requires one fully consumed live batch")
    packet_receipts = _nominal_packet_receipts(packets, _live_batch_episodes(manifest, batch))
    ledger = batch_registration.ledger
    ledger._verify_disk()
    ledger_registration = _LEDGER_REGISTRY.get(id(ledger))
    if (
        not isinstance(ledger_registration, _LedgerRegistration)
        or ledger_registration.ledger is not ledger
        or ledger_registration.generation != batch_registration.ledger_generation
        or ledger_registration.record_sha256 != batch_registration.ledger_record_sha256
        or ledger_registration.bindings_sha256 != ledger._bindings_sha256
        or ledger_registration.artifact_identity != ledger._last_identity
        or ledger_registration.record_bytes != ledger._last_bytes
        or ledger_registration.directory_pin is not ledger._directory_pin
        or ledger_registration.directory_binding_sha256 != ledger._directory_binding_sha256
        or ledger_registration.directory_capability_sha256 != ledger._directory_capability_sha256
    ):
        raise PermissionError("evaluator provenance ledger receipt differs")
    expected_truth_sha256 = _expected_scene_evidence_truth_digest(episode)
    if evidence_truth_sha256 != expected_truth_sha256:
        raise PermissionError("evaluator evidence truth receipt differs from its episode")
    receipt = _EvaluatorProvenanceReceipt(
        split=episode.split,
        ordinal=episode.ordinal,
        episode_identity=id(episode),
        episode_sha256=episode_registration[6],
        episode_truth_sha256=_episode_truth_digest(episode),
        evidence_truth_sha256=evidence_truth_sha256,
        packet_receipts=tuple(
            (item.packet_id, item.frame_index, item.packet_sha256) for item in packet_receipts
        ),
        token_identity=id(token),
        token_sha256=_ordinal_capability_sha256(token),
        batch_identity=id(batch),
        batch_sha256=batch_registration.batch_sha256,
        batch_ordinals=batch.ordinals,
        consumed_ordinals=tuple(sorted(batch_registration.consumed_ordinals)),
        ledger_identity=id(ledger),
        ledger_stage=ledger.stage,
        ledger_path=os.fspath(ledger.path),
        ledger_generation=ledger_registration.generation,
        ledger_record_sha256=ledger_registration.record_sha256,
        ledger_bindings_sha256=ledger_registration.bindings_sha256,
        ledger_artifact_identity=ledger_registration.artifact_identity,
        run_directory_binding_sha256=ledger_registration.directory_binding_sha256,
        run_directory_identity=_pinned_directory_registration(
            ledger_registration.directory_pin
        ).directory_identity,
    )
    validated_sha256(
        _provenance_receipt_sha256(receipt),
        label="evaluator provenance receipt",
    )
    return receipt


def _register_scene_evidence(
    value: SceneEvidence,
    *,
    episode: _PacketEpisode,
    packets: Sequence[ObservationPacket],
) -> None:
    if not isinstance(value, SceneEvidence):
        raise TypeError("scene evidence has the wrong nominal type")
    if id(value) in _EVIDENCE_REGISTRY:
        raise PermissionError("scene evidence identity was already registered")
    manifest, batch = _validate_episode_registration(episode)
    episode_registration = _EPISODE_REGISTRY[id(episode)]
    token = episode_registration[3]
    if (
        value.split != episode.split
        or value.ordinal != episode.ordinal
        or value.scene_sha256 != episode.scene_sha256
        or value.non_radius_scene_sha256 != episode.non_radius_scene_sha256
        or value.primitive_index != episode.primitive_index
        or value.pair_variant != episode.pair_variant
        or value.radius_role != episode.radius_role
        or value.camera_stratum != episode.camera_stratum
        or value.twin_ordinal != episode.twin_ordinal
        or value.pair_variant_twin_ordinal != episode.pair_variant_twin_ordinal
        or not _evidence_matches_episode_truth(value, episode)
    ):
        raise PermissionError("scene evidence does not match its exact episode truth receipt")
    episodes = _live_batch_episodes(manifest, batch)
    packet_receipts = _nominal_packet_receipts(packets, episodes)
    evidence_truth_sha256 = _scene_evidence_truth_digest(value)
    provenance_receipt = _evaluator_provenance_receipt(
        episode,
        packets,
        evidence_truth_sha256=evidence_truth_sha256,
    )
    provenance_sha256 = _provenance_receipt_sha256(provenance_receipt)
    if (
        validated_sha256(value.provenance_sha256, label="scene evidence provenance")
        != provenance_sha256
    ):
        raise PermissionError("scene evidence evaluator provenance receipt differs")
    _EVIDENCE_REGISTRY[id(value)] = _LiveEvidenceRegistration(
        evidence=value,
        manifest=manifest,
        batch=batch,
        token=token,
        episode=episode,
        split=value.split,
        ordinal=value.ordinal,
        batch_sha256=_batch_capability_sha256(batch),
        token_sha256=_ordinal_capability_sha256(token),
        episode_sha256=_episode_digest(episode),
        episode_truth_sha256=_episode_truth_digest(episode),
        evidence_truth_sha256=evidence_truth_sha256,
        packet_receipts=packet_receipts,
        provenance_receipt=provenance_receipt,
        provenance_sha256=provenance_sha256,
        evidence_sha256=_scene_evidence_digest(value),
    )


def _validate_live_evidence_registration(
    value: SceneEvidence,
    registration: _LiveEvidenceRegistration,
) -> None:
    manifest, batch = _validate_episode_registration(registration.episode)
    episode_registration = _EPISODE_REGISTRY[id(registration.episode)]
    episodes = _live_batch_episodes(manifest, batch)
    current_receipts = _nominal_packet_receipts(
        tuple(receipt.packet for receipt in registration.packet_receipts),
        episodes,
    )
    receipt_scalars = tuple(
        (receipt.packet_id, receipt.frame_index, receipt.packet_sha256)
        for receipt in registration.packet_receipts
    )
    current_scalars = tuple(
        (receipt.packet_id, receipt.frame_index, receipt.packet_sha256)
        for receipt in current_receipts
    )
    current_provenance_receipt = _evaluator_provenance_receipt(
        registration.episode,
        tuple(receipt.packet for receipt in registration.packet_receipts),
        evidence_truth_sha256=_scene_evidence_truth_digest(value),
    )
    current_provenance_sha256 = _provenance_receipt_sha256(current_provenance_receipt)
    if (
        registration.evidence is not value
        or registration.manifest is not manifest
        or registration.batch is not batch
        or registration.token is not episode_registration[3]
        or registration.split != value.split
        or registration.ordinal != value.ordinal
        or registration.batch_sha256 != _batch_capability_sha256(batch)
        or registration.token_sha256 != _ordinal_capability_sha256(registration.token)
        or registration.episode_sha256 != _episode_digest(registration.episode)
        or registration.episode_truth_sha256 != _episode_truth_digest(registration.episode)
        or registration.evidence_truth_sha256 != _scene_evidence_truth_digest(value)
        or registration.provenance_receipt != current_provenance_receipt
        or registration.provenance_sha256 != current_provenance_sha256
        or value.provenance_sha256 != current_provenance_sha256
        or registration.evidence_sha256 != _scene_evidence_digest(value)
        or receipt_scalars != current_scalars
        or not _evidence_matches_episode_truth(value, registration.episode)
    ):
        raise PermissionError("scene evidence live episode/packet/truth receipt differs")


def _finalize_batch_provenance(
    episodes: Sequence[_PacketEpisode],
    rows: Sequence[SceneEvidence],
    packets: Sequence[ObservationPacket],
) -> None:
    manifest, batch = _validate_episode_batch(episodes)
    if (
        type(rows) not in {list, tuple}
        or len(rows) != BATCH_SIZE
        or tuple(row.ordinal for row in rows) != batch.ordinals
    ):
        raise PermissionError("evidence ordinals differ from packet batch receipt")
    packet_receipts = _nominal_packet_receipts(packets, episodes)
    owned_packets = tuple(
        registration[0]
        for registration in _PACKET_REGISTRY.values()
        if registration[1] is manifest and registration[2] is batch
    )
    if owned_packets != tuple(packets):
        raise PermissionError("batch packet registry is not the exact ordered 16-frame set")

    registrations: list[_LiveEvidenceRegistration] = []
    for row, episode in zip(rows, episodes, strict=True):
        registration = _EVIDENCE_REGISTRY.get(id(row))
        if not isinstance(registration, _LiveEvidenceRegistration):
            raise PermissionError("scene evidence differs from live batch receipt")
        _validate_live_evidence_registration(row, registration)
        if registration.episode is not episode:
            raise PermissionError("scene evidence is rebound to a different batch episode")
        expected_packet_scalars = tuple(
            (receipt.packet_id, receipt.frame_index, receipt.packet_sha256)
            for receipt in packet_receipts
        )
        registered_packet_scalars = tuple(
            (receipt.packet_id, receipt.frame_index, receipt.packet_sha256)
            for receipt in registration.packet_receipts
        )
        if registered_packet_scalars != expected_packet_scalars:
            raise PermissionError("scene evidence packet frame/digest set differs")
        registrations.append(registration)
    live_owned_evidence = tuple(
        registration.evidence
        for registration in _EVIDENCE_REGISTRY.values()
        if isinstance(registration, _LiveEvidenceRegistration)
        and registration.manifest is manifest
        and registration.batch is batch
    )
    if live_owned_evidence != tuple(rows):
        raise PermissionError("batch evidence registry is not the exact ordered row set")

    for packet in packets:
        _validate_packet_registration(packet)
    for episode in episodes:
        _validate_episode_registration(episode)
    final_registrations = tuple(
        _FinalEvidenceRegistration(
            evidence=registration.evidence,
            split=registration.split,
            ordinal=registration.ordinal,
            batch_id=id(registration.batch),
            token_id=id(registration.token),
            episode_id=id(registration.episode),
            batch_sha256=registration.batch_sha256,
            token_sha256=registration.token_sha256,
            episode_sha256=registration.episode_sha256,
            episode_truth_sha256=registration.episode_truth_sha256,
            evidence_truth_sha256=registration.evidence_truth_sha256,
            packet_receipts=tuple(
                (receipt.packet_id, receipt.frame_index, receipt.packet_sha256)
                for receipt in registration.packet_receipts
            ),
            provenance_receipt=registration.provenance_receipt,
            provenance_sha256=registration.provenance_sha256,
            evidence_sha256=registration.evidence_sha256,
        )
        for registration in registrations
    )

    for row, registration in zip(rows, final_registrations, strict=True):
        _EVIDENCE_REGISTRY[id(row)] = registration
    for packet in packets:
        del _PACKET_REGISTRY[id(packet)]
    for episode in episodes:
        del _EPISODE_REGISTRY[id(episode)]


def _retire_derived_batch_packets(
    episodes: Sequence[_PacketEpisode],
    nominal_packets: Sequence[ObservationPacket],
) -> None:
    manifest, batch = _validate_episode_batch(episodes)
    _nominal_packet_receipts(nominal_packets, episodes)
    nominal_ids = {id(packet) for packet in nominal_packets}
    derived = tuple(
        registration[0]
        for registration in _PACKET_REGISTRY.values()
        if registration[1] is manifest
        and registration[2] is batch
        and id(registration[0]) not in nominal_ids
    )
    for packet in derived:
        registration = _validate_packet_registration(packet)
        if registration[5] == "nominal_independent_packet":
            raise PermissionError("extra nominal packet cannot retire as derived")
    for packet in derived:
        del _PACKET_REGISTRY[id(packet)]


def _validated_evidence(
    value: SceneEvidence,
    *,
    split: str,
    ordinal: int,
) -> SceneEvidence:
    if not isinstance(value, SceneEvidence):
        raise TypeError("scene evidence has the wrong nominal type")
    if (
        type(split) is not str
        or split not in SPLITS
        or type(ordinal) is not int
        or not 0 <= ordinal < SCENES_PER_SPLIT
        or type(value.split) is not str
        or type(value.ordinal) is not int
        or any(
            type(axis) is not int
            for axis in (
                value.primitive_index,
                value.pair_variant,
                value.radius_role,
                value.camera_stratum,
                value.twin_ordinal,
                value.pair_variant_twin_ordinal,
            )
        )
    ):
        raise TypeError("scene evidence address axes must use exact builtin scalar types")
    if value.split != split or value.ordinal != ordinal:
        raise ValueError("scene evidence split/ordinal binding differs")
    validated_sha256(value.scene_sha256, label="scene evidence signature")
    validated_sha256(
        value.non_radius_scene_sha256,
        label="scene evidence non-radius signature",
    )
    validated_sha256(value.provenance_sha256, label="scene evidence provenance")
    if (
        value.primitive_index != ordinal // 32
        or value.pair_variant != (ordinal % 32) // 16
        or value.radius_role != (ordinal % 16) // 8
        or value.camera_stratum != ordinal % 8
        or value.twin_ordinal != counterfactual_twin_ordinal(ordinal)
        or value.pair_variant_twin_ordinal != pair_variant_twin_ordinal(ordinal)
    ):
        raise ValueError("scene evidence ordinal axes differ")
    for name, shape in _EVIDENCE_TENSOR_SHAPES.items():
        tensor = getattr(value, name)
        if not isinstance(tensor, Tensor) or tensor.shape != shape:
            raise ValueError(f"scene evidence {name} must have shape {shape}")
        if tensor.device.type != "cpu" or not tensor.is_contiguous():
            raise ValueError(f"scene evidence {name} must be contiguous CPU")
        if name in {"radius_valid", "radius_in_bounds", "active", "rollout_active"}:
            if tensor.dtype is not torch.bool:
                raise TypeError(f"scene evidence {name} must be bool")
        elif name == "object_ids":
            if tensor.dtype is not torch.int64:
                raise TypeError("scene evidence object_ids must be int64")
        else:
            if tensor.dtype is not torch.float32 or not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"scene evidence {name} must be finite float32")
    if type(value.diagnostics) is not tuple:
        raise TypeError("scene diagnostics must be an exact tuple")
    names: list[str] = []
    for item in value.diagnostics:
        if (
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or type(item[1]) is not float
            or not math.isfinite(item[1])
        ):
            raise ValueError("scene diagnostics must contain exact finite float pairs")
        names.append(item[0])
    if names != sorted(names) or len(names) != len(set(names)):
        raise ValueError("scene diagnostics must be sorted and unique")
    registration = _EVIDENCE_REGISTRY.get(id(value))
    if isinstance(registration, _LiveEvidenceRegistration):
        _validate_live_evidence_registration(value, registration)
    elif isinstance(registration, _FinalEvidenceRegistration):
        provenance_receipt = registration.provenance_receipt
        if (
            registration.evidence is not value
            or registration.split != split
            or registration.ordinal != ordinal
            or type(registration.batch_id) is not int
            or type(registration.token_id) is not int
            or type(registration.episode_id) is not int
            or validated_sha256(
                registration.batch_sha256,
                label="final evidence batch receipt",
            )
            != registration.batch_sha256
            or validated_sha256(
                registration.token_sha256,
                label="final evidence token receipt",
            )
            != registration.token_sha256
            or validated_sha256(
                registration.episode_sha256,
                label="final evidence episode receipt",
            )
            != registration.episode_sha256
            or validated_sha256(
                registration.episode_truth_sha256,
                label="final evidence episode truth receipt",
            )
            != registration.episode_truth_sha256
            or registration.evidence_truth_sha256 != _scene_evidence_truth_digest(value)
            or not isinstance(provenance_receipt, _EvaluatorProvenanceReceipt)
            or provenance_receipt.split != split
            or provenance_receipt.ordinal != ordinal
            or provenance_receipt.episode_identity != registration.episode_id
            or provenance_receipt.token_identity != registration.token_id
            or provenance_receipt.batch_identity != registration.batch_id
            or provenance_receipt.batch_sha256 != registration.batch_sha256
            or provenance_receipt.token_sha256 != registration.token_sha256
            or provenance_receipt.episode_sha256 != registration.episode_sha256
            or provenance_receipt.episode_truth_sha256 != registration.episode_truth_sha256
            or provenance_receipt.evidence_truth_sha256 != registration.evidence_truth_sha256
            or provenance_receipt.packet_receipts != registration.packet_receipts
            or validated_sha256(
                provenance_receipt.ledger_record_sha256,
                label="final evidence ledger record receipt",
            )
            != provenance_receipt.ledger_record_sha256
            or validated_sha256(
                provenance_receipt.ledger_bindings_sha256,
                label="final evidence ledger bindings receipt",
            )
            != provenance_receipt.ledger_bindings_sha256
            or type(provenance_receipt.ledger_generation) is not int
            or type(provenance_receipt.ledger_identity) is not int
            or type(provenance_receipt.ledger_stage) is not str
            or type(provenance_receipt.ledger_path) is not str
            or registration.provenance_sha256 != _provenance_receipt_sha256(provenance_receipt)
            or value.provenance_sha256 != registration.provenance_sha256
            or registration.evidence_sha256 != _scene_evidence_digest(value)
            or len(registration.packet_receipts) != HISTORY_FRAME_COUNT
            or tuple(item[1] for item in registration.packet_receipts) != HISTORY_FRAME_INDICES
            or len({item[0] for item in registration.packet_receipts}) != HISTORY_FRAME_COUNT
            or any(
                validated_sha256(item[2], label="final evidence packet receipt") != item[2]
                for item in registration.packet_receipts
            )
        ):
            raise PermissionError("scene evidence finalized receipt differs")
    else:
        raise PermissionError("scene evidence is substituted, mutated, or not receipt-owned")
    return value


def _independent_raster(
    positions: Tensor,
    radius: Tensor,
    world_from_camera: Tensor,
    intrinsics: Tensor,
    albedo: Tensor,
) -> tuple[Tensor, Tensor]:
    """Render exact source-bound packets without a public renderer call."""

    height, width = IMAGE_SIZE
    camera_position = world_from_camera[:, :3, 3]
    relative = positions - camera_position[:, None, :]
    rotation = world_from_camera[:, :3, :3]
    points_camera = torch.einsum("foi,fij->foj", relative, rotation)
    pixel_y, pixel_x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    ray_x = (pixel_x[None] - intrinsics[:, None, None, 0, 2]) / intrinsics[:, None, None, 0, 0]
    ray_y = (pixel_y[None] - intrinsics[:, None, None, 1, 2]) / intrinsics[:, None, None, 1, 1]
    ray_norm_squared = 1.0 + ray_x.square() + ray_y.square()
    ray_dot_centre = (
        ray_x[:, None] * points_camera[..., 0, None, None]
        + ray_y[:, None] * points_camera[..., 1, None, None]
        + points_camera[..., 2, None, None]
    )
    centre_cross_ray = torch.stack(
        (
            points_camera[..., 1, None, None] - points_camera[..., 2, None, None] * ray_y[:, None],
            points_camera[..., 2, None, None] * ray_x[:, None] - points_camera[..., 0, None, None],
            points_camera[..., 0, None, None] * ray_y[:, None]
            - points_camera[..., 1, None, None] * ray_x[:, None],
        ),
        dim=-1,
    )
    discriminant = ray_norm_squared[:, None] * radius[
        None, :, None, None
    ].square() - centre_cross_ray.square().sum(dim=-1)
    square_root = discriminant.clamp_min(0.0).sqrt()
    denominator = ray_dot_centre + square_root
    constant = (
        points_camera.square().sum(dim=-1)[..., None, None] - radius[None, :, None, None].square()
    )
    surface_depth = constant / denominator.clamp_min(1.0e-12)
    full_mask = (
        (points_camera[..., 2] > radius[None, :] + 1.0e-4)[..., None, None]
        & (discriminant >= 0.0)
        & (denominator > 0.0)
        & (surface_depth > 0.0)
        & torch.isfinite(surface_depth)
    )
    ordered = torch.where(full_mask, surface_depth, torch.full_like(surface_depth, torch.inf))
    depth_buffer, winner = ordered.min(dim=1)
    foreground = torch.isfinite(depth_buffer)
    winner = torch.where(
        foreground,
        winner.to(torch.int64),
        torch.full_like(winner, -1),
    )
    depth = torch.where(foreground, depth_buffer, torch.zeros_like(depth_buffer))[:, None]
    rgb = positions.new_zeros((positions.shape[0], height, width, 3))
    for object_index in OBJECT_INDICES:
        rgb = torch.where(
            (winner == object_index)[..., None],
            albedo[object_index][None, None, None, :],
            rgb,
        )
    return rgb.permute(0, 3, 1, 2).contiguous(), depth.contiguous()


def _materialise_authorized_episode(
    *,
    split: str,
    ordinal: int,
    manifest: _ManifestCapability,
    batch: _BatchCapability,
    token: _OrdinalCapability,
) -> _PacketEpisode:
    """Consume authority before the first formal-scene API call."""

    if (
        type(split) is not str
        or split != manifest.split
        or split != batch.split
        or split != token.split
        or type(ordinal) is not int
        or ordinal != token.ordinal
        or ordinal not in batch.ordinals
    ):
        raise PermissionError("formal scene address differs from every capability binding")
    manifest._ledger._assert_formal_access()
    manifest.consume_ordinal(batch, token, ordinal=ordinal)
    specification = scene_specification(split, ordinal)
    non_radius_metadata = scene_metadata(specification)
    for key in ("ordinal", "pair_variant", "radius_role", "radius_rational"):
        non_radius_metadata.pop(key)
    trajectory = manual_kinematic_trajectory(specification)
    frames = tuple(
        pure_orbital_camera_frame(
            specification.camera_stratum,
            frame_index / FRAME_RATE_HZ,
        )
        for frame_index in range(FRAME_COUNT)
    )
    world_from_camera = torch.stack([frame.world_from_camera for frame in frames])
    intrinsics = torch.stack([frame.intrinsics for frame in frames])
    radius = specification.radius_tensor().squeeze(-1)
    albedo = specification.albedo_tensor()
    rgb, depth = _independent_raster(
        trajectory.positions,
        radius,
        world_from_camera,
        intrinsics,
        albedo,
    )
    timestamps = torch.arange(FRAME_COUNT, dtype=torch.float32) / FRAME_RATE_HZ
    result = _PacketEpisode(
        split=split,
        ordinal=ordinal,
        scene_sha256=scene_signature(specification),
        non_radius_scene_sha256=canonical_sha256(non_radius_metadata),
        primitive_index=specification.primitive_index,
        pair_variant=specification.pair_variant,
        radius_role=specification.radius_role,
        camera_stratum=specification.camera_stratum,
        twin_ordinal=counterfactual_twin_ordinal(ordinal),
        pair_variant_twin_ordinal=pair_variant_twin_ordinal(ordinal),
        rgb=rgb,
        depth=depth,
        timestamps=timestamps,
        world_from_camera=world_from_camera,
        intrinsics=intrinsics,
        position_truth=trajectory.positions,
        velocity_truth=trajectory.velocities,
        radius_truth=radius,
        albedo_truth=albedo,
    )
    expected_shapes = {
        "rgb": (FRAME_COUNT, 3, *IMAGE_SIZE),
        "depth": (FRAME_COUNT, 1, *IMAGE_SIZE),
        "timestamps": (FRAME_COUNT,),
        "world_from_camera": (FRAME_COUNT, 4, 4),
        "intrinsics": (FRAME_COUNT, 3, 3),
        "position_truth": (FRAME_COUNT, 2, 3),
        "velocity_truth": (FRAME_COUNT, 2, 3),
        "radius_truth": (2,),
        "albedo_truth": (2, 3),
    }
    for name, shape in expected_shapes.items():
        tensor = getattr(result, name)
        if tensor.shape != shape or tensor.dtype is not torch.float32:
            raise RuntimeError(f"authorized episode {name} differs from {shape} float32")
        if not bool(torch.isfinite(tensor).all()):
            raise FloatingPointError(f"authorized episode {name} is nonfinite")
    registration = _CAPABILITY_REGISTRY.get(id(token))
    if (
        registration is None
        or registration[3] != "consumed"
        or registration[4] != _ordinal_capability_sha256(token)
    ):
        raise PermissionError("formal episode token was not durably consumed first")
    if id(result) in _EPISODE_REGISTRY:
        raise PermissionError("formal episode identity was already registered")
    _EPISODE_REGISTRY[id(result)] = (
        result,
        manifest,
        batch,
        token,
        split,
        ordinal,
        _episode_digest(result),
    )
    return result


def _packet_for_frame(
    episodes: Sequence[_PacketEpisode],
    frame_index: int,
    *,
    differentiable: bool,
) -> ObservationPacket:
    _validate_episode_batch(episodes)
    if not 0 <= frame_index < HISTORY_FRAME_COUNT:
        raise IndexError(frame_index)
    rgb = torch.stack([episode.rgb[frame_index] for episode in episodes])
    depth = torch.stack([episode.depth[frame_index] for episode in episodes])
    world_from_camera = torch.stack(
        [episode.world_from_camera[frame_index] for episode in episodes]
    )
    intrinsics = torch.stack([episode.intrinsics[frame_index] for episode in episodes])
    if differentiable:
        rgb = rgb.clone().requires_grad_(True)
        depth = depth.clone().requires_grad_(True)
        world_from_camera = world_from_camera.clone().requires_grad_(True)
        intrinsics = intrinsics.clone().requires_grad_(True)
    timestamp_values = [float(episode.timestamps[frame_index]) for episode in episodes]
    if len(set(timestamp_values)) != 1:
        raise RuntimeError("batched scene timestamps differ")
    packet = ObservationPacket(
        modality="rgbd",
        sensor_id="camera0:rgbd",
        timestamp=timestamp_values[0],
        payload={"rgb": rgb, "depth": depth},
        calibration={
            "world_from_camera": world_from_camera,
            "intrinsics": intrinsics,
        },
        frame_id="camera:camera0:rgbd",
        confidence=1.0,
        metadata={
            "image_size": IMAGE_SIZE,
            "depth_semantics": "observable_camera_z_surface_depth_zero_means_no_return",
        },
    )
    _register_packet(
        packet,
        episodes=episodes,
        frame_index=frame_index,
        provenance="nominal_independent_packet",
    )
    return packet


def _physical_mapping(estimate: Tensor, truth: Tensor) -> Tensor:
    """Return physical object index for each of exactly two estimate slots."""

    if estimate.shape != (2, 3) or truth.shape != (2, 3):
        raise ValueError("two-object mapping requires [2,3] estimates and truth")
    identity = (estimate - truth).square().sum()
    swapped = (estimate - truth.flip(0)).square().sum()
    if not bool(torch.isfinite(torch.stack((identity, swapped))).all()):
        raise FloatingPointError("two-object mapping cost is nonfinite")
    return (
        torch.tensor([0, 1], dtype=torch.int64, device=estimate.device)
        if bool(identity <= swapped)
        else torch.tensor([1, 0], dtype=torch.int64, device=estimate.device)
    )


def _gather_physical_by_slot(value: Tensor, physical_by_slot: Tensor) -> Tensor:
    if value.shape[0] != 2 or physical_by_slot.shape != (2,):
        raise ValueError("physical gather requires exactly two slots")
    inverse = torch.empty_like(physical_by_slot)
    inverse[physical_by_slot] = torch.arange(2, device=physical_by_slot.device)
    return value[inverse]


def _tensor_bit_exact(left: Tensor, right: Tensor) -> bool:
    if (
        type(left) is not Tensor
        or type(right) is not Tensor
        or left.shape != right.shape
        or left.dtype != right.dtype
        or left.device != right.device
        or left.layout != right.layout
        or tuple(left.stride()) != tuple(right.stride())
    ):
        return False
    left_digest = hashlib.sha256()
    right_digest = hashlib.sha256()
    _update_tensor_digest(left_digest, "tensor_tree_value", left)
    _update_tensor_digest(right_digest, "tensor_tree_value", right)
    return left_digest.digest() == right_digest.digest()


def _tensor_tree_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, Tensor):
        return _tensor_bit_exact(left, right)
    if isinstance(left, Mapping):
        return list(left) == list(right) and all(
            _tensor_tree_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(
            _tensor_tree_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    if hasattr(left, "__dataclass_fields__"):
        return all(
            _tensor_tree_equal(getattr(left, name), getattr(right, name))
            for name in left.__dataclass_fields__
        )
    return left == right


def _tensor_tree_has_autograd(value: Any) -> bool:
    if isinstance(value, Tensor):
        return value.requires_grad or value.grad_fn is not None
    if isinstance(value, Mapping):
        return any(_tensor_tree_has_autograd(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_tensor_tree_has_autograd(item) for item in value)
    if hasattr(value, "__dataclass_fields__"):
        return any(
            _tensor_tree_has_autograd(getattr(value, name)) for name in value.__dataclass_fields__
        )
    return False


@dataclass(frozen=True, slots=True)
class _LiveMeasurementCapture:
    measurement: MeasurementSet
    measurement_identity: int
    call_index: int


def _record_live_measurement(
    captures: list[_LiveMeasurementCapture],
    measured: MeasurementSet,
) -> MeasurementSet:
    if type(captures) is not list or any(
        type(capture) is not _LiveMeasurementCapture for capture in captures
    ):
        raise TypeError("live measurement capture ledger must be one exact list")
    if type(measured) is not MeasurementSet:
        raise TypeError("correct measured input must be one exact MeasurementSet")
    measured.validate()
    captures.append(
        _LiveMeasurementCapture(
            measurement=measured,
            measurement_identity=id(measured),
            call_index=len(captures),
        )
    )
    return measured


def _validated_live_measurement_capture(
    capture: _LiveMeasurementCapture,
    public_measurement: MeasurementSet,
    *,
    expected_call_index: int,
) -> MeasurementSet:
    if (
        type(capture) is not _LiveMeasurementCapture
        or type(public_measurement) is not MeasurementSet
        or type(expected_call_index) is not int
        or expected_call_index < 0
        or type(capture.measurement) is not MeasurementSet
        or type(capture.measurement_identity) is not int
        or capture.measurement_identity != id(capture.measurement)
        or type(capture.call_index) is not int
        or capture.call_index != expected_call_index
        or capture.measurement is public_measurement
    ):
        raise PermissionError("live measurement capture identity or call index differs")
    capture.measurement.validate()
    public_measurement.validate()
    detached = capture.measurement.detach()
    if _tensor_tree_has_autograd(public_measurement):
        raise RuntimeError("public measurement diagnostic was not recursively detached")
    if not _tensor_tree_equal(detached, public_measurement):
        raise RuntimeError("live measurement capture differs bit-exactly from public diagnostic")
    return capture.measurement


def _persistent_runtime_tensor_bytes(model: OnlineWorldModel) -> int:
    seen: set[int] = set()
    total = 0

    def visit(value: Any) -> None:
        nonlocal total
        if isinstance(value, Tensor):
            pointer = value.untyped_storage().data_ptr()
            if pointer not in seen:
                seen.add(pointer)
                total += value.untyped_storage().nbytes()
            return
        if isinstance(value, Mapping):
            for item in value.values():
                visit(item)
            return
        if isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
            return
        if hasattr(value, "__dataclass_fields__"):
            for name in value.__dataclass_fields__:
                visit(getattr(value, name))

    visit(model.state)
    return total


def _storage_or_object_alias(left: Tensor, right: Tensor) -> bool:
    if not isinstance(left, Tensor) or not isinstance(right, Tensor):
        raise TypeError("rollout alias audit requires exact tensor operands")
    return bool(
        left is right
        or (
            left.numel() > 0
            and right.numel() > 0
            and left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
        )
    )


def _rollout_output_alias_count(trajectory: Any, belief: Any) -> int:
    pairs = (
        (trajectory.positions, belief.objects.position),
        (trajectory.positions, belief.objects.velocity),
        (trajectory.velocities, belief.objects.position),
        (trajectory.velocities, belief.objects.velocity),
        (trajectory.active_mask, belief.objects.active),
        (trajectory.timestamps, belief.timestamp),
    )
    return sum(_storage_or_object_alias(left, right) for left, right in pairs)


def _process_max_rss_bytes() -> float:
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if os.uname().sysname == "Darwin" else raw * 1024.0


def _model_state_sha256(model: OnlineWorldModel) -> str:
    state = model.state_dict()
    if len(state) != 0:
        raise RuntimeError("qualified model state ceased to be empty")
    return canonical_sha256([])


EMPTY_MODEL_STATE_SHA256 = canonical_sha256([])


def _new_strict_runtime(
    config: OrpheusConfig,
    *,
    strict_profile: bool,
    reviewed_state: Mapping[str, Tensor] | None,
    expected_state_sha256: str | None,
) -> OnlineWorldModel:
    model = new_public_model(config) if strict_profile else OnlineWorldModel.from_config(config)
    if tuple(model.parameters()) or tuple(model.buffers()) or len(model.state_dict()) != 0:
        raise RuntimeError("qualified runtime acquired persistent module state")
    if reviewed_state is None:
        if expected_state_sha256 is not None:
            raise ValueError("state hash cannot be supplied without reviewed state")
        return model
    if type(reviewed_state) is not dict or reviewed_state:
        raise ValueError("reviewed variable-radius model state must be one exact empty dict")
    expected = validated_sha256(
        expected_state_sha256,
        label="reviewed empty model state",
    )
    if expected != EMPTY_MODEL_STATE_SHA256:
        raise ValueError("reviewed model-state hash differs from frozen empty state")
    result = model.load_state_dict(reviewed_state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("strict reviewed empty-state load changed model schema")
    if _model_state_sha256(model) != expected:
        raise RuntimeError("strict reviewed state load changed the empty-state hash")
    return model


def _measurement_physical_mapping(measurement: Any, truth: Tensor, batch_index: int) -> Tensor:
    positions = measurement.auxiliary.get("world_position")
    if not isinstance(positions, Tensor) or positions.shape[1:] != (2, 3):
        raise RuntimeError("runtime did not expose exactly two public metric positions")
    return _physical_mapping(positions[batch_index], truth)


def _validated_vjp_anchor_targets(
    *,
    raw_anchor_vjp: Tensor,
    deployed_anchor_vjp: Tensor,
) -> dict[str, Tensor]:
    targets = {
        "raw": raw_anchor_vjp,
        "deployed": deployed_anchor_vjp,
    }
    for stage, target in targets.items():
        if type(target) is not Tensor or target.shape != (BATCH_SIZE, 2):
            raise RuntimeError(f"{stage} VJP anchor must be one exact [4,2] Tensor")
        if target.dtype is not torch.float32 or not bool(torch.isfinite(target).all()):
            raise RuntimeError(f"{stage} VJP anchor must be finite float32")
        if target.requires_grad is not True or target.grad_fn is None:
            raise RuntimeError(f"{stage} VJP anchor lost its live autograd graph")
    return targets


def _vjp_metrics(
    *,
    packets: Sequence[ObservationPacket],
    raw_anchor_vjp: Tensor,
    deployed_anchor_vjp: Tensor,
    scene_sha256s: Sequence[str],
    audit_indices: tuple[int, ...],
) -> dict[str, float]:
    if (
        len(packets) != HISTORY_FRAME_COUNT
        or len(scene_sha256s) != BATCH_SIZE
        or len(set(scene_sha256s)) != BATCH_SIZE
        or type(audit_indices) is not tuple
        or not audit_indices
        or any(type(index) is not int for index in audit_indices)
        or tuple(sorted(set(audit_indices))) != audit_indices
        or any(not 0 <= index < BATCH_SIZE for index in audit_indices)
        or len({scene_sha256s[index] for index in audit_indices}) != len(audit_indices)
    ):
        raise RuntimeError("VJP audit requires exact unique target indices in one B=4 history")
    targets = _validated_vjp_anchor_targets(
        raw_anchor_vjp=raw_anchor_vjp,
        deployed_anchor_vjp=deployed_anchor_vjp,
    )
    inputs: dict[str, tuple[Tensor, ...]] = {
        "rgb": tuple(packet.payload["rgb"] for packet in packets),
        "depth": tuple(packet.payload["depth"] for packet in packets),
        "intrinsics": tuple(packet.calibration["intrinsics"] for packet in packets),
        "world_from_camera": tuple(packet.calibration["world_from_camera"] for packet in packets),
    }
    if any(not tensor.requires_grad for values in inputs.values() for tensor in values):
        raise RuntimeError("VJP audit sources must retain differentiable packet tensors")

    aggregate: dict[str, list[float]] = {}
    gradient_cache: dict[tuple[str, int, int, str], tuple[Tensor, ...]] = {}
    started = time.perf_counter()
    losses = [
        (stage, batch_index, object_index)
        for stage in VJP_STAGES
        for batch_index in audit_indices
        for object_index in OBJECT_INDICES
    ]
    all_sources = tuple(
        tensor for modality in (*VJP_MODALITIES, "world_from_camera") for tensor in inputs[modality]
    )
    for loss_index, (stage, batch_index, object_index) in enumerate(losses):
        output = targets[stage][batch_index, object_index]
        gradients = torch.autograd.grad(
            output,
            all_sources,
            retain_graph=loss_index + 1 < len(losses),
            allow_unused=True,
        )
        offset = 0
        for modality in (*VJP_MODALITIES, "world_from_camera"):
            source_values = inputs[modality]
            resolved: list[Tensor] = []
            for source in source_values:
                gradient = gradients[offset]
                offset += 1
                value = torch.zeros_like(source) if gradient is None else gradient
                if not bool(torch.isfinite(value).all()):
                    raise FloatingPointError("VJP audit produced nonfinite gradients")
                resolved.append(value)
            gradient_cache[(stage, batch_index, object_index, modality)] = tuple(resolved)
            per_frame_target = torch.stack(
                [value[batch_index].abs().reshape(-1).sum() for value in resolved]
            )
            per_scene = torch.stack(
                [
                    sum(
                        (value[scene_index].abs().reshape(-1).sum() for value in resolved),
                        start=resolved[0].new_zeros(()),
                    )
                    for scene_index in range(BATCH_SIZE)
                ]
            )
            cross = torch.cat((per_scene[:batch_index], per_scene[batch_index + 1 :]))
            if modality == "world_from_camera":
                key = f"world_from_camera_gradient_l1/{stage}/object_{object_index}"
                aggregate.setdefault(key, []).append(float(per_scene[batch_index]))
                continue
            prefix = f"{stage}/object_{object_index}/{modality}"
            values = {
                f"gradient_l1/{prefix}": float(per_scene[batch_index]),
                f"gradient_max_l1/{prefix}": float(per_scene[batch_index]),
                f"gradient_anchor_history_frame_l1/{prefix}": float(
                    per_frame_target[ANCHOR_FRAME_INDEX]
                ),
                f"gradient_nonanchor_max_history_frame_l1/{prefix}": float(
                    torch.cat(
                        (
                            per_frame_target[:ANCHOR_FRAME_INDEX],
                            per_frame_target[ANCHOR_FRAME_INDEX + 1 :],
                        )
                    ).max()
                ),
                f"gradient_supported_history_frames/{prefix}": float(
                    (
                        per_frame_target
                        >= (
                            DEFAULT_GATES.minimum_rgb_radius_gradient_l1
                            if modality == "rgb"
                            else DEFAULT_GATES.minimum_depth_intrinsics_radius_gradient_l1
                        )
                    ).sum()
                ),
                f"gradient_cross_scene_max_l1/{prefix}": float(cross.max()),
            }
            for key, value in values.items():
                aggregate.setdefault(key, []).append(value)

    metrics: dict[str, float] = {}
    for key, values in aggregate.items():
        if key.startswith(
            (
                "gradient_l1/",
                "gradient_anchor_history_frame_l1/",
                "gradient_supported_history_frames/",
            )
        ):
            metrics[key] = float(min(values))
        else:
            metrics[key] = float(max(values))

    for object_index in OBJECT_INDICES:
        for modality in VJP_MODALITIES:
            maximum_difference = 0.0
            support_mismatch = 0
            for batch_index in audit_indices:
                raw = gradient_cache[("raw", batch_index, object_index, modality)]
                deployed = gradient_cache[("deployed", batch_index, object_index, modality)]
                for raw_frame, deployed_frame in zip(raw, deployed, strict=True):
                    maximum_difference = max(
                        maximum_difference,
                        float((raw_frame - deployed_frame).abs().max()),
                    )
                    raw_support = raw_frame != 0.0
                    deployed_support = deployed_frame != 0.0
                    support_mismatch += int((raw_support != deployed_support).sum())
            prefix = f"object_{object_index}/{modality}"
            metrics[f"gradient_raw_deployed_max_abs_difference/{prefix}"] = float(
                maximum_difference
            )
            metrics[f"gradient_raw_deployed_support_mismatch_count/{prefix}"] = float(
                support_mismatch
            )

    metrics.update(
        {
            "gradient_vector_count": 12.0,
            "gradient_audit_scene_count": float(len(audit_indices)),
            "gradient_audit_unique_scene_fraction": 1.0,
            "gradient_audit_primitive_coverage_fraction": 0.0,
            "gradient_audit_pair_variant_coverage_fraction": 0.0,
            "gradient_audit_radius_role_coverage_fraction": 0.0,
            "gradient_audit_camera_stratum_coverage_fraction": 0.0,
            "vjp_latency_seconds": float(time.perf_counter() - started),
        }
    )
    expected = set(_vjp_metric_names())
    if set(metrics) != expected:
        raise RuntimeError(
            f"VJP metric schema differs: missing={sorted(expected - set(metrics))}, "
            f"extra={sorted(set(metrics) - expected)}"
        )
    return metrics


def _merge_vjp_metrics(
    parts: Sequence[Mapping[str, float]],
    records: Sequence[tuple[int, str, int, int, int, int]],
) -> dict[str, float]:
    if type(parts) not in {list, tuple} or len(parts) != len(VJP_AUDIT_ORDINALS):
        raise RuntimeError("VJP reducer requires one partial for each audited ordinal")
    if type(records) not in {list, tuple} or len(records) != len(VJP_AUDIT_ORDINALS):
        raise RuntimeError("VJP reducer requires eight exact scene receipts")
    expected_schema = set(_vjp_metric_names())
    for part in parts:
        if type(part) is not dict or set(part) != expected_schema:
            raise RuntimeError("VJP partial metric schema differs")
        if (
            part["gradient_vector_count"] != 12.0
            or part["gradient_audit_scene_count"] != 1.0
            or part["gradient_audit_unique_scene_fraction"] != 1.0
        ):
            raise RuntimeError("VJP partial topology denominator differs")
    if tuple(record[0] for record in records) != VJP_AUDIT_ORDINALS:
        raise RuntimeError("VJP scene receipts differ from predeclared ordinal order")
    hashes = tuple(
        validated_sha256(record[1], label="VJP audit scene receipt") for record in records
    )
    if len(set(hashes)) != len(VJP_AUDIT_ORDINALS):
        raise RuntimeError("VJP audit scene receipt hashes are not exactly unique")
    for ordinal, _, primitive, pair_variant, radius_role, camera_stratum in records:
        if (
            primitive != ordinal // 32
            or pair_variant != (ordinal % 32) // 16
            or radius_role != (ordinal % 16) // 8
            or camera_stratum != ordinal % 8
        ):
            raise RuntimeError("VJP scene receipt axis binding differs")

    minimum_prefixes = (
        "gradient_l1/",
        "gradient_anchor_history_frame_l1/",
        "gradient_supported_history_frames/",
    )
    summed_suffixes = ("gradient_raw_deployed_support_mismatch_count/",)
    handled = {
        "gradient_vector_count",
        "gradient_audit_scene_count",
        "gradient_audit_unique_scene_fraction",
        "gradient_audit_primitive_coverage_fraction",
        "gradient_audit_pair_variant_coverage_fraction",
        "gradient_audit_radius_role_coverage_fraction",
        "gradient_audit_camera_stratum_coverage_fraction",
        "vjp_latency_seconds",
    }
    result: dict[str, float] = {}
    for key in sorted(expected_schema - handled):
        values = [part[key] for part in parts]
        if key.startswith(minimum_prefixes):
            result[key] = float(min(values))
        elif key.startswith(summed_suffixes):
            result[key] = float(sum(values))
        else:
            result[key] = float(max(values))
    result.update(
        {
            "gradient_vector_count": 12.0,
            "gradient_audit_scene_count": float(len(records)),
            "gradient_audit_unique_scene_fraction": float(
                len(set(hashes)) / len(VJP_AUDIT_ORDINALS)
            ),
            "gradient_audit_primitive_coverage_fraction": float(
                len({record[2] for record in records}) / PRIMITIVES_PER_SPLIT
            ),
            "gradient_audit_pair_variant_coverage_fraction": float(
                len({record[3] for record in records}) / PAIR_VARIANTS_PER_PRIMITIVE
            ),
            "gradient_audit_radius_role_coverage_fraction": float(
                len({record[4] for record in records}) / RADIUS_ROLES_PER_PRIMITIVE
            ),
            "gradient_audit_camera_stratum_coverage_fraction": float(
                len({record[5] for record in records}) / CAMERA_STRATA
            ),
            "vjp_latency_seconds": float(sum(part["vjp_latency_seconds"] for part in parts)),
        }
    )
    if set(result) != expected_schema:
        raise RuntimeError("merged VJP metric schema differs")
    return result


def _split_result(
    *,
    split: str,
    metrics: Mapping[str, float],
    model_state_sha256: str,
    evidence_sha256: str,
    provenance_sha256: str,
) -> dict[str, Any]:
    failures = gate_failures(metrics)
    result = {
        "split": split,
        "manifest": list(_manifest_rows(split)),
        "manifest_sha256": MANIFEST_SHA256[split],
        "metrics": dict(metrics),
        "failures": failures,
        "passed": not failures,
        "materialization_started": True,
        "access_completed": True,
        "outcome": "passed" if not failures else "gate_failed",
        "optimizer_updates": 0,
        "runtime_api": {
            "packet_type": "ObservationPacket",
            "ingest_frames": list(HISTORY_FRAME_INDICES),
            "rollout_method": "OnlineWorldModel.predict",
            "horizons_seconds": list(HORIZONS_SECONDS),
            "model_visible_metadata": [
                "image_size",
                "depth_semantics",
            ],
        },
        "scene_materializer": "private_receipt_owned_independent_packet_synthesis",
        "model_state_sha256": validated_sha256(
            model_state_sha256,
            label=f"{split} model state",
        ),
        "evidence_sha256": validated_sha256(
            evidence_sha256,
            label=f"{split} sufficient evidence",
        ),
        "provenance_sha256": validated_sha256(
            provenance_sha256,
            label=f"{split} evaluator provenance",
        ),
    }
    expected = {
        "split",
        "manifest",
        "manifest_sha256",
        "metrics",
        "failures",
        "passed",
        "materialization_started",
        "access_completed",
        "outcome",
        "optimizer_updates",
        "runtime_api",
        "scene_materializer",
        "model_state_sha256",
        "evidence_sha256",
        "provenance_sha256",
    }
    if set(result) != expected:
        raise RuntimeError("split result schema differs")
    if result["model_state_sha256"] != EMPTY_MODEL_STATE_SHA256:
        raise RuntimeError("split result is not bound to exact empty model state")
    _validate_manifest_rows(split, result["manifest"])
    if type(result["failures"]) is not list or type(result["passed"]) is not bool:
        raise TypeError("split result pass/failure fields have wrong exact types")
    return result


def _collect_manifest_once(
    config: OrpheusConfig,
    *,
    split: str,
    manifest: _ManifestCapability,
    reviewed_state: Mapping[str, Tensor] | None,
    expected_state_sha256: str | None,
    boundary_guard: Callable[[str], None],
) -> tuple[
    list[SceneEvidence],
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
]:
    """Consume 0..63 once in sixteen receipt-owned four-scene batches."""

    if not callable(boundary_guard):
        raise TypeError("manifest collector requires a callable boundary guard")
    if manifest.split != split:
        raise PermissionError("collector split differs from manifest owner")
    rows: list[SceneEvidence] = []
    vjp_parts: list[dict[str, float]] = []
    vjp_records: list[tuple[int, str, int, int, int, int]] = []
    prior_totals: dict[str, float] = {}
    legacy: dict[str, float] | None = None
    adversarial: dict[str, float] | None = None
    perception_seconds = 0.0
    rollout_seconds = 0.0
    persistent_bytes = 0.0
    process_rss = 0.0
    rss_delta = 0.0

    boundary_guard(f"{split} manifest before access")
    for start in range(0, SCENES_PER_SPLIT, BATCH_SIZE):
        boundary_guard(f"{split} batch {start // BATCH_SIZE} before access")
        ordinals = (start, start + 1, start + 2, start + 3)
        batch = manifest.begin_batch(ordinals)
        episodes = [
            _materialise_authorized_episode(
                split=split,
                ordinal=ordinal,
                manifest=manifest,
                batch=batch,
                token=token,
            )
            for ordinal, token in zip(batch.ordinals, batch.tokens, strict=True)
        ]
        vjp_audit_indices = tuple(
            index for index, ordinal in enumerate(ordinals) if ordinal in VJP_AUDIT_ORDINALS
        )
        batch_rows, batch_vjp, audit = _evaluate_nominal_batch(
            config,
            episodes,
            vjp_audit_indices=vjp_audit_indices,
            reviewed_state=reviewed_state,
            expected_state_sha256=expected_state_sha256,
        )
        if [row.ordinal for row in batch_rows] != list(ordinals):
            raise RuntimeError("nominal evaluator returned reordered evidence")
        batch_prior = _alternate_prior_metrics(
            config,
            audit["packets"],
            audit,
            reviewed_state,
            expected_state_sha256,
        )
        for key, value in batch_prior.items():
            if key == "alternate_prior_count":
                if value != 2.0:
                    raise RuntimeError("alternate-prior audit count changed")
                continue
            prior_totals[key] = prior_totals.get(key, 0.0) + value
        if vjp_audit_indices:
            if not batch_vjp or len(audit["vjp_scene_receipts"]) != 1:
                raise RuntimeError("predeclared VJP ordinal omitted its exact partial receipt")
            vjp_parts.append(batch_vjp)
            vjp_records.extend(audit["vjp_scene_receipts"])
        elif batch_vjp or audit["vjp_scene_receipts"]:
            raise RuntimeError("VJP evidence was produced outside predeclared ordinals")
        if start == 0:
            legacy = _legacy_control_metrics(
                config,
                audit["packets"],
                reviewed_state=reviewed_state,
                expected_state_sha256=expected_state_sha256,
            )
            adversarial = _adversarial_metrics(
                config,
                audit["packets"],
                reviewed_state=reviewed_state,
                expected_state_sha256=expected_state_sha256,
            )
        perception_seconds += audit["perception_latency_seconds"]
        rollout_seconds += audit["state_only_rollout_latency_seconds"]
        persistent_bytes = max(
            persistent_bytes,
            audit["persistent_runtime_tensor_state_bytes"],
        )
        process_rss = max(
            process_rss,
            max(_diagnostics(row)["process_max_rss_bytes"] for row in batch_rows),
        )
        rss_delta = max(rss_delta, audit["process_rss_delta_bytes"])
        batch_digest = _evidence_sha256(batch_rows)
        _retire_derived_batch_packets(episodes, audit["packets"])
        _finalize_batch_provenance(
            episodes,
            batch_rows,
            audit["packets"],
        )
        manifest.complete_batch(batch, result_sha256=batch_digest)
        rows.extend(batch_rows)
        boundary_guard(f"{split} batch {start // BATCH_SIZE} after access")

    if (
        len(rows) != SCENES_PER_SPLIT
        or [row.ordinal for row in rows] != list(ORDINALS)
        or legacy is None
        or adversarial is None
    ):
        raise RuntimeError("manifest collector did not produce one complete ordered cache")
    vjp = _merge_vjp_metrics(vjp_parts, vjp_records)
    prior = {**prior_totals, "alternate_prior_count": 2.0}
    resources = {
        "perception_latency_seconds": float(perception_seconds),
        "state_only_rollout_latency_seconds": float(rollout_seconds),
        "persistent_runtime_tensor_state_bytes_max": float(persistent_bytes),
        "process_max_rss_bytes": float(process_rss),
        "process_rss_delta_bytes": float(rss_delta),
    }
    return rows, vjp, prior, legacy, adversarial, resources


def _evaluate_manifest_once(
    config: OrpheusConfig,
    *,
    split: str,
    manifest: _ManifestCapability,
    reviewed_state: Mapping[str, Tensor] | None,
    expected_state_sha256: str | None,
    boundary_guard: Callable[[str], None],
) -> dict[str, Any]:
    rows, vjp, prior, legacy, adversarial, resources = _collect_manifest_once(
        config,
        split=split,
        manifest=manifest,
        reviewed_state=reviewed_state,
        expected_state_sha256=expected_state_sha256,
        boundary_guard=boundary_guard,
    )
    metrics = _aggregate_split_metrics(
        rows,
        vjp=vjp,
        prior=prior,
        legacy=legacy,
        adversarial=adversarial,
        resources=resources,
    )
    result = _split_result(
        split=split,
        metrics=metrics,
        model_state_sha256=EMPTY_MODEL_STATE_SHA256,
        evidence_sha256=_evidence_sha256(rows),
        provenance_sha256=_evidence_provenance_sha256(rows),
    )
    manifest.close(result)
    return result


CHECKPOINT_SCHEMA = frozenset(
    {
        "artifact_kind",
        "architecture_version",
        "architecture_attempt",
        "model_state",
        "model_state_sha256",
        "optimizer_state",
        "scheduler_state",
        "optimizer_updates",
        "resolved_config_sha256",
        "protocol_sha256",
        "scene_certificate",
        "development_result",
        "source_provenance",
        "project_version",
        "specification_version",
        "simulator_version",
        "device",
        "precision",
    }
)
REPORT_BASE_SCHEMA = frozenset(
    {
        "artifact_kind",
        "stage",
        "architecture_version",
        "architecture_attempt",
        "protocol",
        "resolved_config_sha256",
        "scene_certificate",
        "source_provenance",
        "manifest_sha256",
        "results",
        "model_state_sha256",
        "optimizer_updates",
        "optimizer_state_entry_count",
        "rng_state_entry_count",
        "passed",
        "terminal_ledger_sha256",
        "opened_splits",
        "stopped_after",
        "materialization_started",
        "access_completed",
        "outcome",
        "error",
    }
)
DEVELOPMENT_REPORT_SCHEMA = frozenset({*REPORT_BASE_SCHEMA, "checkpoint"})
QUALIFICATION_REPORT_SCHEMA = frozenset({*REPORT_BASE_SCHEMA, "reviewed_development"})


def _require_exact_keys(value: Any, expected: frozenset[str], *, label: str) -> None:
    if type(value) is not dict:
        raise TypeError(f"{label} must be an exact dict")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} schema differs: missing={sorted(expected - actual)!r}, "
            f"extra={sorted(actual - expected)!r}"
        )


def _checkpoint_bytes(
    *,
    config: OrpheusConfig,
    development_result: Mapping[str, Any],
    source_provenance: Mapping[str, Any],
) -> bytes:
    assert_rgbd_variable_radius_config(config)
    if type(development_result) is not dict or development_result.get("passed") is not True:
        raise ValueError("checkpoint requires passed development evidence")
    source = _validated_published_source(
        dict(source_provenance),
        label="checkpoint source",
    )
    protocol = bridge_protocol()
    payload = {
        "artifact_kind": "rgbd_variable_radius_empty_state_checkpoint",
        "architecture_version": ARCHITECTURE_VERSION,
        "architecture_attempt": ARCHITECTURE_ATTEMPT,
        "model_state": {},
        "model_state_sha256": EMPTY_MODEL_STATE_SHA256,
        "optimizer_state": None,
        "scheduler_state": None,
        "optimizer_updates": 0,
        "resolved_config_sha256": FROZEN_CONFIG_SHA256,
        "protocol_sha256": protocol["protocol_sha256"],
        "scene_certificate": _frozen_scene_certificate_binding(),
        "development_result": copy.deepcopy(dict(development_result)),
        "source_provenance": source,
        "project_version": __version__,
        "specification_version": SPECIFICATION_VERSION,
        "simulator_version": SIMULATOR_VERSION,
        "device": "cpu",
        "precision": "float32",
    }
    _require_exact_keys(payload, CHECKPOINT_SCHEMA, label="checkpoint payload")
    stream = io.BytesIO()
    torch.save(payload, stream)
    contents = stream.getvalue()
    if not contents or len(contents) > MAX_CHECKPOINT_BYTES:
        raise RuntimeError("empty-state checkpoint exceeds its fixed byte budget")
    _validate_checkpoint_payload(contents, expected_source=source)
    return contents


def _load_checkpoint_payload(contents: bytes) -> Mapping[str, Any]:
    if type(contents) is not bytes or not contents or len(contents) > MAX_CHECKPOINT_BYTES:
        raise ValueError("checkpoint bytes are empty, oversized, or not exact bytes")
    value = torch.load(io.BytesIO(contents), map_location="cpu", weights_only=True)
    if type(value) is not dict:
        raise TypeError("checkpoint root must be an exact dict")
    _require_exact_keys(value, CHECKPOINT_SCHEMA, label="checkpoint payload")
    if type(value["model_state"]) is not dict or value["model_state"]:
        raise ValueError("checkpoint model_state must be one exact empty dict")
    if value["optimizer_state"] is not None or value["scheduler_state"] is not None:
        raise ValueError("checkpoint optimizer and scheduler states must be exact nulls")
    return value


def _validate_checkpoint_payload(
    contents: bytes,
    *,
    expected_source: Mapping[str, Any],
) -> dict[str, Any]:
    payload = dict(_load_checkpoint_payload(contents))
    _require_exact_keys(payload, CHECKPOINT_SCHEMA, label="checkpoint payload")
    expected_scalars = {
        "artifact_kind": "rgbd_variable_radius_empty_state_checkpoint",
        "architecture_version": ARCHITECTURE_VERSION,
        "architecture_attempt": ARCHITECTURE_ATTEMPT,
        "model_state_sha256": EMPTY_MODEL_STATE_SHA256,
        "optimizer_state": None,
        "scheduler_state": None,
        "optimizer_updates": 0,
        "resolved_config_sha256": FROZEN_CONFIG_SHA256,
        "protocol_sha256": bridge_protocol()["protocol_sha256"],
        "scene_certificate": _frozen_scene_certificate_binding(),
        "project_version": __version__,
        "specification_version": SPECIFICATION_VERSION,
        "simulator_version": SIMULATOR_VERSION,
        "device": "cpu",
        "precision": "float32",
    }
    for name, expected in expected_scalars.items():
        _exact_equal(payload[name], expected, label=f"checkpoint.{name}")
    if type(payload["model_state"]) is not dict or payload["model_state"]:
        raise ValueError("checkpoint model_state must be one exact empty dict")
    if type(payload["development_result"]) is not dict:
        raise ValueError("checkpoint development result is not passed empty-state evidence")
    validated_development_result = _validated_split_result(
        payload["development_result"],
        split="development",
    )
    _exact_equal(
        payload["development_result"],
        validated_development_result,
        label="checkpoint development result",
    )
    if (
        validated_development_result["passed"] is not True
        or validated_development_result["model_state_sha256"] != EMPTY_MODEL_STATE_SHA256
    ):
        raise ValueError("checkpoint development result is not passed empty-state evidence")
    source = _validated_published_source(
        payload["source_provenance"],
        label="checkpoint source",
    )
    expected = _validated_published_source(
        dict(expected_source),
        label="expected checkpoint source",
    )
    _exact_equal(source, expected, label="checkpoint source provenance")
    return payload


def _save_review_checkpoint(
    path: Path,
    *,
    directory_pin: _PinnedDirectory,
    config: OrpheusConfig,
    development_result: Mapping[str, Any],
    source_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    _require_canonical_path(
        path,
        canonical_checkpoint_path(),
        label="checkpoint path",
        directory_pin=directory_pin,
    )
    contents = _checkpoint_bytes(
        config=config,
        development_result=development_result,
        source_provenance=source_provenance,
    )
    _pinned_durable_create(directory_pin, path, contents)
    metadata = _pinned_require_single_link_regular(
        directory_pin,
        path,
        label="development checkpoint",
    )
    reread = _pinned_stable_read_bytes(
        directory_pin,
        path,
        label="development checkpoint",
    )
    if reread != contents:
        raise RuntimeError("checkpoint bytes changed during publication")
    return {
        "path": str(path.relative_to(REPOSITORY_ROOT)),
        "sha256": sha256_bytes(contents),
        "bytes": float(metadata.st_size),
        "model_state_sha256": EMPTY_MODEL_STATE_SHA256,
        "model_state_entry_count": 0,
        "optimizer_state_entry_count": 0,
        "rng_state_entry_count": 0,
    }


def _strict_json_loads(contents: bytes, *, label: str) -> dict[str, Any]:
    if type(contents) is not bytes:
        raise TypeError(f"{label} contents must be exact bytes")

    def reject_constant(value: str) -> None:
        raise ValueError(f"{label} contains nonfinite JSON constant {value!r}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    value = json.loads(
        contents.decode("utf-8", errors="strict"),
        parse_constant=reject_constant,
        object_pairs_hook=unique_object,
    )
    if type(value) is not dict:
        raise TypeError(f"{label} root must be an exact object")
    return value


def _validate_terminal_ledger(
    contents: bytes,
    *,
    stage: Literal["development", "qualification"],
    bindings: Mapping[str, Any],
    expected_outcome: Literal["passed", "gate_failed", "error"] = "passed",
    expected_opened_splits: Sequence[str] | None = None,
    expected_error: Mapping[str, str] | None = None,
    expected_report_sha256: str | None = None,
    expected_results: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    record = _strict_json_loads(contents, label=f"{stage} terminal ledger")
    supplied_hash = record.pop("record_sha256", None)
    if supplied_hash != canonical_sha256(record):
        raise ValueError(f"{stage} terminal ledger self-hash differs")
    expected_order = (
        ["development"] if stage == "development" else ["selector", "confirmation", "final_test"]
    )
    expected_keys = {
        "artifact_kind",
        "architecture_attempt",
        "stage",
        "order",
        "bindings",
        "batch_size",
        "scenes_per_split",
        "splits",
        "attempt_reserved",
        "status",
        "generation",
    }
    if expected_outcome == "error":
        expected_keys.add("error")
    if set(record) != expected_keys:
        raise ValueError(f"{stage} terminal ledger schema differs")
    if expected_outcome not in {"passed", "gate_failed", "error"}:
        raise ValueError("terminal ledger expected outcome differs")
    if expected_opened_splits is None:
        if expected_outcome == "passed":
            opened_splits = expected_order
        else:
            raise ValueError("non-passed terminal ledger requires its exact opened prefix")
    else:
        if type(expected_opened_splits) not in {list, tuple} or any(
            type(split) is not str for split in expected_opened_splits
        ):
            raise TypeError("terminal ledger opened splits must be an exact sequence of strings")
        opened_splits = list(expected_opened_splits)
    if opened_splits != expected_order[: len(opened_splits)]:
        raise ValueError(f"{stage} terminal ledger opened split prefix differs")
    if expected_outcome == "passed" and opened_splits != expected_order:
        raise ValueError(f"{stage} passed ledger must open every exact split")
    if expected_outcome == "gate_failed" and not opened_splits:
        raise ValueError(f"{stage} gate-failed ledger requires one opened split")
    if expected_results is None:
        if expected_outcome != "error":
            raise ValueError("passed/gate-failed terminal ledger requires exact results")
        result_values: list[Mapping[str, Any]] = []
    else:
        if type(expected_results) not in {list, tuple} or any(
            type(result) is not dict for result in expected_results
        ):
            raise TypeError("terminal ledger expected results must be exact dicts")
        result_values = list(expected_results)
    if len(result_values) > len(opened_splits):
        raise ValueError("terminal ledger has fewer opened splits than exact results")
    for split, result in zip(opened_splits, result_values, strict=False):
        if result.get("split") != split or type(result.get("passed")) is not bool:
            raise ValueError("terminal ledger expected result prefix differs")
    if expected_outcome in {"passed", "gate_failed"} and len(result_values) != len(opened_splits):
        raise ValueError("terminal ledger normal outcome requires every exact split result")

    expected_status = {
        "passed": "complete_passed",
        "gate_failed": "complete_failed",
        "error": "terminal_error",
    }[expected_outcome]
    expected_bindings = {
        "artifact_kind": "rgbd_variable_radius_exactly_once_access_ledger",
        "architecture_attempt": ARCHITECTURE_ATTEMPT,
        "stage": stage,
        "order": expected_order,
        "bindings": dict(bindings),
        "batch_size": BATCH_SIZE,
        "scenes_per_split": SCENES_PER_SPLIT,
        "attempt_reserved": True,
        "status": expected_status,
    }
    for name, expected in expected_bindings.items():
        _exact_equal(
            record[name],
            expected,
            label=f"{stage} terminal ledger.{name}",
        )
    if type(record["splits"]) is not dict or set(record["splits"]) != set(expected_order):
        raise ValueError(f"{stage} terminal ledger split inventory differs")

    state_schema = {
        "status",
        "access_started",
        "next_ordinal",
        "active_batch",
        "batch_result_sha256",
        "split_result_sha256",
    }

    def validate_digests(state: Mapping[str, Any], *, split: str) -> None:
        values = state["batch_result_sha256"]
        if type(values) is not list:
            raise TypeError(f"{stage} terminal ledger {split} batch digests differ")
        for value in values:
            validated_sha256(value, label=f"{stage} terminal ledger {split} batch digest")
        if state["split_result_sha256"] is not None:
            validated_sha256(
                state["split_result_sha256"],
                label=f"{stage} terminal ledger {split} result digest",
            )

    transition_count = 0
    for index, split in enumerate(expected_order):
        state = record["splits"][split]
        if type(state) is not dict or set(state) != state_schema:
            raise ValueError(f"{stage} terminal ledger {split} schema differs")
        validate_digests(state, split=split)
        is_opened = index < len(opened_splits)
        if not is_opened:
            _exact_equal(
                state,
                {
                    "status": "unopened",
                    "access_started": False,
                    "next_ordinal": 0,
                    "active_batch": None,
                    "batch_result_sha256": [],
                    "split_result_sha256": None,
                },
                label=f"{stage} terminal ledger unopened {split}",
            )
            continue

        if expected_outcome in {"passed", "gate_failed"}:
            expected_split_status = (
                "failed"
                if expected_outcome == "gate_failed" and index == len(opened_splits) - 1
                else "passed"
            )
            if (
                state["status"] != expected_split_status
                or state["access_started"] is not True
                or type(state["next_ordinal"]) is not int
                or state["next_ordinal"] != SCENES_PER_SPLIT
                or state["active_batch"] is not None
                or len(state["batch_result_sha256"]) != SCENES_PER_SPLIT // BATCH_SIZE
                or state["split_result_sha256"] is None
            ):
                raise ValueError(f"{stage} terminal ledger {split} state differs")
            expected_result = result_values[index]
            if expected_result["passed"] is not (expected_split_status == "passed"):
                raise ValueError(f"{stage} terminal ledger {split} pass binding differs")
            if state["split_result_sha256"] != canonical_sha256(expected_result):
                raise ValueError(f"{stage} terminal ledger {split} result digest differs")
            transition_count += 34
            continue

        status = state["status"]
        if index < len(opened_splits) - 1 and status != "passed":
            raise ValueError(f"{stage} error ledger completed prefix is not passed")
        if status in {"passed", "failed"}:
            if (
                state["access_started"] is not True
                or type(state["next_ordinal"]) is not int
                or state["next_ordinal"] != SCENES_PER_SPLIT
                or state["active_batch"] is not None
                or len(state["batch_result_sha256"]) != SCENES_PER_SPLIT // BATCH_SIZE
                or state["split_result_sha256"] is None
            ):
                raise ValueError(f"{stage} terminal error ledger {split} state differs")
            if index < len(result_values):
                expected_result = result_values[index]
                if expected_result["passed"] is not (status == "passed"):
                    raise ValueError(f"{stage} terminal error ledger result status differs")
                if state["split_result_sha256"] != canonical_sha256(expected_result):
                    raise ValueError(f"{stage} terminal error ledger result digest differs")
            elif state["split_result_sha256"] is not None:
                raise ValueError(f"{stage} terminal error ledger has an unbound result")
            transition_count += 34
        elif status == "access_started":
            _exact_equal(
                state,
                {
                    "status": "access_started",
                    "access_started": True,
                    "next_ordinal": 0,
                    "active_batch": None,
                    "batch_result_sha256": [],
                    "split_result_sha256": None,
                },
                label=f"{stage} terminal error ledger access-started {split}",
            )
            transition_count += 1
        elif status == "evaluating":
            next_ordinal = state["next_ordinal"]
            batch_digests = state["batch_result_sha256"]
            active_batch = state["active_batch"]
            if (
                state["access_started"] is not True
                or type(next_ordinal) is not int
                or not 0 <= next_ordinal <= SCENES_PER_SPLIT
                or next_ordinal % BATCH_SIZE != 0
                or len(batch_digests) != next_ordinal // BATCH_SIZE
                or state["split_result_sha256"] is not None
            ):
                raise ValueError(f"{stage} terminal error ledger {split} progress differs")
            if active_batch is None:
                active_transition = 0
            else:
                expected_active = list(range(next_ordinal, next_ordinal + BATCH_SIZE))
                if (
                    type(active_batch) is not list
                    or next_ordinal + BATCH_SIZE > SCENES_PER_SPLIT
                    or active_batch != expected_active
                ):
                    raise ValueError(f"{stage} terminal error ledger active batch differs")
                active_transition = 1
            transition_count += 1 + 2 * len(batch_digests) + active_transition
        else:
            raise ValueError(f"{stage} terminal error ledger {split} status differs")

    expected_generation = transition_count + 1
    if type(record["generation"]) is not int or record["generation"] != expected_generation:
        raise ValueError(f"{stage} terminal ledger generation differs")
    if expected_outcome == "error":
        if type(expected_error) is not dict or set(expected_error) != {"type", "message"}:
            raise ValueError("terminal error ledger requires exact expected error fields")
        if any(type(expected_error[name]) is not str for name in ("type", "message")):
            raise TypeError("terminal error ledger expected error fields must be strings")
        expected_error_record = {
            **dict(expected_error),
            "report_sha256": validated_sha256(
                expected_report_sha256,
                label=f"{stage} terminal intended error report",
            ),
        }
        _exact_equal(record["error"], expected_error_record, label=f"{stage} ledger error")
    elif expected_error is not None or expected_report_sha256 is not None:
        raise ValueError("non-error terminal ledger cannot bind expected error evidence")
    return {**record, "record_sha256": supplied_hash}


def _report_root(
    *,
    stage: Literal["development", "qualification"],
    source_provenance: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    terminal_ledger_sha256: str,
) -> dict[str, Any]:
    protocol = bridge_protocol()
    opened_splits = [result["split"] for result in results]
    expected_count = 1 if stage == "development" else 3
    passed = len(results) == expected_count and all(
        result.get("passed") is True for result in results
    )
    outcome = "passed" if passed else "gate_failed"
    report = {
        "artifact_kind": "rgbd_variable_radius_qualification_report",
        "stage": stage,
        "architecture_version": ARCHITECTURE_VERSION,
        "architecture_attempt": ARCHITECTURE_ATTEMPT,
        "protocol": protocol,
        "resolved_config_sha256": FROZEN_CONFIG_SHA256,
        "scene_certificate": _frozen_scene_certificate_binding(),
        "source_provenance": _validated_published_source(
            dict(source_provenance),
            label=f"{stage} report source",
        ),
        "manifest_sha256": {
            result["split"]: MANIFEST_SHA256[result["split"]] for result in results
        },
        "results": [copy.deepcopy(dict(result)) for result in results],
        "model_state_sha256": EMPTY_MODEL_STATE_SHA256,
        "optimizer_updates": 0,
        "optimizer_state_entry_count": 0,
        "rng_state_entry_count": 0,
        "passed": passed,
        "opened_splits": opened_splits,
        "stopped_after": opened_splits[-1] if opened_splits else None,
        "materialization_started": bool(results),
        "access_completed": True,
        "outcome": outcome,
        "error": None,
        "terminal_ledger_sha256": validated_sha256(
            terminal_ledger_sha256,
            label=f"{stage} terminal ledger",
        ),
    }
    native = _json_native(report, label=f"{stage} report")
    if type(native) is not dict:
        raise RuntimeError(f"{stage} report must be one exact JSON object")
    return native


def _ledger_bindings(
    *,
    stage: Literal["development", "qualification"],
    directory_pin: _PinnedDirectory,
    source_provenance: Mapping[str, Any],
    reviewed_development: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    bindings = {
        "protocol_sha256": bridge_protocol()["protocol_sha256"],
        "resolved_config_sha256": FROZEN_CONFIG_SHA256,
        "scene_certificate_sha256": FROZEN_CERTIFICATE_SHA256,
        "run_directory": _pinned_directory_binding(directory_pin),
        "manifest_sha256": {
            split: MANIFEST_SHA256[split]
            for split in (
                ("development",)
                if stage == "development"
                else ("selector", "confirmation", "final_test")
            )
        },
        "source_provenance": _validated_published_source(
            dict(source_provenance),
            label=f"{stage} ledger source",
        ),
    }
    if stage == "development":
        if reviewed_development is not None:
            raise ValueError("development ledger cannot bind reviewed development")
        return bindings
    if type(reviewed_development) is not dict or set(reviewed_development) != {
        "checkpoint_sha256",
        "report_sha256",
        "ledger_sha256",
    }:
        raise ValueError("qualification ledger requires exactly three reviewed hashes")
    return {
        **bindings,
        "reviewed_development": {
            name: validated_sha256(value, label=f"reviewed {name}")
            for name, value in reviewed_development.items()
        },
    }


def _validate_development_report_extras(report: Mapping[str, Any]) -> None:
    checkpoint = report["checkpoint"]
    if checkpoint is not None:
        checkpoint_allowed = report["passed"] is True or (
            report["outcome"] == "error"
            and type(report["results"]) is list
            and len(report["results"]) == 1
            and report["results"][0]["split"] == "development"
            and report["results"][0]["passed"] is True
        )
        if not checkpoint_allowed:
            raise ValueError("development report cannot claim an unearned checkpoint")
        if type(checkpoint) is not dict or set(checkpoint) != {
            "path",
            "sha256",
            "bytes",
            "model_state_sha256",
            "model_state_entry_count",
            "optimizer_state_entry_count",
            "rng_state_entry_count",
        }:
            raise ValueError("passed development report checkpoint schema differs")
        if checkpoint["path"] != str(canonical_checkpoint_path().relative_to(REPOSITORY_ROOT)):
            raise ValueError("development report checkpoint path differs")
        validated_sha256(checkpoint["sha256"], label="development checkpoint")
        if (
            checkpoint["model_state_sha256"] != EMPTY_MODEL_STATE_SHA256
            or type(checkpoint["model_state_entry_count"]) is not int
            or checkpoint["model_state_entry_count"] != 0
            or type(checkpoint["optimizer_state_entry_count"]) is not int
            or checkpoint["optimizer_state_entry_count"] != 0
            or type(checkpoint["rng_state_entry_count"]) is not int
            or checkpoint["rng_state_entry_count"] != 0
            or type(checkpoint["bytes"]) is not float
            or checkpoint["bytes"] <= 0.0
            or checkpoint["bytes"] > MAX_CHECKPOINT_BYTES
        ):
            raise ValueError("development checkpoint report evidence differs")
    elif report["passed"] is True:
        raise ValueError("passed development report must publish its exact checkpoint")


def _validate_qualification_report_extras(report: Mapping[str, Any]) -> None:
    reviewed = report["reviewed_development"]
    if type(reviewed) is not dict or set(reviewed) != {
        "checkpoint_sha256",
        "report_sha256",
        "ledger_sha256",
    }:
        raise ValueError("qualification reviewed-development schema differs")
    for name, value in reviewed.items():
        validated_sha256(value, label=f"qualification reviewed {name}")


@dataclass(frozen=True, slots=True)
class _ReviewedDevelopmentPin:
    checkpoint_bytes: bytes
    report_bytes: bytes
    ledger_bytes: bytes
    checkpoint_identity: tuple[int, int, int, int, int]
    report_identity: tuple[int, int, int, int, int]
    ledger_identity: tuple[int, int, int, int, int]
    run_directory_binding_sha256: str


def _validate_reviewed_development_boundary(
    *,
    directory_pin: _PinnedDirectory,
    pin: _ReviewedDevelopmentPin,
    reviewed: Mapping[str, str],
    expected_source: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(pin, _ReviewedDevelopmentPin):
        raise TypeError("protected boundary requires an immutable development pin")
    directory_binding_sha256 = canonical_sha256(_pinned_directory_binding(directory_pin))
    if pin.run_directory_binding_sha256 != directory_binding_sha256:
        raise PermissionError("reviewed development changed run-directory identity")
    if type(reviewed) is not dict or set(reviewed) != {
        "checkpoint_sha256",
        "report_sha256",
        "ledger_sha256",
    }:
        raise ValueError("protected boundary reviewed hash schema differs")
    _validate_run_tree(
        directory_pin,
        DEVELOPMENT_ARTIFACT_NAMES,
        stage="protected development boundary",
    )
    paths = {
        "checkpoint": canonical_checkpoint_path(),
        "report": canonical_development_report_path(),
        "ledger": development_ledger_path(),
    }
    expected_bytes = {
        "checkpoint": pin.checkpoint_bytes,
        "report": pin.report_bytes,
        "ledger": pin.ledger_bytes,
    }
    expected_identities = {
        "checkpoint": pin.checkpoint_identity,
        "report": pin.report_identity,
        "ledger": pin.ledger_identity,
    }
    reviewed_names = {
        "checkpoint": "checkpoint_sha256",
        "report": "report_sha256",
        "ledger": "ledger_sha256",
    }
    current: dict[str, bytes] = {}
    for name, path in paths.items():
        before = _pinned_artifact_identity(directory_pin, path)
        contents = _pinned_stable_read_bytes(
            directory_pin,
            path,
            label=f"protected reviewed {name}",
        )
        after = _pinned_artifact_identity(directory_pin, path)
        if (
            before != expected_identities[name]
            or after != expected_identities[name]
            or contents != expected_bytes[name]
            or sha256_bytes(contents) != reviewed[reviewed_names[name]]
        ):
            raise PermissionError(f"protected reviewed {name} bytes or inode changed")
        current[name] = contents
    report = _validate_report(
        _strict_json_loads(current["report"], label="protected development report"),
        stage="development",
        expected_source=expected_source,
    )
    _validate_development_report_extras(report)
    if report["passed"] is not True:
        raise ValueError("protected boundary requires passed development")
    bindings = _ledger_bindings(
        stage="development",
        directory_pin=directory_pin,
        source_provenance=expected_source,
    )
    _validate_terminal_ledger(
        current["ledger"],
        stage="development",
        bindings=bindings,
        expected_results=report["results"],
    )
    payload = _validate_checkpoint_payload(
        current["checkpoint"],
        expected_source=expected_source,
    )
    if (
        report["checkpoint"]["sha256"] != reviewed["checkpoint_sha256"]
        or report["checkpoint"]["bytes"] != float(len(current["checkpoint"]))
        or report["terminal_ledger_sha256"] != reviewed["ledger_sha256"]
    ):
        raise ValueError("protected development cross-artifact binding differs")
    _exact_equal(
        payload["development_result"],
        report["results"][0],
        label="protected checkpoint/report development result",
    )
    return report, payload


def _review_development_bundle(
    *,
    directory_pin: _PinnedDirectory,
    reviewed_checkpoint_sha256: str,
    reviewed_report_sha256: str,
    reviewed_development_ledger_sha256: str,
    expected_source: Mapping[str, Any],
) -> tuple[
    dict[str, Tensor],
    dict[str, str],
    dict[str, Any],
    _ReviewedDevelopmentPin,
]:
    reviewed = {
        "checkpoint_sha256": validated_sha256(
            reviewed_checkpoint_sha256,
            label="reviewed checkpoint",
        ),
        "report_sha256": validated_sha256(
            reviewed_report_sha256,
            label="reviewed development report",
        ),
        "ledger_sha256": validated_sha256(
            reviewed_development_ledger_sha256,
            label="reviewed development ledger",
        ),
    }
    _validate_run_tree(
        directory_pin,
        DEVELOPMENT_ARTIFACT_NAMES,
        stage="reviewed development",
    )
    report_contents = _pinned_stable_read_bytes(
        directory_pin,
        canonical_development_report_path(),
        label="reviewed development report",
    )
    checkpoint_contents = _pinned_stable_read_bytes(
        directory_pin,
        canonical_checkpoint_path(),
        label="reviewed development checkpoint",
    )
    ledger_contents = _pinned_stable_read_bytes(
        directory_pin,
        development_ledger_path(),
        label="reviewed development ledger",
    )
    actual = {
        "checkpoint_sha256": sha256_bytes(checkpoint_contents),
        "report_sha256": sha256_bytes(report_contents),
        "ledger_sha256": sha256_bytes(ledger_contents),
    }
    if actual != reviewed:
        raise ValueError("externally reviewed development hashes differ from disk bytes")
    pin = _ReviewedDevelopmentPin(
        checkpoint_bytes=checkpoint_contents,
        report_bytes=report_contents,
        ledger_bytes=ledger_contents,
        checkpoint_identity=_pinned_artifact_identity(
            directory_pin,
            canonical_checkpoint_path(),
        ),
        report_identity=_pinned_artifact_identity(
            directory_pin,
            canonical_development_report_path(),
        ),
        ledger_identity=_pinned_artifact_identity(
            directory_pin,
            development_ledger_path(),
        ),
        run_directory_binding_sha256=canonical_sha256(_pinned_directory_binding(directory_pin)),
    )
    report, _ = _validate_reviewed_development_boundary(
        directory_pin=directory_pin,
        pin=pin,
        reviewed=reviewed,
        expected_source=expected_source,
    )
    return {}, reviewed, report, pin


def _persist_exception_report(
    *,
    directory_pin: _PinnedDirectory,
    path: Path,
    stage: Literal["development", "qualification"],
    source_provenance: Mapping[str, Any],
    ledger: _AccessLedger | None,
    error: BaseException,
    checkpoint: Mapping[str, Any] | None = None,
    reviewed: Mapping[str, str] | None = None,
    results: Sequence[Mapping[str, Any]] = (),
) -> str | None:
    """Fail closed by binding the intended report before publishing any report bytes."""

    error_fields = {
        "type": type(error).__name__,
        "message": str(error),
    }
    expected_order = (
        ["development"] if stage == "development" else ["selector", "confirmation", "final_test"]
    )
    if ledger is None or ledger._terminal:
        return None
    opened_splits: list[str] = []
    try:
        ledger._verify_disk()
        ledger_record = copy.deepcopy(ledger._record)
        opened_splits = [
            split
            for split in expected_order
            if ledger_record["splits"][split]["access_started"] is True
        ]
        if opened_splits != expected_order[: len(opened_splits)]:
            raise PermissionError("error ledger opened split prefix differs")
    except BaseException:
        return None
    result_values = [copy.deepcopy(dict(result)) for result in results]
    result_splits = [result.get("split") for result in result_values]
    if not opened_splits and result_splits:
        opened_splits = list(result_splits)
    try:
        source = _validated_published_source(
            dict(source_provenance),
            label=f"{stage} terminal error source",
        )
        record: dict[str, Any] = {
            "artifact_kind": "rgbd_variable_radius_qualification_report",
            "stage": stage,
            "architecture_version": ARCHITECTURE_VERSION,
            "architecture_attempt": ARCHITECTURE_ATTEMPT,
            "protocol": bridge_protocol(),
            "resolved_config_sha256": FROZEN_CONFIG_SHA256,
            "scene_certificate": _frozen_scene_certificate_binding(),
            "source_provenance": source,
            "manifest_sha256": {
                split: MANIFEST_SHA256[split]
                for split in result_splits
                if type(split) is str and split in MANIFEST_SHA256
            },
            "results": result_values,
            "model_state_sha256": EMPTY_MODEL_STATE_SHA256,
            "optimizer_updates": 0,
            "optimizer_state_entry_count": 0,
            "rng_state_entry_count": 0,
            "passed": False,
            "terminal_ledger_sha256": None,
            "opened_splits": opened_splits,
            "stopped_after": opened_splits[-1] if opened_splits else None,
            "materialization_started": bool(opened_splits),
            "access_completed": False,
            "outcome": "error",
            "error": error_fields,
        }
        if stage == "development":
            record["checkpoint"] = copy.deepcopy(checkpoint)
        else:
            record["reviewed_development"] = copy.deepcopy(reviewed)
        validated = _validate_report(
            record,
            stage=stage,
            expected_source=source,
        )
        if stage == "development":
            _validate_development_report_extras(validated)
        else:
            _validate_qualification_report_extras(validated)
        intended = _report_bytes(validated)
        report_digest = sha256_bytes(intended)
    except BaseException:
        return None
    try:
        ledger.fail(
            error_type=error_fields["type"],
            error_message=error_fields["message"],
            report_sha256=report_digest,
        )
        terminal_contents = _pinned_stable_read_bytes(
            directory_pin,
            ledger.path,
            label=f"{stage} terminal error ledger",
        )
        _validate_terminal_ledger(
            terminal_contents,
            stage=stage,
            bindings=copy.deepcopy(ledger._record["bindings"]),
            expected_outcome="error",
            expected_opened_splits=opened_splits,
            expected_error=error_fields,
            expected_report_sha256=report_digest,
            expected_results=result_values,
        )
    except BaseException:
        return None
    try:
        if _pinned_lexists(directory_pin, path, label=f"{stage} error report"):
            _pinned_durable_replace(directory_pin, path, intended)
        else:
            _pinned_durable_create(directory_pin, path, intended)
        _pinned_require_single_link_regular(
            directory_pin,
            path,
            label=f"{stage} terminal error report",
        )
        reread = _pinned_stable_read_bytes(
            directory_pin,
            path,
            label=f"{stage} terminal error report",
        )
        if reread != intended or sha256_bytes(reread) != report_digest:
            return None
    except BaseException:
        return None
    return report_digest


def run_development(
    config: OrpheusConfig,
    *,
    config_path: Path,
    report_path: Path,
    checkpoint_path: Path,
    source_provenance: Mapping[str, Any],
    invocation_seal: _RunnerInvocationSeal,
) -> int:
    """Consume development once and publish review bytes without protected access."""

    _require_frozen_cli_caller(consume_depth=2)
    if not isinstance(invocation_seal, _RunnerInvocationSeal):
        raise PermissionError("development requires a frozen CLI invocation seal")
    directory_pin = _runner_invocation_pin(
        invocation_seal,
        statuses=frozenset({"issued"}),
    )
    _require_canonical_path(config_path, _frozen_config_path(), label="config path")
    _require_canonical_path(
        report_path,
        canonical_development_report_path(),
        label="development report path",
        directory_pin=directory_pin,
    )
    _require_canonical_path(
        checkpoint_path,
        canonical_checkpoint_path(),
        label="checkpoint path",
        directory_pin=directory_pin,
    )
    source = _validated_published_source(
        dict(source_provenance),
        label="development invocation source",
    )
    _guard_frozen_inputs(
        config,
        config_path=config_path,
        published_source=source,
        label="development preflight",
    )
    invocation_context = _runner_invocation_context(
        stage="development",
        directory_pin=directory_pin,
        config=config,
        config_path=config_path,
        report_path=report_path,
        checkpoint_path=checkpoint_path,
        development_report_path=report_path,
        source_provenance=source,
        reviewed_development=None,
    )
    _consume_runner_invocation_seal(invocation_seal, context=invocation_context)
    _validate_run_tree(directory_pin, frozenset(), stage="development preflight")
    _validate_distinct_canonical_paths(
        directory_pin,
        {
            "development_report": report_path,
            "checkpoint": checkpoint_path,
            "development_ledger": development_ledger_path(),
            "qualification_report": canonical_qualification_report_path(),
            "qualification_ledger": qualification_ledger_path(),
        },
    )
    bindings = _ledger_bindings(
        stage="development",
        directory_pin=directory_pin,
        source_provenance=source,
    )
    ledger: _AccessLedger | None = None
    checkpoint_record: dict[str, Any] | None = None
    results: list[dict[str, Any]] = []
    try:
        authorization_context = _run_authorization_context(
            stage="development",
            config=config,
            config_path=config_path,
            report_path=report_path,
            checkpoint_path=checkpoint_path,
            development_report_path=report_path,
            ledger_path=development_ledger_path(),
            ledger_bindings=bindings,
            source_provenance=source,
            reviewed_development=None,
            invocation_seal=invocation_seal,
        )
        authorization = _mint_run_authorization(
            invocation_seal=invocation_seal,
            context=authorization_context,
        )
        ledger = _AccessLedger(
            development_ledger_path(),
            stage="development",
            bindings=bindings,
            directory_pin=directory_pin,
            authorization=authorization,
        )
        manifest = _ManifestCapability(split="development", ledger=ledger)

        def guard(label: str) -> None:
            assert ledger is not None
            ledger._verify_disk()
            _guard_frozen_inputs(
                config,
                config_path=config_path,
                published_source=source,
                label=label,
            )

        result = _evaluate_manifest_once(
            config,
            split="development",
            manifest=manifest,
            reviewed_state=None,
            expected_state_sha256=None,
            boundary_guard=guard,
        )
        results.append(result)
        if result["passed"]:
            checkpoint_record = _save_review_checkpoint(
                checkpoint_path,
                directory_pin=directory_pin,
                config=config,
                development_result=result,
                source_provenance=source,
            )
        terminal_ledger_sha256 = ledger.finish()
        terminal_ledger_contents = _pinned_stable_read_bytes(
            directory_pin,
            ledger.path,
            label="development terminal ledger",
        )
        if sha256_bytes(terminal_ledger_contents) != terminal_ledger_sha256:
            raise RuntimeError("development terminal ledger digest changed")
        _validate_terminal_ledger(
            terminal_ledger_contents,
            stage="development",
            bindings=bindings,
            expected_outcome="passed" if result["passed"] else "gate_failed",
            expected_opened_splits=["development"],
            expected_results=results,
        )
        report = _report_root(
            stage="development",
            source_provenance=source,
            results=[result],
            terminal_ledger_sha256=terminal_ledger_sha256,
        )
        report["checkpoint"] = checkpoint_record
        validated = _validate_report(
            report,
            stage="development",
            expected_source=source,
        )
        _validate_development_report_extras(validated)
        _write_report_fresh(directory_pin, report_path, validated)
        expected_names = (
            DEVELOPMENT_ARTIFACT_NAMES
            if result["passed"]
            else frozenset({DEVELOPMENT_REPORT_NAME, DEVELOPMENT_LEDGER_NAME})
        )
        _validate_run_tree(directory_pin, expected_names, stage="development terminal")
        return 0 if result["passed"] else 1
    except BaseException as error:
        _persist_exception_report(
            directory_pin=directory_pin,
            path=report_path,
            stage="development",
            source_provenance=source,
            ledger=ledger,
            error=error,
            checkpoint=checkpoint_record,
            results=results,
        )
        raise


def run_qualification(
    config: OrpheusConfig,
    *,
    config_path: Path,
    report_path: Path,
    checkpoint_path: Path,
    development_report_path: Path,
    reviewed_checkpoint_sha256: str,
    reviewed_report_sha256: str,
    reviewed_development_ledger_sha256: str,
    source_provenance: Mapping[str, Any],
    invocation_seal: _RunnerInvocationSeal,
) -> int:
    """Consume selector, confirmation, then one-shot final under reviewed bytes."""

    _require_frozen_cli_caller(consume_depth=2)
    if not isinstance(invocation_seal, _RunnerInvocationSeal):
        raise PermissionError("qualification requires a frozen CLI invocation seal")
    directory_pin = _runner_invocation_pin(
        invocation_seal,
        statuses=frozenset({"issued"}),
    )
    _require_canonical_path(config_path, _frozen_config_path(), label="config path")
    _require_canonical_path(
        report_path,
        canonical_qualification_report_path(),
        label="qualification report path",
        directory_pin=directory_pin,
    )
    _require_canonical_path(
        checkpoint_path,
        canonical_checkpoint_path(),
        label="checkpoint path",
        directory_pin=directory_pin,
    )
    _require_canonical_path(
        development_report_path,
        canonical_development_report_path(),
        label="development report path",
        directory_pin=directory_pin,
    )
    source = _validated_published_source(
        dict(source_provenance),
        label="qualification invocation source",
    )
    _guard_frozen_inputs(
        config,
        config_path=config_path,
        published_source=source,
        label="qualification preflight",
    )
    invocation_reviewed = {
        "checkpoint_sha256": reviewed_checkpoint_sha256,
        "report_sha256": reviewed_report_sha256,
        "ledger_sha256": reviewed_development_ledger_sha256,
    }
    invocation_context = _runner_invocation_context(
        stage="qualification",
        directory_pin=directory_pin,
        config=config,
        config_path=config_path,
        report_path=report_path,
        checkpoint_path=checkpoint_path,
        development_report_path=development_report_path,
        source_provenance=source,
        reviewed_development=invocation_reviewed,
    )
    _consume_runner_invocation_seal(invocation_seal, context=invocation_context)
    reviewed_state, reviewed, _, reviewed_pin = _review_development_bundle(
        directory_pin=directory_pin,
        reviewed_checkpoint_sha256=reviewed_checkpoint_sha256,
        reviewed_report_sha256=reviewed_report_sha256,
        reviewed_development_ledger_sha256=reviewed_development_ledger_sha256,
        expected_source=source,
    )
    reviewed_identities = {
        "checkpoint": list(reviewed_pin.checkpoint_identity),
        "report": list(reviewed_pin.report_identity),
        "ledger": list(reviewed_pin.ledger_identity),
    }
    _exact_equal(
        reviewed_identities,
        invocation_context["reviewed_artifact_identities"],
        label="qualification CLI seal reviewed artifact identities",
    )
    _validate_distinct_canonical_paths(
        directory_pin,
        {
            "development_report": development_report_path,
            "checkpoint": checkpoint_path,
            "development_ledger": development_ledger_path(),
            "qualification_report": report_path,
            "qualification_ledger": qualification_ledger_path(),
        },
    )
    if _pinned_lexists(
        directory_pin,
        report_path,
        label="qualification report preflight",
    ) or _pinned_lexists(
        directory_pin,
        qualification_ledger_path(),
        label="qualification ledger preflight",
    ):
        raise FileExistsError("qualification report and ledger must both be fresh")
    bindings = _ledger_bindings(
        stage="qualification",
        directory_pin=directory_pin,
        source_provenance=source,
        reviewed_development=reviewed,
    )
    ledger: _AccessLedger | None = None
    results: list[dict[str, Any]] = []
    try:
        authorization_context = _run_authorization_context(
            stage="qualification",
            config=config,
            config_path=config_path,
            report_path=report_path,
            checkpoint_path=checkpoint_path,
            development_report_path=development_report_path,
            ledger_path=qualification_ledger_path(),
            ledger_bindings=bindings,
            source_provenance=source,
            reviewed_development=reviewed,
            invocation_seal=invocation_seal,
        )
        authorization = _mint_run_authorization(
            invocation_seal=invocation_seal,
            context=authorization_context,
        )
        ledger = _AccessLedger(
            qualification_ledger_path(),
            stage="qualification",
            bindings=bindings,
            directory_pin=directory_pin,
            authorization=authorization,
        )

        def guard(label: str) -> None:
            assert ledger is not None
            ledger._verify_disk()
            _guard_frozen_inputs(
                config,
                config_path=config_path,
                published_source=source,
                label=label,
            )
            _validate_reviewed_development_boundary(
                directory_pin=directory_pin,
                pin=reviewed_pin,
                reviewed=reviewed,
                expected_source=source,
            )

        for split in ("selector", "confirmation", "final_test"):
            result = _evaluate_manifest_once(
                config,
                split=split,
                manifest=_ManifestCapability(split=split, ledger=ledger),
                reviewed_state=reviewed_state,
                expected_state_sha256=EMPTY_MODEL_STATE_SHA256,
                boundary_guard=guard,
            )
            results.append(result)
            if not result["passed"]:
                break
        guard("qualification before terminal publication")
        terminal_ledger_sha256 = ledger.finish()
        terminal_ledger_contents = _pinned_stable_read_bytes(
            directory_pin,
            ledger.path,
            label="qualification terminal ledger",
        )
        if sha256_bytes(terminal_ledger_contents) != terminal_ledger_sha256:
            raise RuntimeError("qualification terminal ledger digest changed")
        terminal_outcome = (
            "passed"
            if len(results) == 3 and all(result["passed"] for result in results)
            else "gate_failed"
        )
        _validate_terminal_ledger(
            terminal_ledger_contents,
            stage="qualification",
            bindings=bindings,
            expected_outcome=terminal_outcome,
            expected_opened_splits=[result["split"] for result in results],
            expected_results=results,
        )
        report = _report_root(
            stage="qualification",
            source_provenance=source,
            results=results,
            terminal_ledger_sha256=terminal_ledger_sha256,
        )
        report["reviewed_development"] = dict(reviewed)
        validated = _validate_report(
            report,
            stage="qualification",
            expected_source=source,
        )
        _validate_qualification_report_extras(validated)
        _write_report_fresh(directory_pin, report_path, validated)
        _validate_run_tree(
            directory_pin,
            QUALIFICATION_ARTIFACT_NAMES,
            stage="qualification terminal",
        )
        return 0 if report["passed"] else 1
    except BaseException as error:
        _persist_exception_report(
            directory_pin=directory_pin,
            path=report_path,
            stage="qualification",
            source_provenance=source,
            ledger=ledger,
            error=error,
            reviewed=reviewed,
            results=results,
        )
        raise


__all__ = [
    "ARCHITECTURE_ATTEMPT",
    "ARCHITECTURE_VERSION",
    "BATCH_SIZE",
    "DEFAULT_GATES",
    "EMPTY_MODEL_STATE_SHA256",
    "FROZEN_CONFIG_SHA256",
    "FROZEN_SOURCE_SHA256",
    "GATE_METRIC_SCHEMA",
    "MANIFEST_SHA256",
    "REDUCER_REGISTRY",
    "bridge_protocol",
    "canonical_checkpoint_path",
    "canonical_development_report_path",
    "canonical_qualification_report_path",
    "capture_published_source",
    "development_ledger_path",
    "gate_failures",
    "new_public_model",
    "qualification_ledger_path",
    "require_frozen_config",
    "run_development",
    "run_qualification",
]


def _validated_split_result(value: Any, *, split: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{split} result must be an exact dict")
    expected_keys = {
        "split",
        "manifest",
        "manifest_sha256",
        "metrics",
        "failures",
        "passed",
        "materialization_started",
        "access_completed",
        "outcome",
        "optimizer_updates",
        "runtime_api",
        "scene_materializer",
        "model_state_sha256",
        "evidence_sha256",
        "provenance_sha256",
    }
    if set(value) != expected_keys:
        raise ValueError(f"{split} result schema differs")
    metrics = value["metrics"]
    if type(metrics) is not dict or set(metrics) != set(GATE_METRIC_SCHEMA):
        raise ValueError(f"{split} result metric schema differs")
    for name, metric in metrics.items():
        if type(metric) is not float or not math.isfinite(metric):
            raise TypeError(f"{split} result metric {name!r} is not one finite float")
    expected = _split_result(
        split=split,
        metrics=metrics,
        model_state_sha256=value["model_state_sha256"],
        evidence_sha256=value["evidence_sha256"],
        provenance_sha256=value["provenance_sha256"],
    )
    _exact_equal(value, expected, label=f"{split} result")
    return copy.deepcopy(expected)


def _validate_report(
    value: Mapping[str, Any],
    *,
    stage: Literal["development", "qualification"],
    expected_source: Mapping[str, Any],
) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{stage} report must be an exact dict")
    report = _json_native(value, label=f"{stage} report")
    if type(report) is not dict:
        raise TypeError(f"{stage} report must normalize to one exact JSON object")
    schema = DEVELOPMENT_REPORT_SCHEMA if stage == "development" else QUALIFICATION_REPORT_SCHEMA
    _require_exact_keys(report, schema, label=f"{stage} report")
    expected_scalars = {
        "artifact_kind": "rgbd_variable_radius_qualification_report",
        "stage": stage,
        "architecture_version": ARCHITECTURE_VERSION,
        "architecture_attempt": ARCHITECTURE_ATTEMPT,
        "resolved_config_sha256": FROZEN_CONFIG_SHA256,
        "scene_certificate": _frozen_scene_certificate_binding(),
        "model_state_sha256": EMPTY_MODEL_STATE_SHA256,
        "optimizer_updates": 0,
        "optimizer_state_entry_count": 0,
        "rng_state_entry_count": 0,
    }
    for name, expected in expected_scalars.items():
        _exact_equal(report[name], expected, label=f"{stage} report.{name}")
    protocol = bridge_protocol()
    _exact_equal(report["protocol"], protocol, label=f"{stage} report protocol")
    source = _validated_published_source(
        report["source_provenance"],
        label=f"{stage} report source",
    )
    expected = _validated_published_source(
        dict(expected_source),
        label=f"expected {stage} report source",
    )
    _exact_equal(source, expected, label=f"{stage} report source")
    expected_splits = (
        ["development"] if stage == "development" else ["selector", "confirmation", "final_test"]
    )
    results = report["results"]
    if type(results) is not list or len(results) > len(expected_splits):
        raise ValueError(f"{stage} report result inventory differs")
    if any(type(item) is not dict for item in results):
        raise TypeError(f"{stage} report results must contain exact dicts")
    result_splits = [item["split"] for item in results]
    if result_splits != expected_splits[: len(result_splits)]:
        raise ValueError(f"{stage} report result prefix/order differs")
    validated_results = [
        _validated_split_result(item, split=split)
        for split, item in zip(result_splits, results, strict=True)
    ]
    _exact_equal(results, validated_results, label=f"{stage} report results")

    outcome = report["outcome"]
    if type(outcome) is not str or outcome not in {"passed", "gate_failed", "error"}:
        raise ValueError(f"{stage} report outcome differs")
    if type(report["passed"]) is not bool:
        raise TypeError(f"{stage} report passed must be an exact bool")
    if outcome == "passed":
        if (
            len(results) != len(expected_splits)
            or any(item["passed"] is not True for item in results)
            or report["passed"] is not True
        ):
            raise ValueError(f"{stage} passed report result binding differs")
    elif outcome == "gate_failed":
        if (
            not results
            or any(item["passed"] is not True for item in results[:-1])
            or results[-1]["passed"] is not False
            or report["passed"] is not False
        ):
            raise ValueError(f"{stage} gate-failed report result binding differs")
    elif report["passed"] is not False:
        raise ValueError(f"{stage} error report cannot pass")
    elif (
        any(item["passed"] is not True for item in results[:-1])
        or sum(item["passed"] is False for item in results) > 1
    ):
        raise ValueError(f"{stage} error report completed-result prefix differs")

    opened_splits = report["opened_splits"]
    if (
        type(opened_splits) is not list
        or any(type(split) is not str for split in opened_splits)
        or opened_splits != expected_splits[: len(opened_splits)]
    ):
        raise ValueError(f"{stage} report opened split prefix differs")
    if outcome in {"passed", "gate_failed"}:
        if opened_splits != result_splits:
            raise ValueError(f"{stage} terminal report opened/result split binding differs")
        expected_materialization = True
        expected_access_completed = True
        expected_error: dict[str, str] | None = None
    else:
        if (
            opened_splits[: len(result_splits)] != result_splits
            or len(opened_splits) not in {len(result_splits), len(result_splits) + 1}
            or (results and results[-1]["passed"] is False and opened_splits != result_splits)
        ):
            raise ValueError(f"{stage} error report opened/result split binding differs")
        expected_materialization = bool(opened_splits)
        expected_access_completed = False
        error = report["error"]
        if (
            type(error) is not dict
            or set(error) != {"type", "message"}
            or any(type(error[name]) is not str for name in ("type", "message"))
        ):
            raise ValueError(f"{stage} error report error schema differs")
        expected_error = error
    if report["materialization_started"] is not expected_materialization:
        raise ValueError(f"{stage} report materialization flag differs")
    if report["access_completed"] is not expected_access_completed:
        raise ValueError(f"{stage} report access-completed flag differs")
    expected_stopped_after = opened_splits[-1] if opened_splits else None
    _exact_equal(
        report["stopped_after"],
        expected_stopped_after,
        label=f"{stage} report stopped_after",
    )
    _exact_equal(report["error"], expected_error, label=f"{stage} report error")

    expected_manifests = {split: MANIFEST_SHA256[split] for split in result_splits}
    try:
        _exact_equal(
            report["manifest_sha256"],
            expected_manifests,
            label=f"{stage} report manifests",
        )
    except ValueError as error:
        raise ValueError(f"{stage} report manifest binding differs") from error
    if outcome == "error":
        if report["terminal_ledger_sha256"] is not None:
            raise ValueError(f"{stage} error report cannot preclaim a terminal ledger hash")
    else:
        validated_sha256(
            report["terminal_ledger_sha256"],
            label=f"{stage} report terminal ledger",
        )
    if outcome != "error" and report["error"] is not None:
        raise ValueError(f"{stage} non-error report cannot contain error fields")
    if report["manifest_sha256"] != expected_manifests:
        raise ValueError(f"{stage} report manifest binding differs")
    return report


def _evaluate_nominal_batch(
    config: OrpheusConfig,
    episodes: Sequence[_PacketEpisode],
    *,
    vjp_audit_indices: tuple[int, ...],
    reviewed_state: Mapping[str, Tensor] | None,
    expected_state_sha256: str | None,
) -> tuple[list[SceneEvidence], dict[str, float], dict[str, Any]]:
    """Evaluate one packet-only B=4 batch from fresh empty runtime state."""

    if len(episodes) != BATCH_SIZE:
        raise ValueError("nominal evaluation requires exactly four episodes")
    _validate_episode_batch(episodes)
    expected_vjp_indices = tuple(
        index for index, episode in enumerate(episodes) if episode.ordinal in VJP_AUDIT_ORDINALS
    )
    if (
        type(vjp_audit_indices) is not tuple
        or any(type(index) is not int for index in vjp_audit_indices)
        or vjp_audit_indices != expected_vjp_indices
        or len(vjp_audit_indices) > 1
    ):
        raise ValueError("nominal batch VJP indices differ from the predeclared axis audit")
    model = _new_strict_runtime(
        config,
        strict_profile=True,
        reviewed_state=reviewed_state,
        expected_state_sha256=expected_state_sha256,
    )
    model.eval()
    operation_counts = {"predict": 0, "reset": 0, "correct": 0}
    original_predict = model.predict
    original_reset = model.reset
    original_correct = model.updater.correct
    live_measurement_captures: list[_LiveMeasurementCapture] = []

    def recording_predict(query_times: Sequence[float] | Tensor) -> Any:
        operation_counts["predict"] += 1
        return original_predict(query_times)

    def recording_reset(batch_size: int = 1) -> None:
        operation_counts["reset"] += 1
        original_reset(batch_size=batch_size)

    def recording_correct(*args: Any, **kwargs: Any) -> Any:
        expected_keywords = {
            "prior",
            "measured",
            "predicted",
            "association",
            "innovation",
            "dt",
            "cause",
        }
        if args or set(kwargs) != expected_keywords:
            raise RuntimeError("runtime correction call schema differs from the frozen seam")
        if operation_counts["correct"] != len(live_measurement_captures):
            raise RuntimeError("live measurement capture ledger count differs")
        measured = kwargs["measured"]
        captured = _record_live_measurement(live_measurement_captures, measured)
        if captured is not measured or kwargs["measured"] is not measured:
            raise RuntimeError("live measurement capture changed correction input identity")
        operation_counts["correct"] += 1
        return original_correct(**kwargs)

    model.predict = recording_predict  # type: ignore[method-assign]
    model.reset = recording_reset  # type: ignore[method-assign]
    model.updater.correct = recording_correct  # type: ignore[method-assign]
    model.reset(batch_size=BATCH_SIZE)
    initial_rss = _process_max_rss_bytes()
    packets: list[ObservationPacket] = []
    raw_proposal_history: list[Tensor] = []
    deployed_slot_history: list[Tensor] = []
    raw_history: list[Tensor] = []
    deployed_history: list[Tensor] = []
    radius_valid_history: list[Tensor] = []
    radius_bounds_history: list[Tensor] = []
    residual_history: list[Tensor] = []
    condition_history: list[Tensor] = []
    position_history: list[Tensor] = []
    velocity_history: list[Tensor] = []
    object_id_history: list[Tensor] = []
    active_history: list[Tensor] = []
    physical_by_slot_history: list[Tensor] = []
    emitted_lv_history: list[Tensor] = []
    stored_lv_history: list[Tensor] = []
    raw_anchor_vjp: Tensor | None = None
    matched_pairs = 0
    ambiguous_pairs = 0
    perception_seconds = 0.0
    slow_radius_index: int | None = None

    for frame_index in HISTORY_FRAME_INDICES:
        packet = _packet_for_frame(
            episodes,
            frame_index,
            differentiable=bool(vjp_audit_indices),
        )
        packets.append(packet)
        _validate_packet_registration(packet)
        capture_count_before = len(live_measurement_captures)
        started = time.perf_counter()
        belief = model.ingest(packet)
        perception_seconds += time.perf_counter() - started
        measured = model.last_measurements
        if type(measured) is not MeasurementSet:
            raise RuntimeError("runtime omitted public measurement evidence")
        if (
            capture_count_before != frame_index
            or len(live_measurement_captures) != capture_count_before + 1
            or operation_counts["correct"] != len(live_measurement_captures)
        ):
            raise RuntimeError("runtime must expose exactly one live measurement per ingest")
        live_measured = _validated_live_measurement_capture(
            live_measurement_captures[-1],
            measured,
            expected_call_index=frame_index,
        )
        live_raw_radius = live_measured.auxiliary.get("world_radius")
        public_raw_radius = measured.auxiliary.get("world_radius")
        raw_valid = measured.auxiliary.get("world_radius_valid_mask")
        raw_lv = measured.auxiliary.get("world_radius_log_variance")
        residual = measured.auxiliary.get("surface_fit_radius_relative_error")
        condition = measured.auxiliary.get("surface_fit_condition_number")
        if (
            type(live_raw_radius) is not Tensor
            or live_raw_radius.shape != (BATCH_SIZE, 2, 1)
            or type(public_raw_radius) is not Tensor
            or public_raw_radius.shape != (BATCH_SIZE, 2, 1)
            or not isinstance(raw_valid, Tensor)
            or raw_valid.shape != (BATCH_SIZE, 2)
            or not isinstance(raw_lv, Tensor)
            or raw_lv.shape != (BATCH_SIZE, 2, 1)
            or not isinstance(residual, Tensor)
            or residual.shape != (BATCH_SIZE, 2)
            or not isinstance(condition, Tensor)
            or condition.shape != (BATCH_SIZE, 2)
            or measured.supported_state_fields != ("position", "radius")
        ):
            raise RuntimeError("runtime radius evidence is incomplete or has changed schema")
        if slow_radius_index is None:
            slow_radius_index = slow_packing_map(belief.objects)["geometry"].start
        raw_proposal_history.append(public_raw_radius[..., 0])
        deployed_slot_history.append(belief.objects.geometry[..., 0])
        frame_raw: list[Tensor] = []
        frame_deployed: list[Tensor] = []
        frame_valid: list[Tensor] = []
        frame_bounds: list[Tensor] = []
        frame_residual: list[Tensor] = []
        frame_condition: list[Tensor] = []
        frame_position: list[Tensor] = []
        frame_velocity: list[Tensor] = []
        frame_ids: list[Tensor] = []
        frame_active: list[Tensor] = []
        frame_emitted_lv: list[Tensor] = []
        frame_stored_lv: list[Tensor] = []
        frame_mapping: list[Tensor] = []
        frame_raw_vjp: list[Tensor] = []
        for batch_index, episode in enumerate(episodes):
            truth = episode.position_truth[frame_index]
            measurement_mapping = _measurement_physical_mapping(
                measured,
                truth,
                batch_index,
            )
            belief_mapping = _physical_mapping(
                belief.objects.position[batch_index],
                truth,
            )
            frame_mapping.append(belief_mapping)
            frame_raw.append(
                _gather_physical_by_slot(
                    public_raw_radius[batch_index, :, 0],
                    measurement_mapping,
                )
            )
            if frame_index == ANCHOR_FRAME_INDEX and vjp_audit_indices:
                frame_raw_vjp.append(
                    _gather_physical_by_slot(
                        live_raw_radius[batch_index, :, 0],
                        measurement_mapping,
                    )
                )
            frame_valid.append(
                _gather_physical_by_slot(
                    raw_valid[batch_index],
                    measurement_mapping,
                )
            )
            frame_residual.append(
                _gather_physical_by_slot(
                    residual[batch_index],
                    measurement_mapping,
                )
            )
            frame_condition.append(
                _gather_physical_by_slot(
                    condition[batch_index],
                    measurement_mapping,
                )
            )
            frame_emitted_lv.append(
                _gather_physical_by_slot(
                    raw_lv[batch_index, :, 0],
                    measurement_mapping,
                )
            )
            deployed = _gather_physical_by_slot(
                belief.objects.geometry[batch_index, :, 0],
                belief_mapping,
            )
            frame_deployed.append(deployed)
            frame_bounds.append(
                (deployed >= config.model.rgbd.minimum_world_radius)
                & (deployed <= config.model.rgbd.maximum_world_radius)
            )
            frame_stored_lv.append(
                _gather_physical_by_slot(
                    belief.objects.slow_log_variance[
                        batch_index,
                        :,
                        slow_radius_index,
                    ],
                    belief_mapping,
                )
            )
            frame_position.append(
                _gather_physical_by_slot(
                    belief.objects.position[batch_index],
                    belief_mapping,
                )
            )
            frame_velocity.append(
                _gather_physical_by_slot(
                    belief.objects.velocity[batch_index],
                    belief_mapping,
                )
            )
            frame_ids.append(
                _gather_physical_by_slot(
                    belief.objects.object_id[batch_index],
                    belief_mapping,
                )
            )
            frame_active.append(
                _gather_physical_by_slot(
                    belief.objects.active[batch_index],
                    belief_mapping,
                )
            )
        raw_history.append(torch.stack(frame_raw))
        deployed_history.append(torch.stack(frame_deployed))
        radius_valid_history.append(torch.stack(frame_valid))
        radius_bounds_history.append(torch.stack(frame_bounds))
        residual_history.append(torch.stack(frame_residual))
        condition_history.append(torch.stack(frame_condition))
        emitted_lv_history.append(torch.stack(frame_emitted_lv))
        stored_lv_history.append(torch.stack(frame_stored_lv))
        position_history.append(torch.stack(frame_position))
        velocity_history.append(torch.stack(frame_velocity))
        object_id_history.append(torch.stack(frame_ids))
        active_history.append(torch.stack(frame_active))
        physical_by_slot_history.append(torch.stack(frame_mapping))
        if frame_raw_vjp:
            if frame_index != ANCHOR_FRAME_INDEX or len(frame_raw_vjp) != BATCH_SIZE:
                raise RuntimeError("live raw VJP anchor capture inventory differs")
            raw_anchor_vjp = torch.stack(frame_raw_vjp)
        latest = model.diagnostics.latest
        if latest is None:
            raise RuntimeError("runtime omitted ingest diagnostics")
        matched_pairs += latest.matched_pairs
        ambiguous_pairs += latest.ambiguous_pairs

    model.updater.correct = original_correct  # type: ignore[method-assign]
    if (
        model.state.ingest_count != HISTORY_FRAME_COUNT
        or operation_counts["correct"] != HISTORY_FRAME_COUNT
        or len(live_measurement_captures) != HISTORY_FRAME_COUNT
    ):
        raise RuntimeError("runtime ingest count differs from exact history")
    stream_history = model.state.temporal_histories.get(RUNTIME_STREAM_KEY)
    if stream_history is None:
        raise RuntimeError("runtime omitted RGB-D temporal history")
    raw_tensor = torch.stack(raw_history, dim=1)
    deployed_tensor = torch.stack(deployed_history, dim=1)
    valid_tensor = torch.stack(radius_valid_history, dim=1)
    bounds_tensor = torch.stack(radius_bounds_history, dim=1)
    residual_tensor = torch.stack(residual_history, dim=1)
    condition_tensor = torch.stack(condition_history, dim=1)
    position_tensor = torch.stack(position_history, dim=1)
    velocity_tensor = torch.stack(velocity_history, dim=1)
    object_id_tensor = torch.stack(object_id_history, dim=1)
    active_tensor = torch.stack(active_history, dim=1)
    emitted_lv_tensor = torch.stack(emitted_lv_history, dim=1)
    stored_lv_tensor = torch.stack(stored_lv_history, dim=1)

    started = time.perf_counter()
    trajectory = model.predict(HORIZONS_SECONDS)
    rollout_seconds = time.perf_counter() - started
    belief_after_rollout = model.state.belief
    if belief_after_rollout is None:
        raise RuntimeError("runtime omitted retained belief after public rollout")
    rollout_alias_count = _rollout_output_alias_count(trajectory, belief_after_rollout)
    anchor_mapping = physical_by_slot_history[-1]
    future_positions: list[Tensor] = []
    future_velocities: list[Tensor] = []
    future_active: list[Tensor] = []
    for batch_index in range(BATCH_SIZE):
        mapping = anchor_mapping[batch_index]
        future_positions.append(
            torch.stack(
                [
                    _gather_physical_by_slot(
                        trajectory.positions[batch_index, horizon_index],
                        mapping,
                    )
                    for horizon_index in range(len(HORIZONS_SECONDS))
                ]
            )
        )
        future_velocities.append(
            torch.stack(
                [
                    _gather_physical_by_slot(
                        trajectory.velocities[batch_index, horizon_index],
                        mapping,
                    )
                    for horizon_index in range(len(HORIZONS_SECONDS))
                ]
            )
        )
        future_active.append(
            torch.stack(
                [
                    _gather_physical_by_slot(
                        trajectory.active_mask[batch_index, horizon_index],
                        mapping,
                    )
                    for horizon_index in range(len(HORIZONS_SECONDS))
                ]
            )
        )
    future_position_tensor = torch.stack(future_positions)
    future_velocity_tensor = torch.stack(future_velocities)
    future_active_tensor = torch.stack(future_active)

    vjp: dict[str, float] = {}
    if vjp_audit_indices:
        if type(raw_anchor_vjp) is not Tensor:
            raise RuntimeError("live raw VJP anchor was not captured")
        vjp = _vjp_metrics(
            packets=packets,
            raw_anchor_vjp=raw_anchor_vjp,
            deployed_anchor_vjp=deployed_history[-1],
            scene_sha256s=[episode.scene_sha256 for episode in episodes],
            audit_indices=vjp_audit_indices,
        )
    rows: list[SceneEvidence] = []
    expected_lv = raw_tensor.new_tensor(EXPECTED_RADIUS_LOG_VARIANCE)
    for batch_index, episode in enumerate(episodes):
        provenance_receipt = _evaluator_provenance_receipt(
            episode,
            packets,
            evidence_truth_sha256=_expected_scene_evidence_truth_digest(episode),
        )
        provenance_sha256 = _provenance_receipt_sha256(provenance_receipt)
        ids = object_id_tensor[batch_index]
        identity_switches = int((ids[1:] != ids[:-1]).sum())
        persistent_mismatches = int((ids != ids[:1]).sum())
        history_sample_count = float(stream_history.sample_mask[batch_index].sum(dim=-1).min())
        history_valid_count = float(stream_history.valid_mask[batch_index].sum(dim=-1).min())
        sampled_timestamps = stream_history.timestamps[batch_index][
            stream_history.sample_mask[batch_index]
        ]
        history_span = (
            float(sampled_timestamps.max() - sampled_timestamps.min())
            if sampled_timestamps.numel()
            else 0.0
        )
        owner_error = (deployed_tensor[batch_index] - raw_tensor[batch_index]).abs().max()
        lv_owner_error = (
            (stored_lv_tensor[batch_index] - emitted_lv_tensor[batch_index]).abs().max()
        )
        emitted_pair_error = (
            (emitted_lv_tensor[batch_index, :, 0] - emitted_lv_tensor[batch_index, :, 1])
            .abs()
            .max()
        )
        stored_pair_error = (
            (stored_lv_tensor[batch_index, :, 0] - stored_lv_tensor[batch_index, :, 1]).abs().max()
        )
        diagnostics = {
            "association_ambiguous_pair_count": float(ambiguous_pairs / BATCH_SIZE),
            "association_matched": float(matched_pairs / BATCH_SIZE),
            "association_opportunities": float((HISTORY_FRAME_COUNT - 1) * 2),
            "configured_radius_log_variance_error": float(
                abs(
                    math.log(config.model.rgbd.measurement_radius_variance)
                    - EXPECTED_RADIUS_LOG_VARIANCE
                )
            ),
            "direct_radius_log_variance_owner_error": float(lv_owner_error),
            "direct_radius_owner_error": float(owner_error),
            "emitted_radius_log_variance_pairwise_error": float(emitted_pair_error),
            "history_sample_count": history_sample_count,
            "history_span_error_seconds": float(
                abs(history_span - (HISTORY_FRAME_COUNT - 1) / FRAME_RATE_HZ)
            ),
            "history_valid_count": history_valid_count,
            "identity_switch_count": float(identity_switches),
            "ingested_frame_count": float(HISTORY_FRAME_COUNT),
            "larger_radius_slot": float(
                torch.nonzero(
                    physical_by_slot_history[-1][batch_index] == int(episode.radius_truth.argmax()),
                    as_tuple=False,
                )[0, 0]
            ),
            "persistent_id_mismatch_count": float(persistent_mismatches),
            "process_max_rss_bytes": _process_max_rss_bytes(),
            "public_predict_calls": float(operation_counts["predict"]),
            "public_rollout_output_alias_count": float(
                rollout_alias_count if batch_index == 0 else 0
            ),
            "radius_owner_count": 1.0,
            "model_reset_count": float(operation_counts["reset"]),
            "state_ingest_count": float(model.state.ingest_count),
            "stored_radius_log_variance_pairwise_error": float(stored_pair_error),
        }
        row = SceneEvidence(
            split=episode.split,
            ordinal=episode.ordinal,
            scene_sha256=episode.scene_sha256,
            non_radius_scene_sha256=episode.non_radius_scene_sha256,
            primitive_index=episode.primitive_index,
            pair_variant=episode.pair_variant,
            radius_role=episode.radius_role,
            camera_stratum=episode.camera_stratum,
            twin_ordinal=episode.twin_ordinal,
            pair_variant_twin_ordinal=episode.pair_variant_twin_ordinal,
            provenance_sha256=provenance_sha256,
            radius_truth=episode.radius_truth.detach().cpu().contiguous(),
            anchor_raw_radius=raw_tensor[batch_index, -1].detach().cpu().contiguous(),
            anchor_deployed_radius=(deployed_tensor[batch_index, -1].detach().cpu().contiguous()),
            history_raw_radius=raw_tensor[batch_index].detach().cpu().contiguous(),
            history_deployed_radius=(deployed_tensor[batch_index].detach().cpu().contiguous()),
            radius_valid=valid_tensor[batch_index].detach().cpu().contiguous(),
            radius_in_bounds=bounds_tensor[batch_index].detach().cpu().contiguous(),
            surface_fit_relative_residual=(
                residual_tensor[batch_index].detach().cpu().contiguous()
            ),
            surface_fit_condition=(condition_tensor[batch_index].detach().cpu().contiguous()),
            current_position_truth=(
                episode.position_truth[ANCHOR_FRAME_INDEX].detach().cpu().contiguous()
            ),
            current_position_mean=(position_tensor[batch_index, -1].detach().cpu().contiguous()),
            current_velocity_truth=(
                episode.velocity_truth[ANCHOR_FRAME_INDEX].detach().cpu().contiguous()
            ),
            current_velocity_mean=(velocity_tensor[batch_index, -1].detach().cpu().contiguous()),
            future_position_truth=(
                episode.position_truth[list(TARGET_FRAME_INDICES)].detach().cpu().contiguous()
            ),
            future_position_mean=(future_position_tensor[batch_index].detach().cpu().contiguous()),
            future_velocity_truth=(
                episode.velocity_truth[list(TARGET_FRAME_INDICES)].detach().cpu().contiguous()
            ),
            future_velocity_mean=(future_velocity_tensor[batch_index].detach().cpu().contiguous()),
            object_ids=object_id_tensor[batch_index].detach().cpu().contiguous(),
            active=active_tensor[batch_index].detach().cpu().contiguous(),
            rollout_active=future_active_tensor[batch_index].detach().cpu().contiguous(),
            diagnostics=tuple(sorted((key, float(value)) for key, value in diagnostics.items())),
        )
        _register_scene_evidence(
            row,
            episode=episode,
            packets=packets,
        )
        rows.append(_validated_evidence(row, split=episode.split, ordinal=episode.ordinal))

    audit = {
        "model_state_sha256": _model_state_sha256(model),
        "model_state": copy.deepcopy(model.state_dict()),
        "packets": tuple(packets),
        "raw_proposal_radius": torch.stack(raw_proposal_history, dim=1),
        "deployed_slot_radius": torch.stack(deployed_slot_history, dim=1),
        "raw_radius": raw_tensor,
        "deployed_radius": deployed_tensor,
        "belief": model.state.belief,
        "history": model.state.temporal_histories.get(RUNTIME_STREAM_KEY),
        "prediction": trajectory,
        "perception_latency_seconds": float(perception_seconds),
        "state_only_rollout_latency_seconds": float(rollout_seconds),
        "persistent_runtime_tensor_state_bytes": float(_persistent_runtime_tensor_bytes(model)),
        "process_rss_delta_bytes": float(max(0.0, _process_max_rss_bytes() - initial_rss)),
        "expected_log_variance": expected_lv,
        "vjp_scene_receipts": tuple(
            (
                episodes[index].ordinal,
                episodes[index].scene_sha256,
                episodes[index].primitive_index,
                episodes[index].pair_variant,
                episodes[index].radius_role,
                episodes[index].camera_stratum,
            )
            for index in vjp_audit_indices
        ),
    }
    return rows, vjp, audit


def _detached_packet(packet: ObservationPacket) -> ObservationPacket:
    child = replace(
        packet,
        payload={
            key: value.detach().clone() if isinstance(value, Tensor) else copy.deepcopy(value)
            for key, value in packet.payload.items()
        },
        calibration={
            key: value.detach().clone() if isinstance(value, Tensor) else copy.deepcopy(value)
            for key, value in packet.calibration.items()
        },
        metadata=copy.deepcopy(dict(packet.metadata)),
    )
    return _register_derived_packet(
        packet,
        child,
        provenance="exact_bit_cached_packet_replay",
    )


def _run_cached_packet_trace(
    config: OrpheusConfig,
    packets: Sequence[ObservationPacket],
    *,
    strict_profile: bool,
    reviewed_state: Mapping[str, Tensor] | None,
    expected_state_sha256: str | None,
) -> dict[str, Any]:
    model = _new_strict_runtime(
        config,
        strict_profile=strict_profile,
        reviewed_state=reviewed_state,
        expected_state_sha256=expected_state_sha256,
    )
    model.eval()
    raw: list[Tensor] = []
    deployed: list[Tensor] = []
    measurements: list[Any] = []
    for packet in packets:
        replay = _detached_packet(packet)
        _validate_packet_registration(replay)
        belief = model.ingest(replay)
        measured = model._last_measurements
        if measured is None:
            raise RuntimeError("cached packet replay omitted measurements")
        radius = measured.auxiliary.get("world_radius")
        if not isinstance(radius, Tensor) or radius.shape != (BATCH_SIZE, 2, 1):
            raise RuntimeError("cached packet replay omitted radius values")
        raw.append(radius[..., 0])
        deployed.append(belief.objects.geometry[..., 0])
        measurements.append(measured)
    prediction = model.predict(HORIZONS_SECONDS)
    return {
        "raw": torch.stack(raw, dim=1),
        "deployed": torch.stack(deployed, dim=1),
        "belief": model.state.belief,
        "history": model.state.temporal_histories.get(RUNTIME_STREAM_KEY),
        "prediction": prediction,
        "measurements": tuple(measurements),
        "model": model,
    }


def _tensor_bit_mismatch_count(left: Tensor, right: Tensor) -> int:
    if left.shape != right.shape or left.dtype != right.dtype:
        return max(left.numel(), right.numel(), 1)
    if left.dtype is torch.float32:
        return int(
            (left.detach().cpu().view(torch.int32) != right.detach().cpu().view(torch.int32)).sum()
        )
    if left.dtype is torch.float64:
        return int(
            (left.detach().cpu().view(torch.int64) != right.detach().cpu().view(torch.int64)).sum()
        )
    return int((left.detach().cpu() != right.detach().cpu()).sum())


def _tensor_tree_mismatch_count(left: Any, right: Any) -> int:
    if type(left) is not type(right):
        return 1
    if isinstance(left, Tensor):
        return _tensor_bit_mismatch_count(left, right)
    if isinstance(left, Mapping):
        if list(left) != list(right):
            return 1
        return sum(_tensor_tree_mismatch_count(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        if len(left) != len(right):
            return 1
        return sum(_tensor_tree_mismatch_count(a, b) for a, b in zip(left, right, strict=True))
    if hasattr(left, "__dataclass_fields__"):
        return sum(
            _tensor_tree_mismatch_count(getattr(left, name), getattr(right, name))
            for name in left.__dataclass_fields__
        )
    return int(left != right)


def _alternate_prior_metrics(
    config: OrpheusConfig,
    packets: Sequence[ObservationPacket],
    baseline: Mapping[str, Any],
    reviewed_state: Mapping[str, Tensor] | None,
    expected_state_sha256: str | None,
) -> dict[str, float]:
    values = (0.195, 0.245)
    totals = {
        "prior_raw_radius_bit_mismatch_count": 0,
        "prior_deployed_radius_bit_mismatch_count": 0,
        "prior_complete_state_bit_mismatch_count": 0,
        "prior_object_id_mismatch_count": 0,
        "prior_history_bit_mismatch_count": 0,
        "prior_prediction_bit_mismatch_count": 0,
    }
    for prior in values:
        rgbd = replace(config.model.rgbd, world_radius=prior)
        model_config = replace(config.model, rgbd=rgbd)
        alternate_config = replace(config, model=model_config)
        trace = _run_cached_packet_trace(
            alternate_config,
            packets,
            strict_profile=False,
            reviewed_state=reviewed_state,
            expected_state_sha256=expected_state_sha256,
        )
        totals["prior_raw_radius_bit_mismatch_count"] += _tensor_tree_mismatch_count(
            baseline["raw_proposal_radius"].detach(),
            trace["raw"],
        )
        totals["prior_deployed_radius_bit_mismatch_count"] += _tensor_tree_mismatch_count(
            baseline["deployed_slot_radius"].detach(),
            trace["deployed"],
        )
        totals["prior_complete_state_bit_mismatch_count"] += _tensor_tree_mismatch_count(
            baseline["belief"],
            trace["belief"],
        )
        baseline_belief = baseline["belief"]
        alternate_belief = trace["belief"]
        totals["prior_object_id_mismatch_count"] += _tensor_bit_mismatch_count(
            baseline_belief.objects.object_id,
            alternate_belief.objects.object_id,
        )
        totals["prior_history_bit_mismatch_count"] += _tensor_tree_mismatch_count(
            baseline["history"],
            trace["history"],
        )
        totals["prior_prediction_bit_mismatch_count"] += _tensor_tree_mismatch_count(
            baseline["prediction"],
            trace["prediction"],
        )
    return {
        **{key: float(value) for key, value in totals.items()},
        "alternate_prior_count": 2.0,
    }


def _legacy_control_metrics(
    config: OrpheusConfig,
    packets: Sequence[ObservationPacket],
    *,
    reviewed_state: Mapping[str, Tensor] | None,
    expected_state_sha256: str | None,
) -> dict[str, float]:
    legacy_rgbd = replace(
        config.model.rgbd,
        metric_radius_estimation_enabled=False,
        world_radius=0.21,
    )
    legacy = replace(config, model=replace(config.model, rgbd=legacy_rgbd))
    trace = _run_cached_packet_trace(
        legacy,
        packets,
        strict_profile=False,
        reviewed_state=reviewed_state,
        expected_state_sha256=expected_state_sha256,
    )
    deployed = trace["deployed"]
    maximum_error = float((deployed - 0.21).abs().max())
    measurements = trace["measurements"]
    supported_count = sum(
        int("radius" in measured.supported_state_fields) for measured in measurements
    )
    belief = trace["belief"]
    slow_index = slow_packing_map(belief.objects)["geometry"].start
    variance_writes = int(
        (
            belief.objects.slow_log_variance[..., slow_index]
            != config.model.filter.max_log_variance * 0.0
        ).sum()
    )
    # The factory's reviewed initial slow variance is exact zero.
    return {
        "legacy_fixed_radius_max_abs_error_m": maximum_error,
        "legacy_radius_variance_write_count": float(variance_writes),
        "legacy_supported_radius_field_count": float(supported_count),
    }


def _replace_packet_inputs(
    packet: ObservationPacket,
    *,
    rgb: Tensor | None = None,
    depth: Tensor | None = None,
) -> ObservationPacket:
    child = replace(
        packet,
        payload={
            "rgb": packet.payload["rgb"] if rgb is None else rgb,
            "depth": packet.payload["depth"] if depth is None else depth,
        },
    )
    return _register_derived_packet(
        packet,
        child,
        provenance="fail_closed_adversarial_packet",
    )


def _adversarial_metrics(
    config: OrpheusConfig,
    packets: Sequence[ObservationPacket],
    *,
    reviewed_state: Mapping[str, Tensor] | None,
    expected_state_sha256: str | None,
) -> dict[str, float]:
    if len(packets) != HISTORY_FRAME_COUNT:
        raise ValueError("adversarial audit requires an exact history")
    results: dict[str, float] = {}
    cases = {
        "missing_depth": _replace_packet_inputs(
            _detached_packet(packets[-1]),
            depth=torch.zeros_like(packets[-1].payload["depth"]),
        ),
        "no_foreground": _replace_packet_inputs(
            _detached_packet(packets[-1]),
            rgb=torch.zeros_like(packets[-1].payload["rgb"]),
            depth=torch.zeros_like(packets[-1].payload["depth"]),
        ),
    }
    for name, adversarial_anchor in cases.items():
        model = _new_strict_runtime(
            config,
            strict_profile=True,
            reviewed_state=reviewed_state,
            expected_state_sha256=expected_state_sha256,
        )
        model.eval()
        for packet in packets[:-1]:
            replay = _detached_packet(packet)
            _validate_packet_registration(replay)
            model.ingest(replay)
        before = model.state.belief.objects.geometry[..., 0].detach().clone()
        _validate_packet_registration(adversarial_anchor)
        after = model.ingest(adversarial_anchor)
        measured = model._last_measurements
        if measured is None:
            raise RuntimeError("adversarial runtime omitted measurement")
        valid = measured.auxiliary["world_radius_valid_mask"]
        results[f"{name}_radius_valid_fraction"] = float(valid.to(torch.float32).mean())
        results[f"{name}_radius_write_count"] = float(
            (after.objects.geometry[..., 0] != before).sum()
        )

    model = _new_strict_runtime(
        config,
        strict_profile=True,
        reviewed_state=reviewed_state,
        expected_state_sha256=expected_state_sha256,
    )
    model.eval()
    for packet in packets[:-1]:
        replay = _detached_packet(packet)
        _validate_packet_registration(replay)
        model.ingest(replay)
    before = model.state.belief.objects.geometry[..., 0].detach().clone()
    anchor = _detached_packet(packets[-1])
    foreground = anchor.payload["depth"] > 0.0
    colour = anchor.payload["rgb"].new_tensor((0.5, 0.5, 0.5))[None, :, None, None]
    ambiguous_rgb = torch.where(foreground.expand_as(anchor.payload["rgb"]), colour, 0.0)
    ambiguous_packet = _replace_packet_inputs(anchor, rgb=ambiguous_rgb)
    _validate_packet_registration(ambiguous_packet)
    after = model.ingest(ambiguous_packet)
    results["ambiguous_radius_write_count"] = float(
        (after.objects.geometry[..., 0] != before).sum()
    )
    measured = model._last_measurements
    if measured is None:
        raise RuntimeError("ambiguous runtime omitted measurement")
    malformed = replace(
        measured,
        auxiliary={
            key: value
            for key, value in measured.auxiliary.items()
            if key != "world_radius_log_variance"
        },
    )
    rejected = 0
    try:
        malformed.validate()
    except (TypeError, ValueError):
        rejected = 1
    results["malformed_radius_group_rejection_count"] = float(rejected)
    return results


def _diagnostics(row: SceneEvidence) -> dict[str, float]:
    result = dict(row.diagnostics)
    if len(result) != len(row.diagnostics):
        raise ValueError("scene evidence contains duplicate diagnostic names")
    return result


def _measured_operation_count_metrics(
    diagnostics: Sequence[Mapping[str, float]],
) -> dict[str, float]:
    if type(diagnostics) not in {list, tuple} or not diagnostics:
        raise ValueError("operation-count metrics require nonempty ordered diagnostics")

    def values(name: str) -> list[float]:
        try:
            result = [item[name] for item in diagnostics]
        except KeyError as error:
            raise ValueError(f"operation-count evidence omitted {name!r}") from error
        if any(type(value) is not float or not math.isfinite(value) for value in result):
            raise ValueError(f"operation-count evidence {name!r} must be finite exact floats")
        return result

    predict = values("public_predict_calls")
    reset = values("model_reset_count")
    return {
        "public_predict_calls_per_batch_min": float(min(predict)),
        "public_predict_calls_per_batch_max": float(max(predict)),
        "model_reset_count_per_batch_min": float(min(reset)),
        "model_reset_count_per_batch_max": float(max(reset)),
    }


def _reduction(values: Tensor, reduction: Reduction, *, label: str) -> float:
    if not isinstance(values, Tensor) or values.numel() == 0:
        raise ValueError(f"{label} reduction requires nonempty tensor evidence")
    resolved = values.to(torch.float64)
    if not bool(torch.isfinite(resolved).all()):
        raise FloatingPointError(f"{label} reduction evidence is nonfinite")
    if reduction == "sum":
        result = resolved.sum()
    elif reduction == "min":
        result = resolved.min()
    elif reduction == "max":
        result = resolved.max()
    elif reduction == "mean":
        result = resolved.mean()
    elif reduction == "rmse":
        result = resolved.square().mean().sqrt()
    else:  # pragma: no cover - closed Literal plus registry validation.
        raise RuntimeError(f"unknown reducer {reduction!r}")
    value = float(result)
    if not math.isfinite(value):
        raise FloatingPointError(f"{label} reducer produced nonfinite scalar")
    return value


def _evidence_sources(rows: Sequence[SceneEvidence]) -> dict[str, Tensor]:
    anchor_error = torch.stack([row.anchor_deployed_radius - row.radius_truth for row in rows])
    history_error = torch.stack(
        [row.history_deployed_radius - row.radius_truth[None, :] for row in rows]
    )
    raw_history_error = torch.stack(
        [row.history_raw_radius - row.radius_truth[None, :] for row in rows]
    )
    diagnostics = [_diagnostics(row) for row in rows]

    def diagnostic(name: str) -> Tensor:
        try:
            values = [item[name] for item in diagnostics]
        except KeyError as error:
            raise ValueError(f"scene evidence omitted diagnostic {name!r}") from error
        return torch.tensor(values, dtype=torch.float64)

    sources = {
        "anchor_radius_error": anchor_error,
        "history_radius_error": history_error,
        "all_radius_abs_error": torch.cat(
            (history_error.abs().reshape(-1), raw_history_error.abs().reshape(-1))
        ),
        "radius_valid": torch.stack([row.radius_valid for row in rows]).to(torch.float64),
        "radius_in_bounds": torch.stack([row.radius_in_bounds for row in rows]).to(torch.float64),
        "surface_fit_relative_residual": torch.stack(
            [row.surface_fit_relative_residual for row in rows]
        ),
        "surface_fit_condition": torch.stack([row.surface_fit_condition for row in rows]),
        "within_track_radius_span": torch.stack(
            [
                row.history_deployed_radius.max(dim=0).values
                - row.history_deployed_radius.min(dim=0).values
                for row in rows
            ]
        ),
        "current_position_error": torch.stack(
            [row.current_position_mean - row.current_position_truth for row in rows]
        ),
        "current_velocity_error": torch.stack(
            [row.current_velocity_mean - row.current_velocity_truth for row in rows]
        ),
        "identity_switch_count": diagnostic("identity_switch_count"),
        "persistent_id_mismatch_count": diagnostic("persistent_id_mismatch_count"),
        "association_ambiguous_pair_count": diagnostic("association_ambiguous_pair_count"),
        "direct_radius_owner_error": diagnostic("direct_radius_owner_error"),
        "direct_radius_log_variance_owner_error": diagnostic(
            "direct_radius_log_variance_owner_error"
        ),
        "configured_radius_log_variance_error": diagnostic("configured_radius_log_variance_error"),
        "emitted_radius_log_variance_pairwise_error": diagnostic(
            "emitted_radius_log_variance_pairwise_error"
        ),
        "stored_radius_log_variance_pairwise_error": diagnostic(
            "stored_radius_log_variance_pairwise_error"
        ),
        "active": torch.stack([row.active for row in rows]).to(torch.float64),
        "rollout_active": torch.stack([row.rollout_active for row in rows]).to(torch.float64),
        "public_rollout_output_alias_count": diagnostic("public_rollout_output_alias_count"),
    }
    expected = {spec.source for spec in REDUCER_REGISTRY}
    if set(sources) != expected:
        raise RuntimeError(
            f"reducer source schema differs: missing={sorted(expected - set(sources))}, "
            f"extra={sorted(set(sources) - expected)}"
        )
    return sources


def _grouped_radius_metrics(rows: Sequence[SceneEvidence]) -> dict[str, float]:
    metrics: dict[str, float] = {}

    def rmse(selected: Sequence[SceneEvidence], object_index: int | None = None) -> float:
        if not selected:
            raise ValueError("radius grouping selected no scenes")
        errors = torch.stack([row.anchor_deployed_radius - row.radius_truth for row in selected])
        if object_index is not None:
            errors = errors[:, object_index]
        return _reduction(errors, "rmse", label="grouped radius")

    for object_index in OBJECT_INDICES:
        metrics[f"anchor_radius_rmse_m/object_{object_index}"] = rmse(
            rows,
            object_index,
        )
    for primitive_index in range(PRIMITIVES_PER_SPLIT):
        metrics[f"anchor_radius_rmse_m/primitive_{primitive_index}"] = rmse(
            [row for row in rows if row.primitive_index == primitive_index]
        )
    for pair_variant in range(PAIR_VARIANTS_PER_PRIMITIVE):
        metrics[f"anchor_radius_rmse_m/pair_variant_{pair_variant}"] = rmse(
            [row for row in rows if row.pair_variant == pair_variant]
        )
    for role in range(RADIUS_ROLES_PER_PRIMITIVE):
        metrics[f"anchor_radius_rmse_m/role_{role}"] = rmse(
            [row for row in rows if row.radius_role == role]
        )
    for stratum in range(CAMERA_STRATA):
        metrics[f"anchor_radius_rmse_m/camera_stratum_{stratum}"] = rmse(
            [row for row in rows if row.camera_stratum == stratum]
        )
    if set(metrics) != set(_radius_group_metric_names()):
        raise RuntimeError("grouped radius metric schema differs")
    return metrics


def _position_velocity_metrics(rows: Sequence[SceneEvidence]) -> dict[str, float]:
    position_error = torch.stack(
        [row.current_position_mean - row.current_position_truth for row in rows]
    )
    velocity_error = torch.stack(
        [row.current_velocity_mean - row.current_velocity_truth for row in rows]
    )
    future_position_error = torch.stack(
        [row.future_position_mean - row.future_position_truth for row in rows]
    )
    future_velocity_error = torch.stack(
        [row.future_velocity_mean - row.future_velocity_truth for row in rows]
    )
    metrics: dict[str, float] = {}
    for object_index in OBJECT_INDICES:
        for axis_index, axis in enumerate(AXIS_NAMES):
            metrics[f"current_position_rmse_m/object_{object_index}/{axis}"] = _reduction(
                position_error[:, object_index, axis_index],
                "rmse",
                label="current position object axis",
            )
            metrics[f"current_velocity_rmse_mps/object_{object_index}/{axis}"] = _reduction(
                velocity_error[:, object_index, axis_index],
                "rmse",
                label="current velocity object axis",
            )
    for horizon_index, horizon in enumerate(HORIZONS_SECONDS):
        label = f"{horizon:.2f}"
        metrics[f"horizon_{label}_position_rmse_m"] = _reduction(
            future_position_error[:, horizon_index],
            "rmse",
            label=f"horizon {label} position",
        )
        metrics[f"horizon_{label}_velocity_rmse_mps"] = _reduction(
            future_velocity_error[:, horizon_index],
            "rmse",
            label=f"horizon {label} velocity",
        )
        for object_index in OBJECT_INDICES:
            for axis_index, axis in enumerate(AXIS_NAMES):
                metrics[f"horizon_{label}_position_rmse_m/object_{object_index}/{axis}"] = (
                    _reduction(
                        future_position_error[:, horizon_index, object_index, axis_index],
                        "rmse",
                        label="horizon position object axis",
                    )
                )
                metrics[f"horizon_{label}_velocity_rmse_mps/object_{object_index}/{axis}"] = (
                    _reduction(
                        future_velocity_error[:, horizon_index, object_index, axis_index],
                        "rmse",
                        label="horizon velocity object axis",
                    )
                )
    if set(metrics) != set(_position_velocity_metric_names()):
        raise RuntimeError("position/velocity metric schema differs")
    return metrics


def _counterfactual_metrics(rows: Sequence[SceneEvidence]) -> dict[str, float]:
    if len(rows) != SCENES_PER_SPLIT:
        raise ValueError("counterfactual metrics require one complete split")
    by_ordinal = {row.ordinal: row for row in rows}
    pair_errors: list[Tensor] = []
    sign_matches: list[Tensor] = []
    pair_count = 0
    non_radius_mismatches = 0
    position_max = 0.0
    velocity_max = 0.0
    for ordinal in ORDINALS:
        twin = counterfactual_twin_ordinal(ordinal)
        if ordinal >= twin:
            continue
        first = by_ordinal[ordinal]
        second = by_ordinal[twin]
        pair_count += 1
        if first.non_radius_scene_sha256 != second.non_radius_scene_sha256:
            non_radius_mismatches += 1
        position_max = max(
            position_max,
            float((first.current_position_truth - second.current_position_truth).abs().max()),
            float((first.future_position_truth - second.future_position_truth).abs().max()),
        )
        velocity_max = max(
            velocity_max,
            float((first.current_velocity_truth - second.current_velocity_truth).abs().max()),
            float((first.future_velocity_truth - second.future_velocity_truth).abs().max()),
        )
        truth_delta = second.radius_truth - first.radius_truth
        predicted_delta = second.anchor_deployed_radius - first.anchor_deployed_radius
        pair_errors.append(predicted_delta - truth_delta)
        nonzero = truth_delta != 0.0
        sign_matches.append(
            ((predicted_delta.sign() == truth_delta.sign()) & nonzero).to(torch.float32)
        )
    errors = torch.stack(pair_errors)
    signs = torch.stack(sign_matches)
    metrics = {
        "counterfactual_pair_count": float(pair_count),
        "paired_radius_delta_rmse_m": _reduction(
            errors,
            "rmse",
            label="paired radius delta",
        ),
        "paired_radius_delta_max_abs_error_m": _reduction(
            errors.abs(),
            "max",
            label="paired radius delta absolute",
        ),
        "paired_radius_delta_sign_fraction": _reduction(
            signs,
            "mean",
            label="paired radius delta sign",
        ),
        "counterfactual_non_radius_certificate_mismatch_count": float(non_radius_mismatches),
        "counterfactual_unintended_truth_position_max_abs_m": float(position_max),
        "counterfactual_unintended_truth_velocity_max_abs_mps": float(velocity_max),
    }

    pair_variant_errors: list[Tensor] = []
    pair_variant_sign_matches: list[Tensor] = []
    pair_variant_pair_count = 0
    pair_variant_non_radius_mismatches = 0
    unordered_pair_matches = 0
    pair_variant_position_max = 0.0
    pair_variant_velocity_max = 0.0
    pair_variant_estimated_position_differences: list[Tensor] = []
    pair_variant_estimated_velocity_differences: list[Tensor] = []
    for ordinal in ORDINALS:
        twin = pair_variant_twin_ordinal(ordinal)
        if ordinal >= twin:
            continue
        first = by_ordinal[ordinal]
        second = by_ordinal[twin]
        pair_variant_pair_count += 1
        if first.non_radius_scene_sha256 != second.non_radius_scene_sha256:
            pair_variant_non_radius_mismatches += 1
        first_unordered = first.radius_truth.sort().values
        second_unordered = second.radius_truth.sort().values
        unordered_pair_matches += int(torch.equal(first_unordered, second_unordered))
        pair_variant_position_max = max(
            pair_variant_position_max,
            float((first.current_position_truth - second.current_position_truth).abs().max()),
            float((first.future_position_truth - second.future_position_truth).abs().max()),
        )
        pair_variant_velocity_max = max(
            pair_variant_velocity_max,
            float((first.current_velocity_truth - second.current_velocity_truth).abs().max()),
            float((first.future_velocity_truth - second.future_velocity_truth).abs().max()),
        )
        pair_variant_estimated_position_differences.append(
            second.current_position_mean - first.current_position_mean
        )
        pair_variant_estimated_velocity_differences.append(
            second.current_velocity_mean - first.current_velocity_mean
        )
        truth_delta = second.radius_truth - first.radius_truth
        predicted_delta = second.anchor_deployed_radius - first.anchor_deployed_radius
        pair_variant_errors.append(predicted_delta - truth_delta)
        nonzero = truth_delta != 0.0
        pair_variant_sign_matches.append(
            ((predicted_delta.sign() == truth_delta.sign()) & nonzero).to(torch.float32)
        )
    variant_errors = torch.stack(pair_variant_errors)
    variant_signs = torch.stack(pair_variant_sign_matches)
    estimated_position_differences = torch.stack(pair_variant_estimated_position_differences)
    estimated_velocity_differences = torch.stack(pair_variant_estimated_velocity_differences)
    metrics.update(
        {
            "pair_variant_counterfactual_pair_count": float(pair_variant_pair_count),
            "pair_variant_paired_radius_delta_rmse_m": _reduction(
                variant_errors,
                "rmse",
                label="pair-variant paired radius delta",
            ),
            "pair_variant_paired_radius_delta_max_abs_error_m": _reduction(
                variant_errors.abs(),
                "max",
                label="pair-variant paired radius delta absolute",
            ),
            "pair_variant_paired_radius_delta_sign_fraction": _reduction(
                variant_signs,
                "mean",
                label="pair-variant paired radius delta sign",
            ),
            "pair_variant_non_radius_certificate_mismatch_count": float(
                pair_variant_non_radius_mismatches
            ),
            "pair_variant_unordered_radius_pair_match_count": float(unordered_pair_matches),
            "pair_variant_unintended_truth_position_max_abs_m": float(pair_variant_position_max),
            "pair_variant_unintended_truth_velocity_max_abs_mps": float(pair_variant_velocity_max),
            "pair_variant_estimated_anchor_position_rmse_m": _reduction(
                estimated_position_differences,
                "rmse",
                label="pair-variant estimated anchor position delta",
            ),
            "pair_variant_estimated_anchor_position_max_abs_m": _reduction(
                estimated_position_differences.abs(),
                "max",
                label="pair-variant estimated anchor position absolute delta",
            ),
            "pair_variant_estimated_anchor_velocity_rmse_mps": _reduction(
                estimated_velocity_differences,
                "rmse",
                label="pair-variant estimated anchor velocity delta",
            ),
            "pair_variant_estimated_anchor_velocity_max_abs_mps": _reduction(
                estimated_velocity_differences.abs(),
                "max",
                label="pair-variant estimated anchor velocity absolute delta",
            ),
        }
    )
    return metrics


def _evidence_sha256(rows: Sequence[SceneEvidence]) -> str:
    if type(rows) not in {list, tuple} or not rows:
        raise ValueError("evidence digest requires ordered nonempty SceneEvidence rows")
    digest = hashlib.sha256()
    _length_framed_digest_update(digest, b"rgbd_variable_radius_evidence_digest_v2")
    _length_framed_digest_update(
        digest,
        _canonical_json({"row_count": len(rows), "row_order": "ordinal_sequence"}),
    )
    for row_index, row in enumerate(rows):
        if not isinstance(row, SceneEvidence):
            raise TypeError("evidence digest rows must be SceneEvidence")
        metadata = _canonical_json(
            {
                "row_index": row_index,
                "split": row.split,
                "ordinal": row.ordinal,
                "scene_sha256": row.scene_sha256,
                "non_radius_scene_sha256": row.non_radius_scene_sha256,
                "axes": [
                    row.primitive_index,
                    row.pair_variant,
                    row.radius_role,
                    row.camera_stratum,
                    row.twin_ordinal,
                    row.pair_variant_twin_ordinal,
                ],
                "provenance_sha256": validated_sha256(
                    row.provenance_sha256,
                    label="evidence row provenance",
                ),
                "diagnostics": list(row.diagnostics),
            }
        )
        _length_framed_digest_update(digest, b"evidence_row_metadata_v2")
        _length_framed_digest_update(digest, metadata)
        for name in _EVIDENCE_TENSOR_SHAPES:
            _update_tensor_digest(
                digest,
                f"row[{row_index}].{name}",
                getattr(row, name),
            )
    return digest.hexdigest()


def _evidence_provenance_sha256(rows: Sequence[SceneEvidence]) -> str:
    if type(rows) not in {list, tuple} or not rows:
        raise ValueError("evaluator provenance digest requires ordered nonempty evidence")
    values = []
    for row in rows:
        if not isinstance(row, SceneEvidence):
            raise TypeError("evaluator provenance digest requires SceneEvidence rows")
        values.append(
            {
                "split": row.split,
                "ordinal": row.ordinal,
                "provenance_sha256": validated_sha256(
                    row.provenance_sha256,
                    label="evaluator row provenance",
                ),
            }
        )
    return canonical_sha256(
        {
            "schema": "variable_radius_ordered_evaluator_provenance_v1",
            "rows": values,
        }
    )


def _aggregate_split_metrics(
    rows: Sequence[SceneEvidence],
    *,
    vjp: Mapping[str, float],
    prior: Mapping[str, float],
    legacy: Mapping[str, float],
    adversarial: Mapping[str, float],
    resources: Mapping[str, float],
) -> dict[str, float]:
    if type(rows) not in {list, tuple} or len(rows) != SCENES_PER_SPLIT:
        raise ValueError("split aggregation requires exactly 64 evidence rows")
    validated = [
        _validated_evidence(row, split=row.split, ordinal=index) for index, row in enumerate(rows)
    ]
    split_names = {row.split for row in validated}
    if len(split_names) != 1:
        raise ValueError("split aggregation mixed conceptual splits")
    sources = _evidence_sources(validated)
    metrics = {
        spec.output: _reduction(
            sources[spec.source],
            spec.reduction,
            label=spec.output,
        )
        for spec in REDUCER_REGISTRY
    }
    metrics.update(_grouped_radius_metrics(validated))
    metrics.update(_position_velocity_metrics(validated))
    metrics.update(_counterfactual_metrics(validated))
    diagnostics = [_diagnostics(row) for row in validated]

    def values(name: str) -> list[float]:
        try:
            result = [item[name] for item in diagnostics]
        except KeyError as error:
            raise ValueError(f"aggregation omitted diagnostic {name!r}") from error
        if any(type(value) is not float or not math.isfinite(value) for value in result):
            raise ValueError(f"diagnostic {name!r} must contain finite exact floats")
        return result

    mismatches = metrics["persistent_id_mismatch_count"]
    association_matches = sum(values("association_matched"))
    association_opportunities = sum(values("association_opportunities"))
    if association_opportunities <= 0.0:
        raise ValueError("association evidence has no opportunity denominator")
    larger_slots = values("larger_radius_slot")
    metrics.update(
        {
            "scene_count": 64.0,
            "object_count": 128.0,
            "identity_coverage": float(
                1.0 - mismatches / (SCENES_PER_SPLIT * HISTORY_FRAME_COUNT * 2)
            ),
            "persistent_object_id_min": float(min(int(row.object_ids.min()) for row in validated)),
            "persistent_object_id_max": float(max(int(row.object_ids.max()) for row in validated)),
            "association_pair_coverage": float(association_matches / association_opportunities),
            "history_sample_count_min": float(min(values("history_sample_count"))),
            "history_sample_count_max": float(max(values("history_sample_count"))),
            "history_valid_count_min": float(min(values("history_valid_count"))),
            "history_valid_count_max": float(max(values("history_valid_count"))),
            "history_span_max_abs_error_seconds": float(max(values("history_span_error_seconds"))),
            "radius_owner_count_min": float(min(values("radius_owner_count"))),
            "radius_owner_count_max": float(max(values("radius_owner_count"))),
            "larger_radius_slot_zero_fraction": float(
                sum(value == 0.0 for value in larger_slots) / len(larger_slots)
            ),
            "larger_radius_slot_one_fraction": float(
                sum(value == 1.0 for value in larger_slots) / len(larger_slots)
            ),
            "ingested_frame_count_min": float(min(values("ingested_frame_count"))),
            "ingested_frame_count_max": float(max(values("ingested_frame_count"))),
            "state_ingest_count_min": float(min(values("state_ingest_count"))),
            "state_ingest_count_max": float(max(values("state_ingest_count"))),
            "packet_factory_call_count": 0.0,
            "simulator_constructor_call_count": 0.0,
            "public_renderer_call_count": 0.0,
            "formal_certificate_recomputation_count": 0.0,
            "optimizer_updates": 0.0,
            "optimizer_state_entry_count": 0.0,
            "rng_state_entry_count": 0.0,
            "learned_parameter_count": 0.0,
            "learned_parameter_bytes": 0.0,
            "module_tensor_buffer_count": 0.0,
            "persistent_module_state_key_count": 0.0,
            "persistent_module_state_bytes": 0.0,
            "source_scene_count": 256.0,
            "source_split_count": 4.0,
            "source_counterfactual_pairs": 128.0,
            "unique_scene_specification_fraction": float(
                len({row.scene_sha256 for row in validated}) / SCENES_PER_SPLIT
            ),
            **_measured_operation_count_metrics(diagnostics),
        }
    )
    for supplied, label in (
        (vjp, "VJP"),
        (prior, "prior"),
        (legacy, "legacy"),
        (adversarial, "adversarial"),
        (resources, "resources"),
    ):
        if type(supplied) is not dict:
            raise TypeError(f"{label} metrics must be an exact dict")
        for key, value in supplied.items():
            if (
                key in metrics
                or type(key) is not str
                or type(value) is not float
                or not math.isfinite(value)
            ):
                raise ValueError(f"{label} metric {key!r} is duplicate or invalid")
            metrics[key] = value
    if set(metrics) != set(GATE_METRIC_SCHEMA):
        raise RuntimeError(
            "aggregate metric schema differs: "
            f"missing={sorted(set(GATE_METRIC_SCHEMA) - set(metrics))!r}:"
            f"extra={sorted(set(metrics) - set(GATE_METRIC_SCHEMA))!r}"
        )
    return metrics
