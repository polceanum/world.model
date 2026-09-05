"""Static and fake-data tests for the known-action qualification shell.

These tests never import the formal scene, construct a governed bundle, or call
the renderer, physics engine, runtime bridge, public model, or planner.  Fake
evaluators receive only seedless addresses and one-shot capabilities.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import importlib.util
import json
import os
import pickle
import subprocess
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from world_model.training import rgbd_known_action_qualification as qualification

qualification._activate_runtime_dependencies()


def _fake_source() -> dict[str, Any]:
    blobs: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for index, (name, path) in enumerate(
        qualification.PUBLICATION_SURFACE_PATHS.items(),
        start=1,
    ):
        digest = hashlib.sha256(f"fake publication {name}".encode()).hexdigest()
        blobs[name] = {
            "path": path,
            "mode": "100644",
            "blob_oid": f"{index:040x}",
            "blob_sha256": digest,
            "worktree_sha256": digest,
            "bytes": index,
        }
        hashes[name] = digest
    commit = "a" * 40
    advertisement = f"{commit}\t{qualification._APPROVED_BRANCH_MERGE_REF}\n".encode("ascii")
    approved = sorted(
        [
            {
                "key": f"branch.{qualification._APPROVED_BRANCH}.remote".casefold(),
                "value": qualification._APPROVED_REMOTE_NAME,
            },
            {
                "key": f"branch.{qualification._APPROVED_BRANCH}.merge".casefold(),
                "value": qualification._APPROVED_BRANCH_MERGE_REF,
            },
            {
                "key": f"remote.{qualification._APPROVED_REMOTE_NAME}.url".casefold(),
                "value": qualification._APPROVED_REMOTE_URL,
            },
        ],
        key=lambda pair: (pair["key"], pair["value"]),
    )
    repository_root = "/private/tmp/fake-known-action-source"
    effective = sorted(
        [
            *approved,
            {"key": "core.fsmonitor", "value": "false"},
            {"key": "core.hookspath", "value": "/dev/null"},
            {"key": "core.untrackedcache", "value": "false"},
        ],
        key=lambda pair: (pair["key"], pair["value"]),
    )
    config_guard = {
        "schema": "rgbd_known_action_git_config_guard_v2",
        "approved": approved,
        "config_paths": {
            "schema": "rgbd_known_action_git_config_paths_v1",
            "local": {
                "path": f"{repository_root}/.git/config",
                "device": 1,
                "inode": 2,
                "mode": qualification.stat.S_IFREG | 0o600,
                "links": 1,
                "bytes": 1,
                "mtime_ns": 1,
                "ctime_ns": 1,
            },
            "worktree": {
                "path": f"{repository_root}/.git/config.worktree",
                "state": "absent",
            },
        },
        "local_pairs": approved,
        "effective_pairs": effective,
    }
    return {
        "schema": "rgbd_known_action_published_source_v2",
        "repository_root": repository_root,
        "branch": qualification._APPROVED_BRANCH,
        "remote_name": qualification._APPROVED_REMOTE_NAME,
        "remote_url": qualification._APPROVED_REMOTE_URL,
        "branch_merge_ref": qualification._APPROVED_BRANCH_MERGE_REF,
        "commit": commit,
        "tree": "b" * 40,
        "upstream_ref": qualification._APPROVED_UPSTREAM_REF,
        "upstream_commit": commit,
        "ahead": 0,
        "behind": 0,
        "dirty": False,
        "object_format": "sha1",
        "remote_publication": {
            "schema": "rgbd_known_action_remote_publication_v1",
            "git_executable": qualification._TRUSTED_GIT,
            "literal_url": qualification._APPROVED_REMOTE_URL,
            "literal_ref": qualification._APPROVED_BRANCH_MERGE_REF,
            "advertised_commit": commit,
            "advertisement_sha256": hashlib.sha256(advertisement).hexdigest(),
            "transport_profile": qualification._remote_transport_profile(),
            "config_guard": config_guard,
            "config_guard_sha256": qualification.canonical_sha256(config_guard),
        },
        "publication_surface_blobs": blobs,
        "publication_surface_sha256": hashes,
    }


def _authorization_bundle(
    *,
    argv: list[str] | None = None,
    source: dict[str, Any] | None = None,
    nonce: str = "9" * 64,
) -> tuple[dict[str, Any], dict[str, Any]]:
    exact_argv = ["--phase", "development"] if argv is None else list(argv)
    exact_source = _fake_source() if source is None else source
    receipt_core = {
        "schema": "rgbd_known_action_outer_preflight_v1",
        "nonce": "8" * 64,
        "parent_pid": os.getppid(),
        "argv": exact_argv,
        "source_provenance": exact_source,
    }
    first_receipt_sha256 = qualification.canonical_sha256(receipt_core)
    body = {
        "schema": "rgbd_known_action_outer_authorization_v1",
        "nonce": nonce,
        "first_receipt_sha256": first_receipt_sha256,
        "runner_blob_sha256": exact_source["publication_surface_blobs"]["runner"]["blob_sha256"],
        "stage": exact_argv[1],
        "argv": exact_argv,
        "source_provenance_sha256": qualification.canonical_sha256(exact_source),
        "outer_pid": os.getppid(),
        "expected_child_parent_pid": os.getppid(),
    }
    record = {**body, "record_sha256": qualification.canonical_sha256(body)}
    receipt = {
        **receipt_core,
        "receipt_sha256": first_receipt_sha256,
        "authorization_record_sha256": qualification.canonical_sha256(record),
        "authorization_nonce": nonce,
    }
    return receipt, record


@contextmanager
def _anonymous_pipe(value: dict[str, Any]) -> Iterator[tuple[str, int]]:
    contents = qualification._canonical_json(value)
    read_fd, write_fd = os.pipe()
    try:
        written = 0
        while written < len(contents):
            written += os.write(write_fd, contents[written:])
    finally:
        os.close(write_fd)
    try:
        yield str(read_fd), read_fd
    finally:
        with suppress(OSError):
            os.close(read_fd)


@contextmanager
def _anonymous_pipe_bytes(contents: bytes) -> Iterator[tuple[str, int]]:
    read_fd, write_fd = os.pipe()
    try:
        written = 0
        while written < len(contents):
            written += os.write(write_fd, contents[written:])
    finally:
        os.close(write_fd)
    try:
        yield str(read_fd), read_fd
    finally:
        with suppress(OSError):
            os.close(read_fd)


def _exact_runner_namespace() -> tuple[dict[str, Any], bytes, Path]:
    path = (
        qualification.REPOSITORY_ROOT / qualification.PUBLICATION_SURFACE_PATHS["runner"]
    ).resolve()
    runner_bytes = path.read_bytes()
    namespace: dict[str, Any] = {
        "__name__": "_rgbd_known_action_verified_bootstrap",
        "__file__": os.fspath(path),
        "__package__": None,
        "__spec__": None,
        "__cached__": None,
        "__loader__": qualification.importlib.machinery.BuiltinImporter,
        "__builtins__": qualification.builtins,
    }
    exec(
        compile(runner_bytes, os.fspath(path), "exec", dont_inherit=True),
        namespace,
        namespace,
    )
    namespace["__name__"] = "__main__"
    return namespace, runner_bytes, path


def _forget_authorization_record(record: dict[str, Any]) -> None:
    qualification._OUTER_AUTHORIZATION_IDENTITY_REGISTRY.discard(
        (qualification.canonical_sha256(record), record["nonce"])
    )


def _fake_formal_authorization_binding() -> dict[str, str]:
    return {
        "schema": "rgbd_known_action_formal_authorization_binding_v1",
        "first_receipt_sha256": "1" * 64,
        "authorization_record_sha256": "2" * 64,
        "authorization_nonce": "3" * 64,
        "runner_blob_sha256": "4" * 64,
        "bootstrap_literal_sha256": "5" * 64,
        "runner_preflight_sha256": "6" * 64,
        "bootstrap_environment_sha256": "7" * 64,
        "caller_environment_sha256": "8" * 64,
    }


def _fake_recovery_inventory(
    stage: qualification.Stage,
    *,
    names: set[str] | None = None,
) -> dict[str, Any]:
    selected = (
        (
            {qualification.DEVELOPMENT_LEDGER_NAME}
            if stage == "development"
            else {
                *qualification.DEVELOPMENT_ARTIFACT_NAMES,
                qualification.QUALIFICATION_LEDGER_NAME,
            }
        )
        if names is None
        else names
    )
    entries = [
        {
            "name": name,
            "identity": [1, index + 2, 0o100600, index + 10, index + 20, index + 30, 1],
        }
        for index, name in enumerate(sorted(selected))
    ]
    return {
        "schema": "rgbd_known_action_ledger_recovery_inventory_v1",
        "entries": entries,
        "entries_sha256": qualification.canonical_sha256(entries),
    }


def _fake_recovery_receipt(
    *,
    stage: qualification.Stage = "development",
    placement: str = "intent",
    action: str = "interrupt_nonterminal",
    outcome: str = "terminal_error",
    predecessor_sha256: str = "a" * 64,
    predecessor_generation: int = 0,
    report_sha256: str | None = None,
    inventory_names: set[str] | None = None,
) -> dict[str, Any]:
    inventory = _fake_recovery_inventory(stage, names=inventory_names)
    runner_authorization = _fake_formal_authorization_binding()
    body = {
        "schema": "rgbd_known_action_ledger_recovery_receipt_v1",
        "stage": stage,
        "execution_mode": "formal",
        "placement": placement,
        "action": action,
        "outcome": outcome,
        "run_directory": {
            "schema": "rgbd_known_action_run_directory_v1",
            "path": "/private/tmp/canonical-recovery-test",
            "parent_identity": [1, 2, 0o40700],
            "directory_identity": [1, 3, 0o40700],
            "canonical": True,
            "execution_mode": "formal",
        },
        "runner_authorization": runner_authorization,
        "runner_authorization_sha256": qualification.canonical_sha256(runner_authorization),
        "source_provenance_sha256": qualification.canonical_sha256(_fake_source()),
        "predecessor_ledger_sha256": predecessor_sha256,
        "predecessor_generation": predecessor_generation,
        "predecessor_inventory": inventory,
        "predecessor_inventory_sha256": qualification.canonical_sha256(inventory),
        "report_sha256": report_sha256,
    }
    return {**body, "receipt_sha256": qualification.canonical_sha256(body)}


def _passing_metrics() -> dict[str, float]:
    result: dict[str, float] = {}
    for rule in qualification.GATE_RULES:
        if rule.operator == "le":
            result[rule.name] = min(0.0, rule.threshold)
        else:
            result[rule.name] = rule.threshold
    return result


def _synthetic_action_reducer_records() -> tuple[list[dict[str, Any]], Any]:
    """Build private, non-governed formal-record shapes for the production reducer."""

    torch = qualification.torch
    belief_module = importlib.import_module("world_model.belief")
    namespace = qualification.types.SimpleNamespace
    candidate_impulses = torch.tensor(
        [
            [1.0 if canonical & (1 << axis) else -1.0 for axis in range(3)]
            for canonical in range(qualification.CANDIDATE_COUNT)
        ],
        dtype=torch.float32,
    )

    def trajectory(target_slots: torch.Tensor, *, action: bool) -> Any:
        batch = qualification.BATCH_SIZE
        steps = len(qualification.HORIZONS_SECONDS)
        objects = 2
        timestamps = (
            torch.tensor(
                qualification.HORIZONS_SECONDS,
                dtype=torch.float32,
            )
            .expand(batch, -1)
            .clone()
        )
        orientations = torch.zeros(batch, steps, objects, 4)
        orientations[..., 3] = 1.0
        event_logits = torch.zeros(
            batch,
            steps,
            objects,
            len(belief_module.MotionMode),
        )
        applied = torch.zeros(batch, steps, objects, dtype=torch.bool)
        known_impulse = torch.zeros(batch, steps, objects, 3)
        if action:
            rows = torch.arange(batch)
            applied[rows, 2, target_slots] = True
            event_logits[
                rows,
                2,
                target_slots,
                int(belief_module.MotionMode.EXTERNALLY_ACTUATED),
            ] = 1.0
        return belief_module.BeliefTrajectory(
            timestamps=timestamps,
            positions=torch.zeros(batch, steps, objects, 3),
            velocities=torch.zeros(batch, steps, objects, 3),
            orientations=orientations,
            motion_mode_logits=torch.zeros(
                batch,
                steps,
                objects,
                len(belief_module.MotionMode),
            ),
            fast_log_variance=torch.zeros(batch, steps, objects, 3),
            active_mask=torch.ones(batch, steps, objects, dtype=torch.bool),
            event_logits=event_logits,
            auxiliary=(
                {
                    "known_action_applied": applied,
                    "known_impulse_world": known_impulse,
                }
                if action
                else {}
            ),
        )

    def plan(
        target_slots: torch.Tensor,
        total_cost: torch.Tensor,
        selected: torch.Tensor,
    ) -> Any:
        trajectories = tuple(
            trajectory(target_slots, action=True) for _ in range(qualification.CANDIDATE_COUNT)
        )
        return namespace(
            trajectories=trajectories,
            object_id_by_slot=torch.tensor([[10, 20]])
            .expand(
                qualification.BATCH_SIZE,
                -1,
            )
            .clone(),
            terminal_squared_error=torch.zeros(
                qualification.BATCH_SIZE,
                qualification.CANDIDATE_COUNT,
            ),
            terminal_position_variance=torch.zeros(
                qualification.BATCH_SIZE,
                qualification.CANDIDATE_COUNT,
            ),
            total_cost=total_cost.clone(),
            selected_index=selected.clone(),
        )

    records: list[dict[str, Any]] = []
    for start in range(0, qualification.SCENES_PER_SPLIT, qualification.BATCH_SIZE):
        ordinals = tuple(range(start, start + qualification.BATCH_SIZE))
        roles = torch.tensor([(ordinal // 8) % 2 for ordinal in ordinals])
        target_slots = roles.to(torch.int64)
        resolved = torch.where(roles.eq(0), 10, 20).to(torch.int64)
        wrong = torch.where(roles.eq(0), 20, 10).to(torch.int64)
        optimum = torch.tensor(
            [ordinal % qualification.CANDIDATE_COUNT for ordinal in ordinals],
            dtype=torch.int64,
        )
        opposite_optimum = optimum ^ 7
        costs = torch.ones(qualification.BATCH_SIZE, qualification.CANDIDATE_COUNT)
        costs.scatter_(1, optimum[:, None], 0.0)
        opposite_costs = torch.ones_like(costs)
        opposite_costs.scatter_(1, opposite_optimum[:, None], 0.0)
        primary_plan = plan(target_slots, costs, optimum)
        rho_plan = plan(target_slots, costs, optimum)
        opposite_plan = plan(target_slots, opposite_costs, opposite_optimum)
        wrong_plan = plan(1 - target_slots, torch.full_like(costs, 2.0), optimum)
        palette_plan = plan(target_slots, costs, optimum)
        none = trajectory(target_slots, action=False)
        zero = trajectory(target_slots, action=False)
        zero.auxiliary.update(
            {
                "known_action_applied": torch.zeros(
                    qualification.BATCH_SIZE,
                    len(qualification.HORIZONS_SECONDS),
                    2,
                    dtype=torch.bool,
                ),
                "known_impulse_world": torch.zeros(
                    qualification.BATCH_SIZE,
                    len(qualification.HORIZONS_SECONDS),
                    2,
                    3,
                ),
            }
        )
        factory = belief_module.BeliefFactory(max_objects=2, appearance_dim=2)
        belief = factory.create(qualification.BATCH_SIZE)
        belief.objects.active[:] = True
        belief.objects.object_id[:] = torch.tensor([10, 20])
        model = torch.nn.Module()
        model.belief = belief
        safe = {
            "candidate_timestamps": torch.ones(
                qualification.BATCH_SIZE,
                qualification.CANDIDATE_COUNT,
            ),
            "candidate_impulses_world": candidate_impulses.expand(
                qualification.BATCH_SIZE,
                -1,
                -1,
            ).clone(),
        }
        public_positions = torch.stack(
            [item.positions for item in primary_plan.trajectories],
            dim=1,
        )
        public_velocities = torch.stack(
            [item.velocities for item in primary_plan.trajectories],
            dim=1,
        )
        specifications = tuple(
            namespace(
                ordinal=ordinal,
                camera_stratum=ordinal % 8,
                primitive_index=(ordinal // 16) % 4,
                handle_role=(ordinal // 8) % 2,
                action_delay_numerator=1 + ordinal % 4,
                optimal_canonical_index=ordinal % qualification.CANDIDATE_COUNT,
            )
            for ordinal in ordinals
        )
        records.append(
            {
                "ordinals": ordinals,
                "specifications": specifications,
                "output": {"model": model},
                "safe": safe,
                "safe_rho": {name: value.clone() for name, value in safe.items()},
                "resolved_id": resolved,
                "rho_id": resolved.clone(),
                "wrong_id": wrong,
                "plan": primary_plan,
                "rho_plan": rho_plan,
                "opposite_plan": opposite_plan,
                "wrong_plan": wrong_plan,
                "palette_plan": palette_plan,
                "none": none,
                "zero": zero,
                "physical_by_slot": torch.tensor([[0, 1]])
                .expand(
                    qualification.BATCH_SIZE,
                    -1,
                )
                .clone(),
                "truth_position": public_positions.clone(),
                "truth_velocity": public_velocities.clone(),
                "truth_none_position": none.positions.clone(),
                "truth_none_velocity": none.velocities.clone(),
                "truth_cost": costs.clone(),
                "opposite_truth_cost": opposite_costs.clone(),
                "canonical_order": torch.arange(
                    qualification.CANDIDATE_COUNT,
                    dtype=torch.int64,
                )
                .expand(qualification.BATCH_SIZE, -1)
                .clone(),
                "public_positions": public_positions,
                "public_velocities": public_velocities,
                "none_positions": none.positions[:, None].clone(),
                "none_velocities": none.velocities[:, None].clone(),
                "wrong_intended_delta": torch.zeros(
                    qualification.BATCH_SIZE,
                    qualification.CANDIDATE_COUNT,
                    len(qualification.HORIZONS_SECONDS),
                    3,
                ),
                "wrong_other_delta": torch.ones(
                    qualification.BATCH_SIZE,
                    qualification.CANDIDATE_COUNT,
                    len(qualification.HORIZONS_SECONDS),
                    3,
                ),
                "palette_position_error": torch.zeros_like(public_positions),
                "palette_velocity_error": torch.zeros_like(public_velocities),
                "palette_cost_error": torch.zeros_like(costs),
                "palette_choice_match": torch.ones(
                    qualification.BATCH_SIZE,
                    dtype=torch.bool,
                ),
                "handle_margin": torch.ones(qualification.BATCH_SIZE),
                "rejection_counts": {
                    "ambiguous": qualification.BATCH_SIZE,
                    "invalid": qualification.BATCH_SIZE,
                    "pre_anchor": qualification.BATCH_SIZE,
                    "equal_anchor": qualification.BATCH_SIZE,
                    "inactive": qualification.BATCH_SIZE,
                    "duplicate": qualification.BATCH_SIZE,
                    "non_free": qualification.BATCH_SIZE,
                },
                "none_repeat_mismatch": 0,
                "observation_hashes": [f"observation/{ordinal & ~8}" for ordinal in ordinals],
                "planning_seconds": 0.0,
                "planning_rss_delta_bytes": 0.0,
                "process_max_rss_bytes": 0.0,
                "live_result_storage_bytes": 0.0,
                "source_belief": belief.clone(),
                "source_belief_bit_mismatch_count": 0,
            }
        )
    scene = namespace(HANDLE_PROTOTYPES=((1.0, 0.0), (0.0, 1.0)))
    scene.candidate_order = lambda _stratum, *, order: tuple(range(qualification.CANDIDATE_COUNT))
    return records, scene


def _passing_inherited_metrics() -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, operator, threshold in qualification.FROZEN_ACCEPTED_ORBITAL_CONSTRAINTS:
        metrics.setdefault(name, min(0.0, threshold) if operator == "le" else threshold)
    assert len(metrics) == 685
    assert qualification._accepted_orbital_gate_failures(metrics) == []
    return metrics


def _synthetic_formal_private_receipt(
    module: Any,
    *,
    split: str,
    public_seal_sha256: str,
    truth_request: Mapping[str, Any],
    fail_metric: str | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for ordinal in range(module.SCENES_PER_SPLIT):
        digest = hashlib.sha256(f"formal-private/{split}/{ordinal}".encode()).hexdigest()
        surface = {
            "schema": "rgbd_known_action_public_materialization_surface_v1",
            "split": split,
            "ordinal": ordinal,
            "batch_index": ordinal // module.BATCH_SIZE,
            "fixed_goal_horizon_seconds": (module.MATERIALIZER_FIXED_GOAL_HORIZON_SECONDS),
            "materializer_port_sha256": digest,
            "token_receipt_sha256": digest,
            "sensor_sha256": digest,
            "safe_task_sha256": digest,
            "public_runtime_sha256": digest,
        }
        nonce = hashlib.sha256(f"formal-private-nonce/{split}/{ordinal}".encode()).digest()
        rows.append(
            {
                "schema": "rgbd_known_action_formal_private_row_commitment_v1",
                "split": split,
                "ordinal": ordinal,
                "public_surface": surface,
                "public_surface_sha256": module.canonical_sha256(surface),
                "evaluation_hashes_sha256": digest,
                "blinding_nonce_hex": nonce.hex(),
                "blinded_commitment": module._blinded_materialization_commitment(
                    surface,
                    nonce,
                ),
            }
        )
    metrics = _passing_metrics()
    if fail_metric is not None:
        rule = next(rule for rule in module.GATE_RULES if rule.name == fail_metric)
        metrics[fail_metric] = (
            rule.threshold + 1.0 if rule.operator in {"le", "eq"} else rule.threshold - 1.0
        )
    return module._formal_private_scoring_receipt(
        split=split,
        public_seal_sha256=public_seal_sha256,
        truth_request=truth_request,
        private_row_commitments=rows,
        metrics=metrics,
        inherited_orbital_evidence=module._inherited_orbital_evidence(_passing_inherited_metrics()),
    )


def _fake_numeric_evidence(
    request: qualification.BatchEvaluationRequest,
    reservation: dict[str, Any],
) -> dict[str, Any]:
    rows = []
    reservation_sha256 = qualification.canonical_sha256(reservation)
    for row_index, ordinal in enumerate(request.ordinals):
        token_body = {
            "schema": "rgbd_known_action_ordinal_capability_receipt_v1",
            "split": request.split,
            "ordinal": ordinal,
            "batch_index": ordinal // qualification.BATCH_SIZE,
            "reservation_generation": reservation["generation"],
            "reservation_sha256": reservation_sha256,
            "capability_nonce": hashlib.sha256(
                f"{request.split}/{ordinal}/{reservation['generation']}".encode()
            ).hexdigest(),
        }
        token_receipt = {
            **token_body,
            "receipt_sha256": qualification.canonical_sha256(token_body),
        }
        body = {
            "schema": "rgbd_known_action_public_numeric_row_evidence_v2",
            "split": request.split,
            "ordinal": ordinal,
            "row_index": row_index,
            "token_receipt": token_receipt,
            "scene_row_sha256": hashlib.sha256(
                f"scene/{request.split}/{ordinal}".encode()
            ).hexdigest(),
            "packet_prefix_numeric_sha256": hashlib.sha256(
                f"packet/{request.split}/{ordinal}".encode()
            ).hexdigest(),
            "safe_task_numeric_sha256": hashlib.sha256(
                f"task/{request.split}/{ordinal}".encode()
            ).hexdigest(),
            "public_runtime_numeric_sha256": hashlib.sha256(
                f"public/{request.split}/{ordinal}".encode()
            ).hexdigest(),
            "control_numeric_sha256": hashlib.sha256(
                f"control/{request.split}/{ordinal}".encode()
            ).hexdigest(),
        }
        rows.append({**body, "receipt_sha256": qualification.canonical_sha256(body)})
    return {
        "schema": "rgbd_known_action_public_numeric_batch_evidence_v2",
        "split": request.split,
        "ordinals": list(request.ordinals),
        "batch_index": request.ordinals[0] // qualification.BATCH_SIZE,
        "tensor_digest_recipe": "length_framed_little_endian_dtype_shape_C_order_v1",
        "row_receipts": rows,
        "ordered_row_receipts_sha256": qualification.canonical_sha256(
            [row["receipt_sha256"] for row in rows]
        ),
        "restricted_rows_exposed": False,
    }


class _FakeEvaluator:
    def __init__(
        self,
        *,
        fail_metric: str | None = None,
        raise_on_call: int | None = None,
        callback: Any = None,
        private_callback: Any = None,
        abort_callback: Any = None,
        raise_on_finalize: bool = False,
        raise_on_private: bool = False,
    ) -> None:
        self.fail_metric = fail_metric
        self.raise_on_call = raise_on_call
        self.callback = callback
        self.private_callback = private_callback
        self.abort_callback = abort_callback
        self.raise_on_finalize = raise_on_finalize
        self.raise_on_private = raise_on_private
        self.calls: list[tuple[str, tuple[int, int, int, int]]] = []
        self.events: list[tuple[str, str]] = []

    def evaluate_public_batch(
        self,
        request: qualification.PublicBatchEvaluationRequest,
    ) -> dict[str, Any]:
        self.events.append(("public_batch", request.split))
        self.calls.append((request.split, request.ordinals))
        reservation = request._manifest._ledger.record
        for ordinal, token in zip(request.ordinals, request.tokens, strict=True):
            request.consume(token, ordinal=ordinal)
        if self.callback is not None:
            self.callback(len(self.calls), request)
        if self.raise_on_call == len(self.calls):
            raise RuntimeError("injected evaluator failure")
        evidence = _fake_numeric_evidence(request, reservation)
        return {
            "schema": "rgbd_known_action_public_batch_evaluation_v2",
            "split": request.split,
            "ordinals": list(request.ordinals),
            "bundle_count": 4,
            "candidate_count": 32,
            "public_evidence": evidence,
            "public_evidence_sha256": qualification.canonical_sha256(evidence),
            "public_metrics": {
                "schema": "rgbd_known_action_public_batch_metrics_v1",
                "public_numeric_evidence_sha256": qualification.canonical_sha256(evidence),
                "public_row_count": 4,
                "public_candidate_count": 32,
            },
            "public_resources": {
                "schema": "rgbd_known_action_public_resource_observation_v1",
                "planning_seconds": 0.0,
                "planning_rss_delta_bytes": 0.0,
                "process_max_rss_bytes": 0.0,
                "live_result_storage_bytes": 0.0,
                "public_call_count": 1,
            },
        }

    def finalize_public_split(
        self,
        split: str,
        batch_results: tuple[dict[str, Any], ...],
    ) -> dict[str, Any]:
        self.events.append(("public_finalize", split))
        assert len(batch_results) == 16
        assert all(result["split"] == split for result in batch_results)
        if self.raise_on_finalize:
            raise RuntimeError("injected public finalization failure")
        return {
            "schema": "rgbd_known_action_public_split_finalization_v1",
            "split": split,
            "public_call_count": 16,
            "public_call_sha256": qualification.canonical_sha256(
                qualification._public_call_rows(batch_results)
            ),
            "public_evidence_sha256": qualification.canonical_sha256(
                [result["public_evidence"] for result in batch_results]
            ),
            "public_metrics_sha256": qualification.canonical_sha256(
                [result["public_metrics"] for result in batch_results]
            ),
            "public_resources_sha256": qualification.canonical_sha256(
                [result["public_resources"] for result in batch_results]
            ),
            "truth_access_count": 0,
        }

    def score_private_split(
        self,
        request: qualification.PrivateSplitScoringRequest,
    ) -> dict[str, Any]:
        self.events.append(("private_score", request.split))
        binding = request.consume()
        if self.private_callback is not None:
            self.private_callback(request)
        if self.raise_on_private:
            raise RuntimeError("injected private scoring failure")
        metrics = _passing_metrics()
        if self.fail_metric is not None:
            rule = next(rule for rule in qualification.GATE_RULES if rule.name == self.fail_metric)
            metrics[self.fail_metric] = (
                rule.threshold + 1.0 if rule.operator in {"le", "eq"} else rule.threshold - 1.0
            )
        return qualification._private_scoring_receipt(
            split=request.split,
            public_seal_sha256=request.public_seal_sha256,
            truth_request=binding,
            metrics=metrics,
            inherited_orbital_evidence=qualification._inherited_orbital_evidence(
                _passing_inherited_metrics()
            ),
        )

    def abort_split(self, split: str) -> None:
        self.events.append(("abort_split", split))
        if self.abort_callback is not None:
            self.abort_callback(split)


@contextmanager
def _pinned_run(tmp_path: Path) -> Iterator[qualification._PinnedDirectory]:
    path = (tmp_path / "run").resolve()
    pin = qualification._acquire_pinned_directory(path, create=True, canonical=False)
    try:
        yield pin
    finally:
        if qualification._PIN_REGISTRY.get(id(pin)) is pin:
            with suppress(BaseException):
                qualification._release_pinned_directory(pin)


def _synthetic_outer_authority(
    *,
    stage: qualification.Stage,
    source: dict[str, Any],
) -> qualification._OuterRunnerAuthority:
    authority = qualification._OuterRunnerAuthority(
        stage=stage,
        receipt_sha256="1" * 64,
        receipt_nonce="2" * 64,
        authorization_record_sha256="3" * 64,
        authorization_nonce="4" * 64,
        runner_blob_sha256="5" * 64,
        bootstrap_literal_sha256="6" * 64,
        runner_preflight_sha256="7" * 64,
        bootstrap_environment_sha256="8" * 64,
        caller_environment_sha256="9" * 64,
        source_sha256=qualification.canonical_sha256(source),
        argv=("--phase", stage),
        caller_code_sha256="a" * 64,
        owner_thread=qualification.threading.get_ident(),
        nonce=object(),
    )
    qualification._OUTER_RUNNER_AUTHORITY_REGISTRY[id(authority)] = (
        authority,
        "issued",
    )
    qualification._OUTER_RECEIPT_IDENTITY_REGISTRY.add(
        (authority.receipt_sha256, authority.receipt_nonce)
    )
    qualification._OUTER_AUTHORIZATION_IDENTITY_REGISTRY.add(
        (authority.authorization_record_sha256, authority.authorization_nonce)
    )
    return authority


def _forget_synthetic_outer_authority(
    authority: qualification._OuterRunnerAuthority,
) -> None:
    qualification._OUTER_RUNNER_AUTHORITY_REGISTRY.pop(id(authority), None)
    qualification._OUTER_RECEIPT_IDENTITY_REGISTRY.discard(
        (authority.receipt_sha256, authority.receipt_nonce)
    )
    qualification._OUTER_AUTHORIZATION_IDENTITY_REGISTRY.discard(
        (authority.authorization_record_sha256, authority.authorization_nonce)
    )


@contextmanager
def _consumed_formal_recovery_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    stage: qualification.Stage = "development",
    suffix: str,
) -> Iterator[
    tuple[
        dict[str, Any],
        qualification._OuterRunnerAuthority,
        qualification._PinnedDirectory,
        qualification._LedgerRecoveryAuthorization,
        Path,
    ]
]:
    source = _fake_source()
    canonical = (tmp_path / suffix).resolve()
    monkeypatch.setattr(qualification, "_canonical_run_directory", lambda: canonical)
    outer = _synthetic_outer_authority(stage=stage, source=source)
    pin: qualification._PinnedDirectory | None = None
    recovery: qualification._LedgerRecoveryAuthorization | None = None
    try:
        pin = qualification._acquire_pinned_directory(
            canonical,
            create=True,
            canonical=True,
            outer_authority=outer,
        )
        ledger_name = (
            qualification.DEVELOPMENT_LEDGER_NAME
            if stage == "development"
            else qualification.QUALIFICATION_LEDGER_NAME
        )
        names = (
            {ledger_name}
            if stage == "development"
            else {*qualification.DEVELOPMENT_ARTIFACT_NAMES, ledger_name}
        )
        for index, name in enumerate(sorted(names)):
            qualification._pinned_durable_create(
                pin,
                pin.path / name,
                f"opaque recovery artifact {index:02d}".encode("ascii"),
            )
        ledger_path = pin.path / ledger_name
        recovery = qualification._mint_ledger_recovery_authorization(
            outer,
            stage=stage,
            directory_pin=pin,
            source_provenance=source,
        )
        qualification._consume_ledger_recovery_authorization(
            recovery,
            stage=stage,
            directory_pin=pin,
            source_provenance=source,
            ledger_path=ledger_path,
        )
        yield source, outer, pin, recovery, ledger_path
    finally:
        if recovery is not None:
            qualification._revoke_ledger_recovery_authorization(recovery)
        if pin is not None and qualification._PIN_REGISTRY.get(id(pin)) is pin:
            qualification._release_pinned_directory(pin)
        if recovery is not None:
            qualification._LEDGER_RECOVERY_AUTHORIZATION_REGISTRY.pop(id(recovery), None)
        _forget_synthetic_outer_authority(outer)


def _synthetic_outer_authority_for_module(
    module: Any,
    *,
    stage: str,
    source: dict[str, Any],
    marker: str,
    reviewed: dict[str, str] | None = None,
) -> Any:
    values = [hashlib.sha256(f"{marker}/{index}".encode()).hexdigest() for index in range(1, 10)]
    argv: tuple[str, ...] = ("--phase", stage)
    if stage == "qualification":
        if reviewed is None:
            raise ValueError("synthetic qualification authority requires reviewed development")
        argv = (
            "--phase",
            "qualification",
            "--reviewed-checkpoint-sha256",
            reviewed["checkpoint_sha256"],
            "--reviewed-report-sha256",
            reviewed["report_sha256"],
            "--reviewed-development-ledger-sha256",
            reviewed["ledger_sha256"],
        )
    authority = module._OuterRunnerAuthority(
        stage=stage,
        receipt_sha256=values[0],
        receipt_nonce=values[1],
        authorization_record_sha256=values[2],
        authorization_nonce=values[3],
        runner_blob_sha256=values[4],
        bootstrap_literal_sha256=values[5],
        runner_preflight_sha256=values[6],
        bootstrap_environment_sha256=values[7],
        caller_environment_sha256=values[8],
        source_sha256=module.canonical_sha256(source),
        argv=argv,
        caller_code_sha256=hashlib.sha256(f"{marker}/caller".encode()).hexdigest(),
        owner_thread=module.threading.get_ident(),
        nonce=object(),
    )
    module._OUTER_RUNNER_AUTHORITY_REGISTRY[id(authority)] = (authority, "issued")
    module._OUTER_RECEIPT_IDENTITY_REGISTRY.add((authority.receipt_sha256, authority.receipt_nonce))
    module._OUTER_AUTHORIZATION_IDENTITY_REGISTRY.add(
        (authority.authorization_record_sha256, authority.authorization_nonce)
    )
    return authority


def _forget_synthetic_outer_authority_for_module(module: Any, authority: Any) -> None:
    module._OUTER_RUNNER_AUTHORITY_REGISTRY.pop(id(authority), None)
    module._OUTER_RECEIPT_IDENTITY_REGISTRY.discard(
        (authority.receipt_sha256, authority.receipt_nonce)
    )
    module._OUTER_AUTHORIZATION_IDENTITY_REGISTRY.discard(
        (authority.authorization_record_sha256, authority.authorization_nonce)
    )


@pytest.fixture(autouse=True)
def _clear_private_registries() -> Iterator[None]:
    qualification._clear_ephemeral_registries_for_tests()
    qualification._clear_pinned_directory_registry_for_tests()
    yield
    qualification._clear_ephemeral_registries_for_tests()
    qualification._clear_pinned_directory_registry_for_tests()


def _development_hashes(pin: qualification._PinnedDirectory) -> dict[str, str]:
    paths = qualification._artifact_paths(pin)
    return {
        "checkpoint_sha256": qualification.sha256_bytes(
            qualification._pinned_read_bytes(
                pin,
                paths["checkpoint"],
                label="test checkpoint",
                maximum=qualification.MAX_CHECKPOINT_BYTES,
            )
        ),
        "report_sha256": qualification.sha256_bytes(
            qualification._pinned_read_bytes(
                pin,
                paths["development_report"],
                label="test report",
            )
        ),
        "ledger_sha256": qualification.sha256_bytes(
            qualification._pinned_read_bytes(
                pin,
                paths["development_ledger"],
                label="test ledger",
            )
        ),
    }


def _new_development_ledger(
    pin: qualification._PinnedDirectory,
) -> tuple[qualification._AccessLedger, dict[str, Any]]:
    source = _fake_source()
    bindings = qualification._ledger_bindings(
        stage="development",
        execution_mode="fake_test",
        directory_pin=pin,
        source_provenance=source,
    )
    path = qualification._artifact_paths(pin)["development_ledger"]
    authorization = qualification._authorize_ledger_creation(
        invocation_seal=qualification._mint_fake_runner_invocation_seal_for_tests(
            stage="development",
            directory_pin=pin,
            source_provenance=source,
        ),
        stage="development",
        directory_pin=pin,
        source_provenance=source,
        bindings=bindings,
        ledger_path=path,
    )
    return (
        qualification._AccessLedger(
            path,
            stage="development",
            bindings=bindings,
            directory_pin=pin,
            authorization=authorization,
        ),
        bindings,
    )


@contextmanager
def _formal_materializer_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    marker: str,
    clear_after: bool = True,
) -> Iterator[
    tuple[
        qualification._AccessLedger,
        qualification._PinnedDirectory,
        dict[str, Any],
    ]
]:
    source = _fake_source()
    canonical = (tmp_path / marker).resolve()
    monkeypatch.setattr(qualification, "_canonical_run_directory", lambda: canonical)
    outer = _synthetic_outer_authority(stage="development", source=source)
    pin: qualification._PinnedDirectory | None = None
    seal: qualification._RunnerInvocationSeal | None = None
    authorization: qualification._RunAuthorization | None = None
    ledger: qualification._AccessLedger | None = None
    config = {
        "schema": "rgbd_known_action_synthetic_materializer_config_v1",
        "resolved_config_sha256": qualification.FROZEN_CONFIG_SHA256,
        "device": "cpu",
        "precision": "float32",
    }
    try:
        pin = qualification._acquire_pinned_directory(
            canonical,
            create=True,
            canonical=True,
            outer_authority=outer,
        )
        seal = qualification._mint_runner_invocation_seal_from_outer(
            outer,
            stage="development",
            directory_pin=pin,
            source_provenance=source,
        )
        formal_authorization = qualification._formal_authorization_binding(outer)
        bindings = qualification._ledger_bindings(
            stage="development",
            execution_mode="formal",
            directory_pin=pin,
            source_provenance=source,
            formal_authorization=formal_authorization,
        )
        ledger_path = qualification._artifact_paths(pin)["development_ledger"]
        authorization = qualification._authorize_ledger_creation(
            invocation_seal=seal,
            stage="development",
            directory_pin=pin,
            source_provenance=source,
            bindings=bindings,
            ledger_path=ledger_path,
        )
        ledger = qualification._AccessLedger(
            ledger_path,
            stage="development",
            bindings=bindings,
            directory_pin=pin,
            authorization=authorization,
        )
        yield ledger, pin, config
    finally:
        if ledger is not None:
            owned_manifests: list[qualification._ManifestCapability] = []
            for registration in qualification._BATCH_REGISTRY.values():
                if registration.manifest._ledger is ledger and all(
                    registration.manifest is not item for item in owned_manifests
                ):
                    owned_manifests.append(registration.manifest)
            for registration in qualification._TRUTH_AUTHORITY_REGISTRY.values():
                if registration[1]._ledger is ledger and all(
                    registration[1] is not item for item in owned_manifests
                ):
                    owned_manifests.append(registration[1])
            for manifest in owned_manifests:
                with suppress(BaseException):
                    manifest.abort()
            for registration in qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY.values():
                if registration.ledger is ledger:
                    with suppress(BaseException):
                        qualification._revoke_trusted_materializer_port(registration.port)
                    if registration.state == "active":
                        registration.state = "revoked"
            for registration in qualification._MATERIALIZER_SPLIT_VAULT_REGISTRY.values():
                if (
                    registration.ledger is ledger
                    and registration.state in qualification.MATERIALIZER_LIVE_VAULT_STATES
                ):
                    with suppress(BaseException):
                        qualification._scrub_materializer_vault(
                            registration,
                            state="revoked",
                        )
            for (
                registration
            ) in qualification._MATERIALIZER_CONSTRUCTION_AUTHORITY_REGISTRY.values():
                if registration.ledger is ledger and registration.state == "issued":
                    registration.state = "revoked"
            for identity, registration in tuple(qualification._TRUTH_AUTHORITY_REGISTRY.items()):
                if registration[1]._ledger is ledger:
                    qualification._TRUTH_AUTHORITY_REGISTRY.pop(identity, None)
                    qualification._TRUTH_AUTHORITY_BINDING_REGISTRY.pop(identity, None)
            for identity, registration in tuple(qualification._BATCH_REGISTRY.items()):
                if registration.manifest._ledger is ledger:
                    qualification._BATCH_REGISTRY.pop(identity, None)
            for identity, registration in tuple(qualification._TOKEN_REGISTRY.items()):
                if registration[1].manifest._ledger is ledger:
                    qualification._TOKEN_REGISTRY.pop(identity, None)
            for identity, registration in tuple(qualification._CONSUMED_ORDINAL_REGISTRY.items()):
                if registration[1]._manifest._ledger is ledger:
                    qualification._CONSUMED_ORDINAL_REGISTRY.pop(identity, None)
            for identity, registration in tuple(
                qualification._MATERIALIZER_PUBLIC_BATCH_EVALUATION_REGISTRY.items()
            ):
                if registration[1]._ledger is ledger:
                    qualification._MATERIALIZER_PUBLIC_BATCH_EVALUATION_REGISTRY.pop(
                        identity,
                        None,
                    )
            qualification._LEDGER_REGISTRY.pop(id(ledger), None)
            qualification._LEDGER_SLOT_REGISTRY.pop(
                ("development", os.fspath(canonical)),
                None,
            )
        if authorization is not None:
            qualification._RUN_AUTHORIZATION_REGISTRY.pop(id(authorization), None)
        if seal is not None:
            qualification._INVOCATION_SEAL_REGISTRY.pop(id(seal), None)
        if pin is not None and qualification._PIN_REGISTRY.get(id(pin)) is pin:
            qualification._release_pinned_directory(pin)
        _forget_synthetic_outer_authority(outer)
        if clear_after:
            qualification._clear_ephemeral_registries_for_tests()


def _materializer_sensor_tensors() -> dict[str, Any]:
    torch = qualification.torch
    timestamps = (
        torch.arange(qualification.HISTORY_FRAME_COUNT, dtype=torch.float32)
        .div(20.0)
        .repeat(qualification.BATCH_SIZE, 1)
        .contiguous()
    )
    angles = torch.arange(qualification.HISTORY_FRAME_COUNT, dtype=torch.float32).mul(0.012)
    cosine = torch.cos(angles)
    sine = torch.sin(angles)
    camera_rows = torch.zeros(
        qualification.HISTORY_FRAME_COUNT,
        4,
        4,
        dtype=torch.float32,
    )
    camera_rows[:, 0, 0] = cosine
    camera_rows[:, 0, 2] = sine
    camera_rows[:, 1, 1] = 1.0
    camera_rows[:, 2, 0] = -sine
    camera_rows[:, 2, 2] = cosine
    camera_rows[:, 3, 3] = 1.0
    camera_rows[:, 0, 3] = qualification.MATERIALIZER_CAMERA_ORBIT_RADIUS_METERS * cosine
    camera_rows[:, 1, 3] = qualification.MATERIALIZER_CAMERA_HEIGHT_METERS
    camera_rows[:, 2, 3] = qualification.MATERIALIZER_CAMERA_ORBIT_RADIUS_METERS * sine
    world_from_camera = camera_rows.repeat(qualification.BATCH_SIZE, 1, 1, 1).contiguous()
    intrinsics = (
        torch.tensor(
            qualification.MATERIALIZER_CAMERA_INTRINSICS,
            dtype=torch.float32,
        )
        .reshape(1, 1, 3, 3)
        .repeat(
            qualification.BATCH_SIZE,
            qualification.HISTORY_FRAME_COUNT,
            1,
            1,
        )
        .contiguous()
    )
    return {
        "rgb": torch.full(
            (
                qualification.BATCH_SIZE,
                qualification.HISTORY_FRAME_COUNT,
                3,
                64,
                64,
            ),
            0.5,
            dtype=torch.float32,
        ),
        "depth": torch.ones(
            (
                qualification.BATCH_SIZE,
                qualification.HISTORY_FRAME_COUNT,
                1,
                64,
                64,
            ),
            dtype=torch.float32,
        ),
        "timestamps": timestamps,
        "world_from_camera": world_from_camera,
        "intrinsics": intrinsics,
    }


def _materializer_safe_task_tensors(
    *,
    start: int = 0,
    split: qualification.Split = "development",
) -> dict[str, Any]:
    torch = qualification.torch
    ordinals = tuple(range(start, start + qualification.BATCH_SIZE))
    timestamps, impulses = qualification._expected_materializer_action_rows(
        split=split,
        ordinals=ordinals,
    )
    palette = torch.tensor(
        qualification.MATERIALIZER_HANDLE_PROTOTYPE_PALETTE,
        dtype=torch.float32,
    )
    handles = torch.tensor(
        [(ordinal % 16) // 8 for ordinal in ordinals],
        dtype=torch.int64,
    )
    goal = torch.tensor([0.5, 0.8, 0.0], dtype=torch.float32).repeat(4, 1)
    opposite = torch.tensor([-0.5, 0.8, 0.0], dtype=torch.float32).repeat(4, 1)
    return {
        "observable_handle_prototype": palette[handles].contiguous(),
        "candidate_timestamps": timestamps.contiguous(),
        "candidate_impulses_world": impulses.contiguous(),
        "goal_positions_world": goal.contiguous(),
        "opposite_goal_positions_world": opposite.contiguous(),
        "alternate_handle_prototype": palette[1 - handles].contiguous(),
    }


def _materializer_public_runtime_tensors() -> dict[str, Any]:
    torch = qualification.torch
    horizons = len(qualification.HORIZONS_SECONDS)
    slots = torch.tensor([[11, 17], [23, 29], [31, 37], [41, 43]], dtype=torch.int64)
    return {
        "resolved_persistent_id": slots[:, 0].clone(),
        "persistent_id_by_slot": slots.clone(),
        "candidate_positions": torch.zeros(
            qualification.BATCH_SIZE,
            qualification.CANDIDATE_COUNT,
            horizons,
            2,
            3,
            dtype=torch.float32,
        ),
        "candidate_velocities": torch.zeros(
            qualification.BATCH_SIZE,
            qualification.CANDIDATE_COUNT,
            horizons,
            2,
            3,
            dtype=torch.float32,
        ),
        "candidate_total_cost": torch.ones(
            qualification.BATCH_SIZE,
            qualification.CANDIDATE_COUNT,
            dtype=torch.float32,
        ),
        "selected_display_index": torch.zeros(
            qualification.BATCH_SIZE,
            dtype=torch.int64,
        ),
        "none_positions": torch.zeros(
            qualification.BATCH_SIZE,
            horizons,
            2,
            3,
            dtype=torch.float32,
        ),
        "none_velocities": torch.zeros(
            qualification.BATCH_SIZE,
            horizons,
            2,
            3,
            dtype=torch.float32,
        ),
    }


def _fake_public_evaluation_without_consumption(
    request: qualification.PublicBatchEvaluationRequest,
) -> dict[str, Any]:
    evidence = _fake_numeric_evidence(request, request._manifest._ledger.record)
    formal_authorities = sorted(
        (
            registration[0]
            for registration in qualification._CONSUMED_ORDINAL_REGISTRY.values()
            if registration[1] is request
        ),
        key=lambda authority: authority.ordinal,
    )
    if formal_authorities:
        assert [authority.ordinal for authority in formal_authorities] == list(request.ordinals)
        for row, authority in zip(
            evidence["row_receipts"],
            formal_authorities,
            strict=True,
        ):
            body = dict(row)
            body.pop("receipt_sha256")
            body["token_receipt"] = qualification.copy.deepcopy(authority.token_receipt)
            row.clear()
            row.update({**body, "receipt_sha256": qualification.canonical_sha256(body)})
        evidence["ordered_row_receipts_sha256"] = qualification.canonical_sha256(
            [row["receipt_sha256"] for row in evidence["row_receipts"]]
        )
    return {
        "schema": "rgbd_known_action_public_batch_evaluation_v2",
        "split": request.split,
        "ordinals": list(request.ordinals),
        "bundle_count": qualification.BATCH_SIZE,
        "candidate_count": qualification.BATCH_SIZE * qualification.CANDIDATE_COUNT,
        "public_evidence": evidence,
        "public_evidence_sha256": qualification.canonical_sha256(evidence),
        "public_metrics": {
            "schema": "rgbd_known_action_public_batch_metrics_v1",
            "public_numeric_evidence_sha256": qualification.canonical_sha256(evidence),
            "public_row_count": qualification.BATCH_SIZE,
            "public_candidate_count": (qualification.BATCH_SIZE * qualification.CANDIDATE_COUNT),
        },
        "public_resources": {
            "schema": "rgbd_known_action_public_resource_observation_v1",
            "planning_seconds": 0.0,
            "planning_rss_delta_bytes": 0.0,
            "process_max_rss_bytes": 0.0,
            "live_result_storage_bytes": 0.0,
            "public_call_count": 1,
        },
    }


def _construct_synthetic_materializer(
    ledger: qualification._AccessLedger,
    config: dict[str, Any],
) -> tuple[
    qualification._EvaluatorConstructionAuthority,
    qualification._FormalKnownActionEvaluator,
    qualification._TrustedMaterializerPort,
]:
    authority = ledger.mint_formal_evaluator_authority(config)
    evaluator = qualification._FormalKnownActionEvaluator(config, authority=authority)
    return authority, evaluator, evaluator._materializer_port


def _materializer_issue_public_batch(
    *,
    manifest: qualification._ManifestCapability,
    port: qualification._TrustedMaterializerPort,
    vault: qualification._MaterializerSplitVault,
    start: int,
) -> tuple[
    qualification.PublicBatchEvaluationRequest,
    qualification._PublicMaterializationEnvelope,
    dict[str, Any],
]:
    ordinals = tuple(range(start, start + qualification.BATCH_SIZE))
    batch = manifest.begin_batch(ordinals)
    request = qualification.PublicBatchEvaluationRequest(
        split=manifest.split,
        ordinals=ordinals,
        tokens=batch.tokens,
        _manifest=manifest,
        _batch=batch,
    )
    envelope = qualification._issue_public_materialization_envelope(
        port,
        vault,
        request,
        sensor_tensors=_materializer_sensor_tensors(),
        safe_task_tensors=_materializer_safe_task_tensors(start=start),
        public_runtime_tensors=_materializer_public_runtime_tensors(),
    )
    evaluation = _fake_public_evaluation_without_consumption(request)
    return request, envelope, evaluation


def _materializer_issue_and_register_public_batch(
    *,
    manifest: qualification._ManifestCapability,
    port: qualification._TrustedMaterializerPort,
    vault: qualification._MaterializerSplitVault,
    start: int,
) -> tuple[
    qualification.PublicBatchEvaluationRequest,
    qualification._PublicMaterializationEnvelope,
    dict[str, Any],
    dict[str, Any],
]:
    request, envelope, evaluation = _materializer_issue_public_batch(
        manifest=manifest,
        port=port,
        vault=vault,
        start=start,
    )
    hashes = qualification._register_public_materializer_evaluation(
        port,
        vault,
        envelope,
        evaluation,
    )
    return request, envelope, evaluation, hashes


def _materializer_public_batch(
    *,
    manifest: qualification._ManifestCapability,
    port: qualification._TrustedMaterializerPort,
    vault: qualification._MaterializerSplitVault,
    start: int,
) -> tuple[
    qualification.PublicBatchEvaluationRequest,
    qualification._PublicMaterializationEnvelope,
    dict[str, Any],
    dict[str, Any],
]:
    request, envelope, evaluation, hashes = _materializer_issue_and_register_public_batch(
        manifest=manifest,
        port=port,
        vault=vault,
        start=start,
    )
    receipt = manifest.complete_batch(request._batch, evaluation)
    return request, envelope, hashes, receipt


def _materializer_public_surface(*, ordinal: int = 0) -> dict[str, Any]:
    return {
        "schema": "rgbd_known_action_public_materialization_surface_v1",
        "split": "development",
        "ordinal": ordinal,
        "batch_index": ordinal // qualification.BATCH_SIZE,
        "fixed_goal_horizon_seconds": (qualification.MATERIALIZER_FIXED_GOAL_HORIZON_SECONDS),
        "materializer_port_sha256": "1" * 64,
        "token_receipt_sha256": "2" * 64,
        "sensor_sha256": "3" * 64,
        "safe_task_sha256": "4" * 64,
        "public_runtime_sha256": "5" * 64,
    }


def _assert_materializer_vault_scrubbed(
    vault: qualification._MaterializerSplitVault,
) -> None:
    registration = qualification._MATERIALIZER_SPLIT_VAULT_REGISTRY[id(vault)]
    assert registration.state in {"failed", "revoked", "consumed"}
    assert registration.private_request is None
    assert registration.private_request_binding is None
    owned_envelopes = {id(envelope): envelope for envelope in registration.envelopes}
    for identity, child in qualification._PUBLIC_MATERIALIZATION_ENVELOPE_REGISTRY.items():
        if (
            type(identity) is int
            and type(child) is qualification._MaterializerEnvelopeRegistration
            and type(child.envelope) is qualification._PublicMaterializationEnvelope
            and (
                child.vault is vault
                or (
                    type(child.envelope.vault_identity) is int
                    and child.envelope.vault_identity == id(vault)
                )
            )
        ):
            owned_envelopes[identity] = child.envelope
    for identity, _envelope in owned_envelopes.items():
        envelope_registration = qualification._PUBLIC_MATERIALIZATION_ENVELOPE_REGISTRY[identity]
        assert envelope_registration.request is None
        assert envelope_registration.sensor_batch is None
        assert envelope_registration.task_batch is None
        assert envelope_registration.runtime_tensors is None
        assert envelope_registration.public_bodies is None
        assert envelope_registration.blinding_nonces is None
        assert envelope_registration.evaluation_hashes is None
        assert envelope_registration.evaluation_hashes_binding_sha256 is None
        assert envelope_registration.state == "retired"
        for row in envelope_registration.rows:
            row_registration = qualification._SEALED_MATERIALIZATION_ROW_REGISTRY[id(row)]
            assert row_registration.state in {"retired", "revoked"}
    owned_sensor_batches = {
        identity: child
        for identity, child in qualification._PUBLIC_SENSOR_BATCH_REGISTRY.items()
        if type(identity) is int
        and type(child) is qualification._MaterializerTensorBatchRegistration
        and type(child.batch) is qualification._PublicSensorBatch
        and (
            child.vault is vault
            or (type(child.batch.vault_identity) is int and child.batch.vault_identity == id(vault))
            or id(child.envelope) in owned_envelopes
        )
    }
    for child in owned_sensor_batches.values():
        assert child.tensors is None
        assert child.state in {"retired", "revoked"}
    owned_task_batches = {
        identity: child
        for identity, child in qualification._SAFE_TASK_BATCH_REGISTRY.items()
        if type(identity) is int
        and type(child) is qualification._MaterializerTensorBatchRegistration
        and type(child.batch) is qualification._SafeTaskBatch
        and (
            child.vault is vault
            or (type(child.batch.vault_identity) is int and child.batch.vault_identity == id(vault))
            or id(child.envelope) in owned_envelopes
        )
    }
    for child in owned_task_batches.values():
        assert child.tensors is None
        assert child.state in {"retired", "revoked"}
    owned_rows = {
        identity: child
        for identity, child in qualification._SEALED_MATERIALIZATION_ROW_REGISTRY.items()
        if type(identity) is int
        and type(child) is qualification._MaterializerRowRegistration
        and type(child.row) is qualification._SealedMaterializationRow
        and (
            child.vault is vault
            or (type(child.row.vault_identity) is int and child.row.vault_identity == id(vault))
            or id(child.envelope) in owned_envelopes
        )
    }
    for child in owned_rows.values():
        assert child.state in {
            "retired",
            "revoked",
        }
    assert not any(
        type(value) is tuple
        and len(value) == 4
        and type(value[2]) is qualification._PublicMaterializationEnvelope
        and (
            id(value[2]) in owned_envelopes
            or (type(value[2].vault_identity) is int and value[2].vault_identity == id(vault))
        )
        for value in qualification._MATERIALIZER_PUBLIC_BATCH_EVALUATION_REGISTRY.values()
    )
    assert registration.commitment_inventory_sha256 is None
    assert registration.terminal_marker is qualification._MATERIALIZER_VAULT_TERMINAL_MARKER


def _assert_materializer_cleanup_refusal_then_restore(
    restore: Callable[[], None],
    *,
    hook_calls: list[str] | None = None,
) -> None:
    """Prove test-only disposal refuses corruption before exact restoration."""

    try:
        assert not qualification._exact_materializer_terminal_tombstone_cut()
        if hook_calls is not None:
            assert hook_calls == []
    finally:
        restore()
    assert qualification._exact_materializer_terminal_tombstone_cut()
    if hook_calls is not None:
        assert hook_calls == []


def _materializer_fill_and_seal_public_split(
    *,
    manifest: qualification._ManifestCapability,
    port: qualification._TrustedMaterializerPort,
    vault: qualification._MaterializerSplitVault,
) -> tuple[
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    dict[str, Any],
]:
    evaluations, receipts = _materializer_fill_public_split(
        manifest=manifest,
        port=port,
        vault=vault,
    )
    commitment_digest = qualification._finalize_materializer_public_split(port, vault)
    finalization = _FakeEvaluator().finalize_public_split(
        manifest.split,
        evaluations,
    )
    _summary, seal = manifest.seal_public_split(
        finalization,
        batch_results=evaluations,
        receipts=receipts,
    )
    assert (
        qualification._MATERIALIZER_SPLIT_VAULT_REGISTRY[id(vault)].commitment_inventory_sha256
        == commitment_digest
    )
    return evaluations, receipts, seal


def _materializer_fill_public_split(
    *,
    manifest: qualification._ManifestCapability,
    port: qualification._TrustedMaterializerPort,
    vault: qualification._MaterializerSplitVault,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    evaluations: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    for start in range(0, qualification.SCENES_PER_SPLIT, qualification.BATCH_SIZE):
        request, _envelope, evaluation, _hashes = _materializer_issue_and_register_public_batch(
            manifest=manifest,
            port=port,
            vault=vault,
            start=start,
        )
        receipt = manifest.complete_batch(request._batch, evaluation)
        evaluations.append(evaluation)
        receipts.append(receipt)
    return tuple(evaluations), tuple(receipts)


def _seal_fake_public_split(
    ledger: qualification._AccessLedger,
    *,
    split: str = "development",
    evaluator: _FakeEvaluator | None = None,
) -> tuple[
    qualification._ManifestCapability,
    _FakeEvaluator,
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    fake = _FakeEvaluator() if evaluator is None else evaluator
    manifest = qualification._ManifestCapability(split=split, ledger=ledger)
    batch_results: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    for start in range(0, qualification.SCENES_PER_SPLIT, qualification.BATCH_SIZE):
        ordinals = tuple(range(start, start + qualification.BATCH_SIZE))
        batch = manifest.begin_batch(ordinals)
        request = qualification.PublicBatchEvaluationRequest(
            split=split,
            ordinals=ordinals,
            tokens=batch.tokens,
            _manifest=manifest,
            _batch=batch,
        )
        evaluation = fake.evaluate_public_batch(request)
        batch_results.append(evaluation)
        receipts.append(manifest.complete_batch(batch, evaluation))
    finalization = fake.finalize_public_split(split, tuple(batch_results))
    summary, seal = manifest.seal_public_split(
        finalization,
        batch_results=batch_results,
        receipts=receipts,
    )
    return manifest, fake, batch_results, receipts, summary, seal


def _terminate_test_ledger(
    pin: qualification._PinnedDirectory,
    ledger: qualification._AccessLedger,
    *,
    message: str = "test abort",
) -> None:
    source = ledger.record["bindings"]["source_provenance"]
    digest = qualification._persist_exception_report(
        directory_pin=pin,
        stage="development",
        source_provenance=source,
        ledger=ledger,
        error=RuntimeError(message),
        results=(),
    )
    assert type(digest) is str and len(digest) == 64


def _complete_fake_development_before_publication(
    pin: qualification._PinnedDirectory,
) -> tuple[
    qualification._AccessLedger,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    ledger, _ = _new_development_ledger(pin)
    source = _fake_source()

    def guard(_label: str) -> None:
        ledger._verify_disk()

    result = qualification._evaluate_split(
        split="development",
        ledger=ledger,
        evaluator=_FakeEvaluator(),
        boundary_guard=guard,
    )
    checkpoint = qualification._save_review_checkpoint(
        pin,
        development_result=result,
        source_provenance=source,
        execution_mode="fake_test",
    )
    template = qualification._report_root(
        stage="development",
        execution_mode="fake_test",
        source_provenance=source,
        results=[result],
        pending_ledger_sha256=None,
        checkpoint=checkpoint,
    )
    return ledger, source, result, checkpoint, template


def _complete_fake_qualification_before_publication(
    pin: qualification._PinnedDirectory,
    *,
    source: dict[str, Any],
    reviewed: dict[str, str],
) -> tuple[qualification._AccessLedger, list[dict[str, Any]], dict[str, Any]]:
    bindings = qualification._ledger_bindings(
        stage="qualification",
        execution_mode="fake_test",
        directory_pin=pin,
        source_provenance=source,
        reviewed_development=reviewed,
    )
    path = qualification._artifact_paths(pin)["qualification_ledger"]
    authorization = qualification._authorize_ledger_creation(
        invocation_seal=qualification._mint_fake_runner_invocation_seal_for_tests(
            stage="qualification",
            directory_pin=pin,
            source_provenance=source,
            reviewed_development=reviewed,
        ),
        stage="qualification",
        directory_pin=pin,
        source_provenance=source,
        bindings=bindings,
        ledger_path=path,
        reviewed_development=reviewed,
    )
    ledger = qualification._AccessLedger(
        path,
        stage="qualification",
        bindings=bindings,
        directory_pin=pin,
        authorization=authorization,
    )

    def guard(_label: str) -> None:
        ledger._verify_disk()

    results = [
        qualification._evaluate_split(
            split=split,
            ledger=ledger,
            evaluator=_FakeEvaluator(),
            boundary_guard=guard,
        )
        for split in qualification.PROTECTED_SPLITS
    ]
    template = qualification._report_root(
        stage="qualification",
        execution_mode="fake_test",
        source_provenance=source,
        results=results,
        pending_ledger_sha256=None,
        reviewed_development=reviewed,
    )
    return ledger, results, template


def _forget_fake_process_state() -> None:
    """Model a stopped process while leaving exact durable artifacts untouched."""

    qualification._clear_ephemeral_registries_for_tests()


def test_protocol_is_seedless_json_native_and_names_batched_k8() -> None:
    protocol = qualification.bridge_protocol()

    assert json.loads(json.dumps(protocol, allow_nan=False)) == protocol
    assert protocol["name"] == "rgbd_known_action_planning_v2"
    assert protocol["architecture_version"] == 2
    assert protocol["architecture_attempt"] == 1
    assert protocol["maximum_architecture_attempts"] == 1
    assert protocol["addressing"]["kind"] == "seedless_split_plus_ordinal"
    assert "seed" not in protocol["addressing"]
    assert protocol["addressing"]["batch_size"] == 4
    assert protocol["execution"]["label"] == "batched K=8"
    assert protocol["execution"]["candidate_vectorized"] is False
    assert protocol["execution"]["candidates"] == 8
    assert protocol["planner"]["runtime_truth_mapping_permitted"] is False
    assert protocol["planner"]["formal_target_resolver_status"] == (
        "causal_public_appearance_resolver"
    )
    gradient = protocol["gradient_contract"]
    assert gradient["action_conditioned_sensor_vjp_groups"] == [
        [0, 9, 18, 27],
        [36, 45, 54, 63],
    ]
    assert gradient["action_conditioned_sensor_vjp_groups_sha256"] == (
        "88d207bd5d93d47fb0bd8dbac3c8b8d4ceb8c51ee02dc73c9bd1b994efa08d7d"
    )
    assert gradient["action_conditioned_sensor_vjp_group_count"] == 2
    assert gradient["action_conditioned_sensor_vjp_display_indices"] == [
        0,
        7,
        6,
        5,
        4,
        3,
        2,
        1,
    ]
    assert gradient["action_conditioned_sensor_vjp_display_indices_sha256"] == (
        "c8a21da525d175798f683533ef272b71a8ccf6e73eaf7d73bca365f351c9d85d"
    )
    assert protocol["action_gate_surface"] == {
        "constraint_count": 119,
        "constraint_manifest_sha256": (
            "68409749ac369006cd1fde13685e969995f82293b0a22296f7b33fa3c93661d4"
        ),
    }
    boundary = protocol["evaluator_data_boundary"]
    assert boundary["private_truth_derivation"] == (
        "post_durable_public_seal_after_private_authority_binding"
    )
    assert boundary["private_truth_release"] == "one_shot_split_scoring_only"
    assert boundary["durable_phase_order"] == [
        "public_evaluating",
        "public_complete",
        "truth_evaluating",
        "passed_or_failed",
    ]
    assert boundary["truth_rows_in_observation_packet"] is False
    assert boundary["truth_rows_in_model_plan"] is False
    materializer = protocol["formal_materializer_security_kernel"]
    assert materializer["status"] == "required_formal_path"
    assert materializer["public_sensor_tensor_keys"] == list(
        qualification.PUBLIC_SENSOR_TENSOR_KEYS
    )
    assert materializer["safe_task_tensor_keys"] == list(qualification.SAFE_TASK_TENSOR_KEYS)
    assert materializer["public_runtime_tensor_keys"] == list(
        qualification.PUBLIC_RUNTIME_TENSOR_KEYS
    )
    assert materializer["fixed_goal_horizon_seconds"] == 2.0
    assert materializer["sensor_depth_meters"] == [0.0, 8.0]
    assert materializer["sensor_timestep_seconds"] == 0.05
    assert materializer["commitment_frames"] == ("uint64_little_endian_json_then_32_byte_nonce")
    assert materializer["resolved_ids_are_public_outputs_only"] is True
    assert materializer["candidate_and_selected_indices_are_public_planner_outputs_only"] is (True)
    assert materializer["private_body_or_nonce_in_durable_public_evidence"] is False
    assert protocol["scene_freeze"]["status"] == "frozen"


def test_gate_schema_and_all_thresholds_are_exact() -> None:
    expected_constraint_count = 119
    expected_constraint_sha256 = "68409749ac369006cd1fde13685e969995f82293b0a22296f7b33fa3c93661d4"
    expected = {
        "batched_k8_planning_median_seconds": ("le", 0.10),
        "batched_k8_planning_p95_seconds": ("le", 0.15),
        "batched_k8_planning_rss_delta_bytes": ("le", 134_217_728.0),
        "batched_k8_process_max_rss_bytes": ("le", 2_684_354_560.0),
        "batched_k8_live_result_storage_bytes": ("le", 262_144.0),
        "public_action_delta_position_max_abs_m": ("le", 1.0e-6),
        "public_action_delta_velocity_max_abs_mps": ("le", 1.0e-6),
        "public_action_conditioned_position_max_abs_m": ("le", 2.0e-6),
        "public_action_conditioned_velocity_max_abs_mps": ("le", 2.0e-6),
        "public_planner_cost_max_abs_error_m2": ("le", 1.0e-7),
        "certificate_minimum_truth_cost_gap_m2": ("ge", 2.39e-4),
        "minimum_end_to_end_winner_cost_gap_m2": ("ge", 2.0e-4),
        "winner_accuracy_fraction": ("eq", 1.0),
        "optimal_candidate_terminal_error_l2_max_m": ("le", 1.0e-3),
        "optimal_candidate_terminal_cost_max_m2": ("le", 1.0e-6),
        "certificate_minimum_palette_handle_margin": ("ge", 0.5),
        "minimum_observed_appearance_handle_margin": ("ge", 0.1),
        "action_jacobian_position_max_abs_error": ("le", 2.0e-6),
        "action_jacobian_velocity_max_abs_error": ("le", 2.0e-6),
        "action_pre_boundary_gradient_max_abs": ("eq", 0.0),
        "action_cross_bundle_gradient_max_abs": ("eq", 0.0),
        "action_cross_candidate_cost_gradient_max_abs": ("eq", 0.0),
        "minimum_post_action_impulse_vjp_l1": ("ge", 1.0e-3),
        "maximum_post_action_impulse_vjp_l1": ("le", 1.0e4),
    }
    actual = {rule.name: (rule.operator, rule.threshold) for rule in qualification.GATE_RULES}
    rows = [[rule.name, rule.operator, float(rule.threshold)] for rule in qualification.GATE_RULES]
    encoded_rows = json.dumps(
        rows,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")

    assert len(rows) == expected_constraint_count
    assert hashlib.sha256(encoded_rows).hexdigest() == expected_constraint_sha256
    assert expected_constraint_count == qualification.ACTION_GATE_CONSTRAINT_COUNT
    assert expected_constraint_sha256 == qualification.ACTION_GATE_CONSTRAINT_SHA256
    assert len(actual) == len(qualification.GATE_METRIC_SCHEMA)
    assert len(actual) == len(set(actual))
    for name, rule in expected.items():
        assert actual[name] == rule
    assert qualification.gate_failures(_passing_metrics()) == []
    with pytest.raises(ValueError, match="metric schema"):
        qualification.gate_failures({})

    records, scene = _synthetic_action_reducer_records()
    passing = _passing_metrics()
    gradient_names = {
        "action_jacobian_position_max_abs_error",
        "action_jacobian_velocity_max_abs_error",
        "action_boundary_velocity_jacobian_error",
        "action_pre_boundary_gradient_max_abs",
        "action_boundary_position_gradient_max_abs",
        "action_cross_bundle_gradient_max_abs",
        "action_cross_candidate_cost_gradient_max_abs",
        "action_own_cost_gradient_finite_fraction",
        "action_own_cost_graph_connected_fraction",
        "action_own_cost_gradient_max_l1",
        "action_distractor_gradient_max_abs",
        "action_off_axis_gradient_max_abs",
        "action_candidate_gradient_finite_fraction",
        "action_candidate_graph_connected_fraction",
        "minimum_post_action_impulse_vjp_l1",
        "maximum_post_action_impulse_vjp_l1",
    }
    gradient_metrics = {name: passing[name] for name in gradient_names}
    sensor_metrics = {
        name: value for name, value in passing.items() if name.startswith("action_sensor_vjp_")
    }
    reduced = qualification._aggregate_action_metrics(
        records,
        scene,
        action_gradient_metrics=gradient_metrics,
        action_sensor_metrics=sensor_metrics,
    )
    assert tuple(reduced) == qualification.GATE_METRIC_SCHEMA
    assert len(reduced) == expected_constraint_count
    assert all(type(value) is float and np.isfinite(value) for value in reduced.values())
    assert qualification.gate_failures(reduced) == []

    row_local_failure = list(records)
    row_local_failure[0] = dict(row_local_failure[0])
    row_local_failure[0]["rejection_counts"] = dict(row_local_failure[0]["rejection_counts"])
    row_local_failure[0]["rejection_counts"]["invalid"] = 3
    rejected = qualification._aggregate_action_metrics(
        row_local_failure,
        scene,
        action_gradient_metrics=gradient_metrics,
        action_sensor_metrics=sensor_metrics,
    )
    assert qualification.gate_failures(rejected) == [
        "invalid_persistent_id_rejection_fraction=0.984375 violates eq 1"
    ]

    incomplete_sensor = dict(sensor_metrics)
    incomplete_sensor.pop("action_sensor_vjp_rgb_min_l1")
    with pytest.raises(RuntimeError, match="action metric reducer differs from the exact schema"):
        qualification._aggregate_action_metrics(
            records,
            scene,
            action_gradient_metrics=gradient_metrics,
            action_sensor_metrics=incomplete_sensor,
        )


def test_frozen_accepted_owner_manifest_and_pure_evidence_reducer_are_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        [name, operator, float(threshold)]
        for name, operator, threshold in qualification.FROZEN_ACCEPTED_ORBITAL_CONSTRAINTS
    ]
    encoded_rows = json.dumps(
        rows,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    names = sorted({row[0] for row in rows})
    encoded_names = json.dumps(
        names,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    duplicate = [row for row in rows if row[0] == "birth_slot_physical_zero_fraction"]
    assert len(rows) == 686
    assert len(names) == 685
    assert hashlib.sha256(encoded_rows).hexdigest() == (
        "dd5752bf552ef73daac233bd60703c360aab99be22ca8e48ef7c15cdb848bffd"
    )
    assert hashlib.sha256(encoded_names).hexdigest() == (
        "2763e8edd2149d30a6b95e89bdeda929bcd3c7641283939ab9006d0f3d36f8eb"
    )
    assert duplicate == [
        ["birth_slot_physical_zero_fraction", "ge", 0.25],
        ["birth_slot_physical_zero_fraction", "le", 0.75],
    ]

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("pure durable inherited validation touched the accepted owner")

    monkeypatch.setattr(qualification.importlib, "import_module", forbidden)
    monkeypatch.setattr(qualification, "_accepted_orbital_owner", forbidden)
    metrics = _passing_inherited_metrics()
    evidence = qualification._inherited_orbital_evidence(metrics)
    assert qualification._validated_inherited_orbital_evidence(evidence) == evidence
    assert evidence["constraint_count"] == 686
    assert evidence["constraint_manifest_sha256"] == hashlib.sha256(encoded_rows).hexdigest()
    assert evidence["metrics_sha256"] == qualification.canonical_sha256(metrics)
    assert evidence["failures"] == []
    assert evidence["failures_sha256"] == qualification.canonical_sha256([])
    body = dict(evidence)
    supplied = body.pop("evidence_sha256")
    assert supplied == qualification.canonical_sha256(body)

    failed = dict(metrics)
    failed["current_position_rmse_m"] = 0.0100001
    assert qualification._accepted_orbital_gate_failures(failed) == [
        "current_position_rmse_m:0.0100001>0.01"
    ]
    for rejected in (True, 1, float("nan")):
        malformed = dict(metrics)
        malformed["current_position_rmse_m"] = rejected
        with pytest.raises((TypeError, ValueError)):
            qualification._accepted_orbital_gate_failures(malformed)
    wrong_schema = dict(metrics)
    wrong_schema["arbitrary"] = wrong_schema.pop("current_position_rmse_m")
    with pytest.raises(ValueError, match="metric schema"):
        qualification._accepted_orbital_gate_failures(wrong_schema)


def test_json_native_and_strict_loader_reject_ambiguous_values() -> None:
    assert qualification._json_native({"tuple": (1, 2)}) == {"tuple": [1, 2]}
    recursive: list[Any] = []
    recursive.append(recursive)
    with pytest.raises(ValueError, match="recursive"):
        qualification._json_native(recursive)
    with pytest.raises(ValueError, match="duplicate"):
        qualification._strict_json_loads(b'{"x":1,"x":2}', label="duplicate")
    with pytest.raises(ValueError, match="nonfinite"):
        qualification._strict_json_loads(b'{"x":NaN}', label="nonfinite")


def test_two_phase_fake_split_orders_public_seal_truth_authority_and_completion(
    tmp_path: Path,
) -> None:
    with _pinned_run(tmp_path) as pin:
        ledger, _ = _new_development_ledger(pin)
        evaluator = _FakeEvaluator()
        boundaries: list[tuple[str, str, int]] = []

        def guard(label: str) -> None:
            ledger._verify_disk()
            boundaries.append((label, ledger.record["status"], ledger.record["generation"]))

        result = qualification._evaluate_split(
            split="development",
            ledger=ledger,
            evaluator=evaluator,
            boundary_guard=guard,
        )
        state = ledger.record["splits"]["development"]
        assert evaluator.events == [
            *[("public_batch", "development")] * 16,
            ("public_finalize", "development"),
            ("private_score", "development"),
        ]
        assert ("after development public seal", "development_public_complete", 34) in boundaries
        assert (
            "before development private scoring",
            "development_truth_evaluating",
            35,
        ) in boundaries
        assert (
            "after development private scoring",
            "development_passed",
            36,
        ) in boundaries
        assert state["status"] == "passed"
        assert state["public_summary"]["ordered_ordinals"] == list(range(64))
        assert state["public_summary"]["batch_count"] == 16
        assert state["public_summary"]["truth_access_count"] == 0
        assert state["public_seal"]["truth_access_count"] == 0
        assert state["private_scoring_receipt"]["scene_access_count"] == 0
        assert state["private_scoring_receipt"]["truth_access_count"] == 0
        assert state["private_scoring_receipt"]["private_row_commitments"] == []
        assert result["public_seal"] == state["public_seal"]
        assert result["private_scoring_receipt"] == state["private_scoring_receipt"]
        assert ledger.record["generation"] == 36
        _terminate_test_ledger(pin, ledger)


def test_two_phase_generation_trace_is_exact_and_every_cut_is_durable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _pinned_run(tmp_path) as pin:
        ledger, _ = _new_development_ledger(pin)
        trace: list[tuple[int, str, int, list[int] | None, str]] = [
            (0, "unopened", 0, None, qualification.canonical_sha256(ledger.record))
        ]
        original_replace = ledger._replace

        def capture_replace(candidate: dict[str, Any], *args: Any, **kwargs: Any) -> None:
            original_replace(candidate, *args, **kwargs)
            parsed = qualification._parse_ledger_bytes(
                ledger._last_bytes,
                label="two-phase generation trace",
            )
            supplied_record_sha256 = parsed.pop("record_sha256")
            assert parsed == ledger.record
            assert supplied_record_sha256 == qualification.canonical_sha256(ledger.record)
            ledger._verify_disk()
            state = ledger.record["splits"]["development"]
            trace.append(
                (
                    ledger.record["generation"],
                    state["status"],
                    state["public_next_ordinal"],
                    state["public_active_batch"],
                    qualification.canonical_sha256(ledger.record),
                )
            )

        monkeypatch.setattr(ledger, "_replace", capture_replace)
        result = qualification._evaluate_split(
            split="development",
            ledger=ledger,
            evaluator=_FakeEvaluator(),
            boundary_guard=lambda _label: ledger._verify_disk(),
        )
        assert result["passed"] is True
        assert [item[0] for item in trace] == list(range(37))
        assert trace[0][1:4] == ("unopened", 0, None)
        assert trace[1][1:4] == ("public_evaluating", 0, None)
        for batch_index in range(16):
            reserve_generation = 2 + 2 * batch_index
            commit_generation = 3 + 2 * batch_index
            start = batch_index * 4
            assert trace[reserve_generation][1:4] == (
                "public_evaluating",
                start,
                list(range(start, start + 4)),
            )
            assert trace[commit_generation][1:4] == (
                "public_evaluating",
                start + 4,
                None,
            )
        assert trace[34][1:4] == ("public_complete", 64, None)
        assert trace[35][1:4] == ("truth_evaluating", 64, None)
        assert trace[36][1:4] == ("passed", 64, None)
        assert all(type(item[4]) is str and len(item[4]) == 64 for item in trace)
        _terminate_test_ledger(pin, ledger)


def test_private_scoring_failure_receipt_durably_completes_failed_split(
    tmp_path: Path,
) -> None:
    source = _fake_source()
    with _pinned_run(tmp_path) as pin:
        evaluator = _FakeEvaluator(fail_metric=qualification.GATE_RULES[0].name)
        assert (
            qualification._execute_development_for_tests(
                directory_pin=pin,
                evaluator=evaluator,
                source_provenance=source,
            )
            == 1
        )
        paths = qualification._artifact_paths(pin)
        assert not qualification._pinned_exists(
            pin,
            paths["checkpoint"],
            label="failed private scoring checkpoint",
        )
        terminal = qualification._validate_terminal_ledger(
            qualification._pinned_read_bytes(
                pin,
                paths["development_ledger"],
                label="failed private scoring ledger",
            ),
            stage="development",
            expected_bindings=qualification._ledger_bindings(
                stage="development",
                execution_mode="fake_test",
                directory_pin=pin,
                source_provenance=source,
            ),
            expected_outcome="gate_failed",
        )
        state = terminal["splits"]["development"]
        assert terminal["status"] == "complete_failed"
        assert state["status"] == "failed"
        assert state["private_scoring_receipt"]["passed"] is False
        assert state["split_result"]["public_seal"] == state["public_seal"]
        assert state["split_result"]["private_scoring_receipt"] == state["private_scoring_receipt"]


def test_public_batch_schema_recursively_rejects_private_semantic_and_arbitrary_keys(
    tmp_path: Path,
) -> None:
    with _pinned_run(tmp_path) as pin:
        ledger, _ = _new_development_ledger(pin)
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        batch = manifest.begin_batch((0, 1, 2, 3))
        request = qualification.PublicBatchEvaluationRequest(
            split="development",
            ordinals=batch.ordinals,
            tokens=batch.tokens,
            _manifest=manifest,
            _batch=batch,
        )
        evaluation = _FakeEvaluator().evaluate_public_batch(request)
        mutations: list[tuple[str, Any, type[BaseException]]] = [
            (
                "truth",
                lambda value: value["public_metrics"].update({"truth_digest": "a" * 64}),
                PermissionError,
            ),
            (
                "label",
                lambda value: value["public_metrics"].update(
                    {"public_numeric_evidence_sha256": {"nested_label": "a" * 64}}
                ),
                PermissionError,
            ),
            (
                "specification",
                lambda value: value["public_resources"].update(
                    {"planning_seconds": {"specification_digest": "a" * 64}}
                ),
                PermissionError,
            ),
            (
                "arbitrary",
                lambda value: value["public_resources"].update({"arbitrary": 0.0}),
                ValueError,
            ),
            (
                "arbitrary_metric",
                lambda value: value["public_metrics"].update({"arbitrary_metric": 0.0}),
                ValueError,
            ),
            (
                "nested_evidence_truth",
                lambda value: value["public_evidence"]["row_receipts"][0].update(
                    {"nested_truth_commitment": "a" * 64}
                ),
                PermissionError,
            ),
        ]
        for _name, mutate, error_type in mutations:
            candidate = json.loads(json.dumps(evaluation))
            mutate(candidate)
            with pytest.raises(error_type):
                qualification._validated_public_batch_evaluation(
                    candidate,
                    split="development",
                    ordinals=(0, 1, 2, 3),
                )
        with suppress(PermissionError):
            manifest.abort()
        _terminate_test_ledger(pin, ledger)


def test_public_seal_rejects_canonically_rehashed_batch_receipt_contradiction(
    tmp_path: Path,
) -> None:
    with _pinned_run(tmp_path) as pin:
        ledger, _ = _new_development_ledger(pin)
        manifest, _, _, receipts, summary, _ = _seal_fake_public_split(ledger)
        mutated = json.loads(json.dumps(receipts))
        first = mutated[0]
        first["public_resources"]["planning_seconds"] = 1.0
        first["public_evaluation"]["public_resources"] = json.loads(
            json.dumps(first["public_resources"])
        )
        first["public_resources_sha256"] = qualification.canonical_sha256(first["public_resources"])
        first["public_result_sha256"] = qualification.canonical_sha256(first["public_evaluation"])
        first_body = dict(first)
        first_body.pop("receipt_sha256")
        first["receipt_sha256"] = qualification.canonical_sha256(first_body)
        assert (
            qualification._validate_batch_receipt(
                first,
                split="development",
                batch_index=0,
            )
            == first
        )
        with pytest.raises(ValueError, match="public split finalization|public summary"):
            qualification._validated_public_split_summary(
                summary,
                split="development",
                receipts=mutated,
            )
        manifest.abort()
        _terminate_test_ledger(pin, ledger)


def test_split_truth_request_is_post_seal_one_shot_reentrant_and_cross_split_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _pinned_run(tmp_path) as pin:
        ledger, _ = _new_development_ledger(pin)
        early = qualification._ManifestCapability(split="development", ledger=ledger)
        fake = _FakeEvaluator()
        for start in range(0, 64, 4):
            ordinals = tuple(range(start, start + 4))
            batch = early.begin_batch(ordinals)
            public_request = qualification.PublicBatchEvaluationRequest(
                split="development",
                ordinals=ordinals,
                tokens=batch.tokens,
                _manifest=early,
                _batch=batch,
            )
            evaluation = fake.evaluate_public_batch(public_request)
            early.complete_batch(batch, evaluation)
        assert ledger.record["splits"]["development"]["status"] == "public_evaluating"
        with pytest.raises(PermissionError, match="cannot precede the durable public seal"):
            early.begin_private_scoring(public_seal={})
        early.abort()
        _terminate_test_ledger(pin, ledger)

    sealed_root = tmp_path / "sealed"
    sealed_root.mkdir()
    with _pinned_run(sealed_root) as pin:
        ledger, _ = _new_development_ledger(pin)
        manifest, _, _, _, _, seal = _seal_fake_public_split(ledger)
        forged_seal = json.loads(json.dumps(seal))
        forged_seal["public_call_sha256"] = "0" * 64
        forged_body = dict(forged_seal)
        forged_body.pop("seal_sha256")
        forged_seal["seal_sha256"] = qualification.canonical_sha256(forged_body)
        with pytest.raises(ValueError, match="public seal/summary binding"):
            manifest.begin_private_scoring(public_seal=forged_seal)
        request = manifest.begin_private_scoring(public_seal=seal)
        assert ledger.record["status"] == "development_truth_evaluating"
        forged = qualification.PrivateSplitScoringRequest(
            split="selector",
            public_seal_sha256=request.public_seal_sha256,
            request_sha256=request.request_sha256,
            _authority=request._authority,
            _manifest=manifest,
        )
        with pytest.raises(PermissionError, match="forged, reentrant, or replayed"):
            forged.consume()

        original_verify = ledger._verify_disk
        reentered = False

        def verify_with_reentry() -> None:
            nonlocal reentered
            if not reentered:
                reentered = True
                with pytest.raises(PermissionError, match="forged, reentrant, or replayed"):
                    request.consume()
            original_verify()

        monkeypatch.setattr(ledger, "_verify_disk", verify_with_reentry)
        binding = request.consume()
        monkeypatch.setattr(ledger, "_verify_disk", original_verify)
        assert reentered is True
        with pytest.raises(PermissionError, match="forged, reentrant, or replayed"):
            request.consume()
        private_receipt = qualification._private_scoring_receipt(
            split="development",
            public_seal_sha256=request.public_seal_sha256,
            truth_request=binding,
            metrics=_passing_metrics(),
            inherited_orbital_evidence=qualification._inherited_orbital_evidence(
                _passing_inherited_metrics()
            ),
        )
        result = qualification._split_result_from_private_receipt(
            split="development",
            state=ledger.record["splits"]["development"],
            private_receipt=private_receipt,
        )
        forged_receipt = json.loads(json.dumps(private_receipt))
        forged_receipt["truth_request_sha256"] = "0" * 64
        forged_body = dict(forged_receipt)
        forged_body.pop("receipt_sha256")
        forged_receipt["receipt_sha256"] = qualification.canonical_sha256(forged_body)
        with pytest.raises(ValueError, match="private scoring receipt binding differs"):
            manifest.close(request, forged_receipt, result)
        assert qualification._TRUTH_AUTHORITY_REGISTRY[id(request._authority)][3] == "consumed"
        manifest.close(request, private_receipt, result)
        assert qualification._TRUTH_AUTHORITY_REGISTRY[id(request._authority)][3] == "retired"
        _terminate_test_ledger(pin, ledger)


@pytest.mark.parametrize(
    ("field", "mutate"),
    [
        ("split", lambda value: "selector"),
        ("directory_pin", lambda value: object()),
        ("ledger_identity", lambda value: value + 1),
        (
            "ledger_file_identity",
            lambda value: (*value[:-1], value[-1] + 1),
        ),
        ("ledger_generation", lambda value: value + 1),
        ("ledger_record_sha256", lambda value: "0" * 64),
        ("public_seal_sha256", lambda value: "1" * 64),
        ("request_bytes", lambda value: b"{}"),
        ("request_sha256", lambda value: "2" * 64),
        ("owner_thread", lambda value: value + 1),
        ("nonce", lambda _value: object()),
    ],
)
def test_truth_authority_individual_binding_tamper_fails_and_revokes_cleanly(
    tmp_path: Path,
    field: str,
    mutate: Any,
) -> None:
    with _pinned_run(tmp_path) as pin:
        ledger, _ = _new_development_ledger(pin)
        manifest, _, _, _, _, seal = _seal_fake_public_split(ledger)
        request = manifest.begin_private_scoring(public_seal=seal)
        authority = request._authority
        original = getattr(authority, field)
        object.__setattr__(authority, field, mutate(original))
        with pytest.raises((PermissionError, TypeError, ValueError)):
            request.consume()
        registration = qualification._TRUTH_AUTHORITY_REGISTRY[id(authority)]
        assert registration[:3] == (authority, manifest, request)
        assert registration[3:] == ("failed", None)
        manifest.abort()
        assert qualification._TRUTH_AUTHORITY_REGISTRY[id(authority)][3] == "revoked"
        _terminate_test_ledger(pin, ledger)


def test_truth_authority_cross_thread_consume_fails_and_revokes_cleanly(
    tmp_path: Path,
) -> None:
    with _pinned_run(tmp_path) as pin:
        ledger, _ = _new_development_ledger(pin)
        manifest, _, _, _, _, seal = _seal_fake_public_split(ledger)
        request = manifest.begin_private_scoring(public_seal=seal)
        observed: list[BaseException] = []

        def consume_on_other_thread() -> None:
            try:
                request.consume()
            except BaseException as error:
                observed.append(error)

        worker = qualification.threading.Thread(target=consume_on_other_thread)
        worker.start()
        worker.join()
        assert len(observed) == 1
        assert isinstance(observed[0], PermissionError)
        assert "authority was rebound" in str(observed[0]) or "ledger attributes differ" in str(
            observed[0]
        )
        authority = request._authority
        assert qualification._TRUTH_AUTHORITY_REGISTRY[id(authority)][3] == "failed"
        manifest.abort()
        assert qualification._TRUTH_AUTHORITY_REGISTRY[id(authority)][3] == "revoked"
        _terminate_test_ledger(pin, ledger)


def test_private_receipt_and_result_cannot_be_swapped_across_protected_splits(
    tmp_path: Path,
) -> None:
    source = _fake_source()
    with _pinned_run(tmp_path) as pin:
        assert (
            qualification._execute_development_for_tests(
                directory_pin=pin,
                evaluator=_FakeEvaluator(),
                source_provenance=source,
            )
            == 0
        )
        reviewed = _development_hashes(pin)
        ledger, results, template = _complete_fake_qualification_before_publication(
            pin,
            source=source,
            reviewed=reviewed,
        )
        selector, confirmation = results[:2]
        with pytest.raises(ValueError, match="split|address|binding"):
            qualification._validated_split_result(confirmation, split="selector")

        swapped = json.loads(json.dumps(selector))
        swapped["private_scoring_receipt"] = json.loads(
            json.dumps(confirmation["private_scoring_receipt"])
        )
        swapped["private_scoring_receipt_sha256"] = qualification.canonical_sha256(
            swapped["private_scoring_receipt"]
        )
        swapped["inherited_orbital_evidence"] = json.loads(
            json.dumps(confirmation["inherited_orbital_evidence"])
        )
        with pytest.raises(ValueError, match="split|binding"):
            qualification._validated_split_result(swapped, split="selector")

        ledger.prepare_publication(template)
        ledger.publish_prepared_report()
        ledger.finish()


@pytest.mark.parametrize(
    ("phase", "message"),
    [
        (
            "public_evaluating",
            "recovered an interrupted public batch without replaying public evidence",
        ),
        (
            "public_complete",
            "recovered after the durable public seal without opening private scoring",
        ),
        (
            "truth_evaluating",
            "recovered during private scoring without replaying private evidence",
        ),
    ],
)
def test_recovery_terminalizes_each_irreversible_two_phase_cut_without_replay(
    tmp_path: Path,
    phase: str,
    message: str,
) -> None:
    source = _fake_source()
    with _pinned_run(tmp_path) as pin:
        ledger, bindings = _new_development_ledger(pin)
        if phase == "public_evaluating":
            manifest = qualification._ManifestCapability(split="development", ledger=ledger)
            manifest.begin_batch((0, 1, 2, 3))
        else:
            manifest, _, _, _, _, seal = _seal_fake_public_split(ledger)
            if phase == "truth_evaluating":
                manifest.begin_private_scoring(public_seal=seal)
        assert ledger.record["splits"]["development"]["status"] == phase
        _forget_fake_process_state()
        qualification._recover_existing_attempt_for_tests(
            directory_pin=pin,
            stage="development",
            source_provenance=source,
        )
        report_bytes = qualification._pinned_read_bytes(
            pin,
            qualification._artifact_paths(pin)["development_report"],
            label=f"{phase} recovery report",
        )
        report = qualification._validate_report(
            qualification._strict_json_loads(report_bytes, label=f"{phase} recovery report"),
            stage="development",
            expected_source=source,
            expected_execution_mode="fake_test",
        )
        terminal = qualification._validate_terminal_ledger(
            qualification._pinned_read_bytes(
                pin,
                qualification._artifact_paths(pin)["development_ledger"],
                label=f"{phase} recovery ledger",
            ),
            stage="development",
            expected_bindings=bindings,
            expected_results=[],
            expected_outcome="error",
        )
        assert report["error"] == {"type": "InterruptedRun", "message": message}
        assert terminal["status"] == "terminal_error"
        assert terminal["splits"]["development"]["status"] == phase


@pytest.mark.parametrize(
    "phase",
    ["public_evaluating", "public_complete", "truth_evaluating"],
)
def test_terminal_error_recognizes_each_two_phase_interruption(
    tmp_path: Path,
    phase: str,
) -> None:
    source = _fake_source()
    with _pinned_run(tmp_path) as pin:
        ledger, bindings = _new_development_ledger(pin)
        evaluator = _FakeEvaluator(
            raise_on_call=1 if phase == "public_evaluating" else None,
            raise_on_private=phase == "truth_evaluating",
        )

        def guard(label: str) -> None:
            ledger._verify_disk()
            if phase == "public_complete" and label == "after development public seal":
                raise RuntimeError("injected post-public-seal interruption")

        with pytest.raises(RuntimeError) as captured:
            qualification._evaluate_split(
                split="development",
                ledger=ledger,
                evaluator=evaluator,
                boundary_guard=guard,
            )
        assert ledger.record["splits"]["development"]["status"] == phase
        qualification._persist_exception_report(
            directory_pin=pin,
            stage="development",
            source_provenance=source,
            ledger=ledger,
            error=captured.value,
            results=[],
        )
        terminal = qualification._validate_terminal_ledger(
            qualification._pinned_read_bytes(
                pin,
                qualification._artifact_paths(pin)["development_ledger"],
                label=f"{phase} terminal error ledger",
            ),
            stage="development",
            expected_bindings=bindings,
            expected_results=[],
            expected_outcome="error",
        )
        assert terminal["status"] == "terminal_error"
        assert terminal["splits"]["development"]["status"] == phase


def test_b4_capabilities_require_all_tokens_before_one_atomic_receipt(tmp_path: Path) -> None:
    with _pinned_run(tmp_path) as pin:
        ledger, bindings = _new_development_ledger(pin)
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        before = ledger.record
        with pytest.raises(PermissionError, match="exact four ordinals"):
            manifest.begin_batch((0, 1, 2, True))
        assert ledger.record == before
        assert qualification._BATCH_REGISTRY == {}
        assert qualification._TOKEN_REGISTRY == {}
        batch = manifest.begin_batch((0, 1, 2, 3))
        for ordinal, token in zip(batch.ordinals[:3], batch.tokens[:3], strict=True):
            manifest.consume_ordinal(batch, token, ordinal=ordinal)
        evaluation: dict[str, Any] = {}
        with pytest.raises(PermissionError, match="all four"):
            manifest.complete_batch(batch, evaluation)
        state = ledger.record["splits"]["development"]
        assert state["public_active_batch"] == [0, 1, 2, 3]
        assert state["public_batch_receipts"] == []
        assert state["public_next_ordinal"] == 0
        manifest.abort()
        _terminate_test_ledger(pin, ledger)
        terminal = qualification._pinned_read_bytes(
            pin,
            qualification._artifact_paths(pin)["development_ledger"],
            label="atomic terminal ledger",
        )
        parsed = qualification._validate_terminal_ledger(
            terminal,
            stage="development",
            expected_bindings=bindings,
        )
        assert parsed["status"] == "terminal_error"
        assert parsed["splits"]["development"]["public_active_batch"] == [0, 1, 2, 3]


def test_fake_batch_cannot_cross_formal_materialization_boundary(tmp_path: Path) -> None:
    with _pinned_run(tmp_path) as pin:
        ledger, _ = _new_development_ledger(pin)
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        batch = manifest.begin_batch((0, 1, 2, 3))
        request = qualification.BatchEvaluationRequest(
            split="development",
            ordinals=batch.ordinals,
            tokens=batch.tokens,
            _manifest=manifest,
            _batch=batch,
        )
        before = ledger.record
        with pytest.raises(PermissionError, match="governed formal ledger"):
            qualification._authorize_formal_batch(request)
        assert ledger.record == before
        assert qualification._BATCH_REGISTRY[id(batch)].consumed == set()
        assert all(
            qualification._TOKEN_REGISTRY[id(token)][2] == "issued" for token in batch.tokens
        )
        manifest.abort()
        _terminate_test_ledger(pin, ledger)


def test_capability_replay_cross_batch_and_cross_ordinal_fail(tmp_path: Path) -> None:
    with _pinned_run(tmp_path) as pin:
        ledger, _ = _new_development_ledger(pin)
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        batch = manifest.begin_batch((0, 1, 2, 3))
        with pytest.raises(PermissionError, match="forged"):
            manifest.consume_ordinal(batch, batch.tokens[0], ordinal=1)
        manifest.consume_ordinal(batch, batch.tokens[0], ordinal=0)
        with pytest.raises(PermissionError, match="live ledger|replayed"):
            manifest.consume_ordinal(batch, batch.tokens[0], ordinal=0)
        forged = qualification._OrdinalCapability("development", 1, object())
        with pytest.raises(PermissionError, match="forged"):
            manifest.consume_ordinal(batch, forged, ordinal=1)
        manifest.abort()
        _terminate_test_ledger(pin, ledger)


def test_formal_nonce_collision_preflight_is_all_or_none_before_any_touch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _pinned_run(tmp_path) as pin:
        ledger, _ = _new_development_ledger(pin)
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        first = manifest.begin_batch((0, 1, 2, 3))
        request = qualification.BatchEvaluationRequest(
            split="development",
            ordinals=first.ordinals,
            tokens=first.tokens,
            _manifest=manifest,
            _batch=first,
        )
        evaluation = _FakeEvaluator().evaluate_public_batch(request)
        manifest.complete_batch(first, evaluation)
        _, prior_rows = qualification._replay_validated_ledger_nonce_prefix(ledger.record)
        collision = prior_rows[0]["capability_nonce"]
        fresh = tuple(
            hashlib.sha256(f"fresh preflight {index}".encode()).hexdigest() for index in range(3)
        )
        entropy = iter((collision, *fresh))
        entropy_calls = 0

        def nonce_source() -> str:
            nonlocal entropy_calls
            entropy_calls += 1
            return next(entropy)

        monkeypatch.setattr(qualification, "_capability_nonce_hex", nonce_source)
        before = ledger.record
        batch_registry = dict(qualification._BATCH_REGISTRY)
        token_registry = dict(qualification._TOKEN_REGISTRY)
        authority_registry = dict(qualification._CONSUMED_ORDINAL_REGISTRY)
        scene_calls = 0

        def touch_scene() -> None:
            nonlocal scene_calls
            scene_calls += 1

        def preflight_then_scene() -> None:
            qualification._preflight_formal_batch_capability_nonces(
                ledger,
                split="development",
                ordinals=(4, 5, 6, 7),
            )
            touch_scene()

        with pytest.raises(PermissionError, match="collides with durable provenance"):
            preflight_then_scene()
        assert entropy_calls == qualification.BATCH_SIZE
        assert scene_calls == 0
        assert manifest._active is None
        assert ledger.record == before
        assert ledger.record["splits"]["development"]["public_active_batch"] is None
        assert batch_registry == qualification._BATCH_REGISTRY
        assert token_registry == qualification._TOKEN_REGISTRY
        assert authority_registry == qualification._CONSUMED_ORDINAL_REGISTRY
        _terminate_test_ledger(pin, ledger)


@pytest.mark.parametrize(
    "candidates",
    [
        ("1" * 64, "1" * 64, "2" * 64, "3" * 64),
        ("A" * 64, "1" * 64, "2" * 64, "3" * 64),
        ("g" * 64, "1" * 64, "2" * 64, "3" * 64),
    ],
    ids=("intra_b4_collision", "uppercase", "nonhex"),
)
def test_formal_nonce_preflight_rejects_nonexact_entropy_before_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    candidates: tuple[str, str, str, str],
) -> None:
    with _pinned_run(tmp_path) as pin:
        ledger, _ = _new_development_ledger(pin)
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        entropy = iter(candidates)
        monkeypatch.setattr(qualification, "_capability_nonce_hex", lambda: next(entropy))
        before = ledger.record
        with pytest.raises(RuntimeError, match="malformed entropy"):
            qualification._preflight_formal_batch_capability_nonces(
                ledger,
                split="development",
                ordinals=(0, 1, 2, 3),
            )
        assert manifest._active is None
        assert ledger.record == before
        assert ledger.record["splits"]["development"]["public_active_batch"] is None
        assert qualification._BATCH_REGISTRY == {}
        assert qualification._TOKEN_REGISTRY == {}
        assert qualification._CONSUMED_ORDINAL_REGISTRY == {}
        _terminate_test_ledger(pin, ledger)


def test_formal_nonce_prechecks_are_both_before_reservation_and_registry_mutation() -> None:
    source = Path(qualification.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    manifest = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "_ManifestCapability"
    )
    begin = next(
        node
        for node in manifest.body
        if isinstance(node, ast.FunctionDef) and node.name == "begin_batch"
    )
    body = ast.get_source_segment(source, begin)
    assert body is not None
    first = body.index("_preflight_formal_batch_capability_nonces(")
    second = body.index("_validated_formal_batch_capability_nonces(")
    reserve = body.index("self._ledger.reserve_batch(")
    batch_registry_write = body.index("_BATCH_REGISTRY[id(batch)] = registration")
    token_registry_write = body.index("_TOKEN_REGISTRY[id(token)] =")
    active_write = body.index("self._active = batch")
    assert first < second < reserve < batch_registry_write < token_registry_write < active_write


def test_formal_nonce_preflight_rejects_cross_split_and_reviewed_development_reuse(
    tmp_path: Path,
) -> None:
    source = _fake_source()
    with _pinned_run(tmp_path) as pin:
        assert (
            qualification._execute_development_for_tests(
                directory_pin=pin,
                evaluator=_FakeEvaluator(),
                source_provenance=source,
            )
            == 0
        )
        reviewed_hashes = _development_hashes(pin)
        reviewed, _, review_seal = qualification._review_development_bundle(
            directory_pin=pin,
            reviewed_checkpoint_sha256=reviewed_hashes["checkpoint_sha256"],
            reviewed_report_sha256=reviewed_hashes["report_sha256"],
            reviewed_development_ledger_sha256=reviewed_hashes["ledger_sha256"],
            expected_source=source,
            expected_execution_mode="fake_test",
        )
        bindings = qualification._ledger_bindings(
            stage="qualification",
            execution_mode="fake_test",
            directory_pin=pin,
            source_provenance=source,
            reviewed_development=reviewed,
        )
        expected_baseline, reviewed_rows = qualification._reviewed_development_nonce_baseline(
            pin,
            expected_ledger_sha256=reviewed["ledger_sha256"],
        )
        assert expected_baseline == {
            "schema": "rgbd_known_action_reviewed_development_nonce_baseline_v1",
            "ordered_rows_sha256": qualification._nonce_provenance_rows_sha256(reviewed_rows),
            "count": 64,
        }
        assert bindings["reviewed_development_nonce_baseline"] == expected_baseline
        malformed_baseline = dict(expected_baseline)
        malformed_baseline["count"] = True
        with pytest.raises(ValueError, match="schema/count"):
            qualification._validated_reviewed_development_nonce_baseline(malformed_baseline)

        path = qualification._artifact_paths(pin)["qualification_ledger"]
        authorization = qualification._authorize_ledger_creation(
            invocation_seal=qualification._mint_fake_runner_invocation_seal_for_tests(
                stage="qualification",
                directory_pin=pin,
                source_provenance=source,
                reviewed_development=reviewed,
            ),
            stage="qualification",
            directory_pin=pin,
            source_provenance=source,
            bindings=bindings,
            ledger_path=path,
            reviewed_development=reviewed,
        )
        ledger = qualification._AccessLedger(
            path,
            stage="qualification",
            bindings=bindings,
            directory_pin=pin,
            authorization=authorization,
        )

        def guard(_label: str) -> None:
            ledger._verify_disk()

        observed_result = qualification._evaluate_split(
            split=qualification.PROTECTED_SPLITS[0],
            ledger=ledger,
            evaluator=_FakeEvaluator(),
            boundary_guard=guard,
        )
        heldout = qualification._ManifestCapability(
            split=qualification.PROTECTED_SPLITS[1],
            ledger=ledger,
        )
        _, current_rows = qualification._replay_validated_ledger_nonce_prefix(ledger.record)
        observed_row = next(
            row for row in current_rows if row["split"] == qualification.PROTECTED_SPLITS[0]
        )
        fresh = tuple(
            hashlib.sha256(f"qualification fresh {index}".encode()).hexdigest()
            for index in range(3)
        )
        before = ledger.record
        for collision in (
            observed_row["capability_nonce"],
            reviewed_rows[0]["capability_nonce"],
        ):
            with pytest.raises(PermissionError, match="collides with durable provenance"):
                qualification._validated_formal_batch_capability_nonces(
                    (collision, *fresh),
                    ledger=ledger,
                    split=qualification.PROTECTED_SPLITS[1],
                    ordinals=(0, 1, 2, 3),
                )
            assert heldout._active is None
            assert ledger.record == before
            assert (
                ledger.record["splits"][qualification.PROTECTED_SPLITS[1]]["public_active_batch"]
                is None
            )

        report_digest = qualification._persist_exception_report(
            directory_pin=pin,
            stage="qualification",
            source_provenance=source,
            ledger=ledger,
            error=RuntimeError("nonce collision test cleanup"),
            results=[observed_result],
            reviewed_development=reviewed,
            reviewed_seal=review_seal,
        )
        assert type(report_digest) is str
        terminal = qualification._pinned_read_bytes(
            pin,
            path,
            label="nonce collision terminal qualification ledger",
        )
        parsed = qualification._validate_terminal_ledger(
            terminal,
            stage="qualification",
            expected_bindings=qualification._ledger_bindings(
                stage="qualification",
                execution_mode="fake_test",
                directory_pin=pin,
                source_provenance=source,
                reviewed_development=reviewed,
            ),
        )
        assert parsed["bindings"]["reviewed_development_nonce_baseline"] == expected_baseline


def test_runner_seal_authorization_and_ledger_are_each_one_shot(tmp_path: Path) -> None:
    with _pinned_run(tmp_path) as pin:
        source = _fake_source()
        bindings = qualification._ledger_bindings(
            stage="development",
            execution_mode="fake_test",
            directory_pin=pin,
            source_provenance=source,
        )
        path = qualification._artifact_paths(pin)["development_ledger"]
        seal = qualification._mint_fake_runner_invocation_seal_for_tests(
            stage="development",
            directory_pin=pin,
            source_provenance=source,
        )
        qualification._consume_runner_invocation_seal(
            seal,
            stage="development",
            directory_pin=pin,
            source_provenance=source,
        )
        with pytest.raises(PermissionError, match="replayed"):
            qualification._consume_runner_invocation_seal(
                seal,
                stage="development",
                directory_pin=pin,
                source_provenance=source,
            )
        authorization = qualification._mint_run_authorization(
            invocation_seal=seal,
            bindings=bindings,
            ledger_path=path,
        )
        ledger = qualification._AccessLedger(
            path,
            stage="development",
            bindings=bindings,
            directory_pin=pin,
            authorization=authorization,
        )
        with pytest.raises(PermissionError, match="replayed"):
            qualification._AccessLedger(
                path,
                stage="development",
                bindings=bindings,
                directory_pin=pin,
                authorization=authorization,
            )
        _terminate_test_ledger(pin, ledger)


def test_access_ledger_rejects_identical_byte_inode_replacement(tmp_path: Path) -> None:
    with _pinned_run(tmp_path) as pin:
        ledger, _ = _new_development_ledger(pin)
        path = qualification._artifact_paths(pin)["development_ledger"]
        original = qualification._pinned_read_bytes(pin, path, label="original ledger")
        replacement_name = ".replacement-ledger"
        descriptor = os.open(
            replacement_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=pin.directory_fd,
        )
        try:
            os.write(descriptor, original)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(
            replacement_name,
            path.name,
            src_dir_fd=pin.directory_fd,
            dst_dir_fd=pin.directory_fd,
        )
        os.fsync(pin.directory_fd)
        with pytest.raises(PermissionError, match="inode changed"):
            ledger.begin_split("development")


def test_pinned_directory_durable_writes_and_namespace_swap_detection(tmp_path: Path) -> None:
    path = (tmp_path / "run").resolve()
    pin = qualification._acquire_pinned_directory(path, create=True, canonical=False)
    artifact = path / qualification.DEVELOPMENT_REPORT_NAME
    qualification._pinned_durable_create(pin, artifact, b"first")
    qualification._pinned_durable_replace(pin, artifact, b"second")
    assert qualification._pinned_read_bytes(pin, artifact, label="test artifact") == b"second"

    moved = tmp_path / "moved"
    os.rename(path, moved)
    path.mkdir()
    with pytest.raises(PermissionError, match="namespace binding changed"):
        qualification._validate_pinned_directory(pin)


def test_pinned_directory_rejects_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "run"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises((PermissionError, TypeError)):
        qualification._acquire_pinned_directory(link, create=False, canonical=False)


def test_canonical_pin_requires_outer_authority_and_fake_helpers_reject_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = (tmp_path / "canonical-run").resolve()
    monkeypatch.setattr(qualification, "_canonical_run_directory", lambda: canonical)
    with pytest.raises(PermissionError, match="outer authority"):
        qualification._acquire_pinned_directory(canonical, create=True, canonical=True)
    assert not canonical.exists()
    with pytest.raises(PermissionError, match="identify each other"):
        qualification._acquire_pinned_directory(canonical, create=True, canonical=False)
    canonical.mkdir()
    audit_pin = qualification._acquire_pinned_directory(
        canonical,
        create=False,
        canonical=True,
        read_only=True,
    )
    try:
        assert audit_pin.mutable is False
        with pytest.raises(PermissionError, match="read-only"):
            qualification._pinned_durable_create(
                audit_pin,
                canonical / qualification.DEVELOPMENT_REPORT_NAME,
                b"forbidden",
            )
        with pytest.raises(PermissionError, match="formal directory pin"):
            qualification._clear_pinned_directory_registry_for_tests()
        assert qualification._PIN_REGISTRY.get(id(audit_pin)) is audit_pin
    finally:
        qualification._release_pinned_directory(audit_pin)

    fake_path = (tmp_path / "fake-run").resolve()
    pin = qualification._acquire_pinned_directory(fake_path, create=True, canonical=False)
    try:
        monkeypatch.setattr(qualification, "_canonical_run_directory", lambda: fake_path)
        with pytest.raises(PermissionError, match="fake execution"):
            qualification._mint_fake_runner_invocation_seal_for_tests(
                stage="development",
                directory_pin=pin,
                source_provenance=_fake_source(),
            )
        with pytest.raises(PermissionError, match="fake execution"):
            qualification._execute_development_for_tests(
                directory_pin=pin,
                evaluator=_FakeEvaluator(),
                source_provenance=_fake_source(),
            )
        assert qualification._pinned_inventory(pin) == frozenset()
    finally:
        qualification._release_pinned_directory(pin)


def test_fake_registry_cleanup_cannot_remove_formal_authority() -> None:
    authority = qualification._OuterRunnerAuthority(
        stage="development",
        receipt_sha256="a" * 64,
        receipt_nonce="d" * 64,
        authorization_record_sha256="e" * 64,
        authorization_nonce="f" * 64,
        runner_blob_sha256="1" * 64,
        bootstrap_literal_sha256="2" * 64,
        runner_preflight_sha256="3" * 64,
        bootstrap_environment_sha256="4" * 64,
        caller_environment_sha256="5" * 64,
        source_sha256="b" * 64,
        argv=("--phase", "development"),
        caller_code_sha256="c" * 64,
        owner_thread=0,
        nonce=object(),
    )
    qualification._OUTER_RUNNER_AUTHORITY_REGISTRY[id(authority)] = (authority, "issued")
    try:
        with pytest.raises(PermissionError, match="test cleanup"):
            qualification._clear_ephemeral_registries_for_tests()
        assert qualification._OUTER_RUNNER_AUTHORITY_REGISTRY[id(authority)] == (
            authority,
            "issued",
        )
    finally:
        qualification._OUTER_RUNNER_AUTHORITY_REGISTRY.pop(id(authority), None)


def test_outer_authorization_pipe_is_anonymous_bounded_closed_and_one_shot() -> None:
    _, record = _authorization_bundle()
    try:
        with _anonymous_pipe(record) as (fd_text, descriptor):
            parsed, digest, identity = qualification._consume_outer_authorization_record(fd_text)
            assert parsed == record
            assert digest == qualification.canonical_sha256(record)
            assert len(identity) == 4
            with pytest.raises(OSError):
                os.fstat(descriptor)
        with (
            _anonymous_pipe(record) as (replay_fd_text, _),
            pytest.raises(PermissionError, match="already consumed"),
        ):
            qualification._consume_outer_authorization_record(replay_fd_text)
        altered = json.loads(json.dumps(record))
        altered["argv"] = ["--phase", "development", "--altered"]
        altered_body = dict(altered)
        altered_body.pop("record_sha256")
        altered["record_sha256"] = qualification.canonical_sha256(altered_body)
        with (
            _anonymous_pipe(altered) as (nonce_replay_fd_text, _),
            pytest.raises(PermissionError, match="nonce was already consumed"),
        ):
            qualification._consume_outer_authorization_record(nonce_replay_fd_text)
    finally:
        _forget_authorization_record(record)


def test_outer_authorization_rejects_oversized_and_noncanonical_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, record = _authorization_bundle(nonce="0" * 64)
    canonical = qualification._canonical_json(record)
    monkeypatch.setattr(qualification, "_MAX_OUTER_AUTHORIZATION_BYTES", 32)
    with (
        _anonymous_pipe_bytes(canonical) as (fd_text, descriptor),
        pytest.raises(PermissionError, match="oversized"),
    ):
        qualification._consume_outer_authorization_record(fd_text)
    with pytest.raises(OSError):
        os.fstat(descriptor)
    monkeypatch.setattr(qualification, "_MAX_OUTER_AUTHORIZATION_BYTES", 16 * 1024)
    with (
        _anonymous_pipe_bytes(canonical + b"\n") as (fd_text, descriptor),
        pytest.raises(PermissionError, match="serialization is not canonical"),
    ):
        qualification._consume_outer_authorization_record(fd_text)
    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_published_bootstrap_literal_cannot_self_authenticate_from_mutable_global() -> None:
    namespace, runner_bytes, path = _exact_runner_namespace()
    contract = qualification._published_runner_contract(runner_bytes, path)
    custom_bootstrap = contract["bootstrap_source"] + "\n# mutable replacement"
    namespace["_BOOTSTRAP"] = custom_bootstrap
    environment = {
        "bootstrap_literal_sha256": qualification.sha256_bytes(custom_bootstrap.encode("utf-8"))
    }
    with pytest.raises(PermissionError, match="bootstrap literal differs"):
        qualification._validate_runner_security_globals(namespace, runner_bytes, path)
    with pytest.raises(PermissionError, match="binding differs"):
        qualification._validate_bootstrap_literal_binding(
            environment,
            contract["bootstrap_sha256"],
        )


def test_exact_runner_security_contract_accepts_complete_published_namespace() -> None:
    namespace, runner_bytes, path = _exact_runner_namespace()
    security = qualification._validate_runner_security_globals(namespace, runner_bytes, path)
    assert {
        "_git_paths",
        "_validated_oid",
        "_object_oid",
        "_stable_read",
        "_capture_outer_receipt",
        "_consume_outer_receipt",
        "_internal_main",
        "_outer_main",
        "main",
    } <= set(security["helper_structures"])
    assert security["main_dunders"]["loader"].endswith(".BuiltinImporter")
    assert security["contract"]["bootstrap_sha256"] == qualification.sha256_bytes(
        namespace["_BOOTSTRAP"].encode("utf-8")
    )


def test_internal_runner_rejects_combined_capture_loader_and_qualification_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace, _, _ = _exact_runner_namespace()
    touched: list[str] = []

    def forbidden_capture(*args: Any, **kwargs: Any) -> Any:
        touched.append("capture")
        pytest.fail("substituted capture ran before bootstrap preflight")

    class ForbiddenLoader:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            touched.append("loader")
            pytest.fail("substituted loader ran before bootstrap preflight")

    namespace["_capture_outer_receipt"] = forbidden_capture
    namespace["_ExactCommitLoader"] = ForbiddenLoader
    namespace["qualification"] = qualification.types.ModuleType("injected_qualification")
    monkeypatch.setattr(
        qualification,
        "_canonical_run_directory",
        lambda: pytest.fail("substituted internal runner reached canonical state"),
    )
    with pytest.raises(PermissionError, match="exact bootstrap capability"):
        namespace["_internal_main"](
            ["--phase", "development"],
            {},
            "9",
            "00",
            lambda: (),
        )
    assert touched == []


def test_outer_runner_rejects_nonisolated_invocation_before_initial_capture() -> None:
    namespace, _, _ = _exact_runner_namespace()
    touched: list[str] = []

    def forbidden_capture(*args: Any, **kwargs: Any) -> Any:
        touched.append("capture")
        pytest.fail("outer capture ran before isolated direct-script preflight")

    namespace["_capture_outer_receipt"] = forbidden_capture
    with pytest.raises(PermissionError, match="exact isolated direct script"):
        namespace["_outer_main"]([])
    assert touched == []


@pytest.mark.parametrize(
    ("owner_name", "attribute"),
    [
        ("subprocess", "run"),
        ("os", "open"),
        ("json", "loads"),
        ("importlib.util", "spec_from_loader"),
    ],
)
def test_runner_security_rejects_mutated_critical_module_attribute(
    owner_name: str,
    attribute: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace, runner_bytes, path = _exact_runner_namespace()
    parts = owner_name.split(".")
    owner = namespace[parts[0]]
    for part in parts[1:]:
        owner = getattr(owner, part)
    monkeypatch.setattr(owner, attribute, lambda *args, **kwargs: None)
    with pytest.raises(PermissionError, match="critical callable"):
        qualification._validate_runner_security_globals(namespace, runner_bytes, path)


@pytest.mark.parametrize(
    ("helper_name", "default_name", "replacement", "message"),
    [
        ("_git_bytes", "_run", lambda *args, **kwargs: None, "captured default"),
        ("_outer_main", "_capture", lambda *args, **kwargs: None, "captured default"),
        ("main", "_consume", lambda *args, **kwargs: None, "captured default"),
        ("_internal_main", "_hash", lambda contents: "0" * 64, "captured default"),
        (
            "_qualification_callables",
            "_getattr",
            lambda *args, **kwargs: None,
            "captured default",
        ),
        ("_stable_read", "_read_flags", 0, "captured value default"),
    ],
)
def test_runner_security_rejects_captured_default_substitution(
    helper_name: str,
    default_name: str,
    replacement: Any,
    message: str,
) -> None:
    namespace, runner_bytes, path = _exact_runner_namespace()
    namespace[helper_name].__kwdefaults__[default_name] = replacement
    with pytest.raises(PermissionError, match=message):
        qualification._validate_runner_security_globals(namespace, runner_bytes, path)


@pytest.mark.parametrize(
    "default_name",
    [
        "_parse",
        "_spec_from_loader",
        "_module_from_spec",
        "_sys",
        "_walk",
        "_dump",
        "_import_node",
        "_import_from_node",
        "_dict_node",
        "_constant_node",
        "_sha256",
        "_modules",
        "_resolve_method",
        "_exec_module_method",
    ],
)
def test_runner_security_rejects_every_lightweight_loader_default_substitution(
    default_name: str,
) -> None:
    namespace, runner_bytes, path = _exact_runner_namespace()
    method = namespace["_ExactCommitLoader"].__dict__["load_lightweight_qualification"]
    method.__kwdefaults__[default_name] = object()
    with pytest.raises(PermissionError, match="loader captured default"):
        qualification._validate_runner_security_globals(namespace, runner_bytes, path)


@pytest.mark.parametrize(
    ("name", "replacement"),
    [
        ("_TRUSTED_GIT", "/tmp/substituted-git"),
        ("REPOSITORY_ROOT", Path("/tmp/substituted-repository")),
        (
            "PUBLICATION_SURFACE_PATHS",
            {"runner": "scripts/substituted.py"},
        ),
    ],
)
def test_runner_security_rejects_mutated_outer_capture_constant(
    name: str,
    replacement: Any,
) -> None:
    namespace, runner_bytes, path = _exact_runner_namespace()
    namespace[name] = replacement
    with pytest.raises((PermissionError, ValueError), match="runner constant"):
        qualification._validate_runner_security_globals(namespace, runner_bytes, path)


def test_runner_security_rejects_mutated_captured_open_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace, runner_bytes, path = _exact_runner_namespace()
    monkeypatch.setattr(qualification.os, "O_RDONLY", qualification.os.O_RDONLY ^ 1)
    with pytest.raises(PermissionError, match="captured value default"):
        qualification._validate_runner_security_globals(namespace, runner_bytes, path)


def test_runner_security_rejects_mutated_sysconfig_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace, runner_bytes, path = _exact_runner_namespace()
    monkeypatch.setattr(qualification.sysconfig, "get_paths", lambda: {})
    with pytest.raises(PermissionError, match="critical callable"):
        qualification._validate_runner_security_globals(namespace, runner_bytes, path)


@pytest.mark.parametrize(
    "name",
    ["_git_paths", "_validated_oid", "_object_oid", "_stable_read"],
)
def test_runner_security_closure_rejects_omitted_transitive_helper(name: str) -> None:
    namespace, runner_bytes, path = _exact_runner_namespace()
    namespace.pop(name)
    with pytest.raises(PermissionError, match="security helper"):
        qualification._validate_runner_security_globals(namespace, runner_bytes, path)


def test_runner_security_closure_rejects_helper_substitution() -> None:
    namespace, runner_bytes, path = _exact_runner_namespace()
    namespace["_git_paths"] = lambda: (path / ".git", path)
    with pytest.raises(PermissionError, match="security helper"):
        qualification._validate_runner_security_globals(namespace, runner_bytes, path)


@pytest.mark.parametrize("name", ["importlib", "stat", "json", "hashlib", "secrets"])
def test_runner_security_closure_rejects_omitted_module_global(name: str) -> None:
    namespace, runner_bytes, path = _exact_runner_namespace()
    namespace.pop(name)
    with pytest.raises(PermissionError, match="stdlib module global"):
        qualification._validate_runner_security_globals(namespace, runner_bytes, path)


@pytest.mark.parametrize(
    "name",
    ["REPOSITORY_ROOT", "_RECEIPT_SCHEMA", "_MAX_SOURCE_BYTES", "_TRUSTED_GIT"],
)
def test_runner_security_closure_rejects_omitted_path_schema_limit_or_git_constant(
    name: str,
) -> None:
    namespace, runner_bytes, path = _exact_runner_namespace()
    namespace.pop(name)
    with pytest.raises(PermissionError, match="runner constant"):
        qualification._validate_runner_security_globals(namespace, runner_bytes, path)


@pytest.mark.parametrize("dunder", ["__loader__", "__spec__", "__cached__", "__package__"])
def test_runner_security_closure_rejects_main_dunder_mutation(dunder: str) -> None:
    namespace, runner_bytes, path = _exact_runner_namespace()
    namespace[dunder] = object()
    with pytest.raises(PermissionError, match="module dunders"):
        qualification._validate_runner_security_globals(namespace, runner_bytes, path)


@pytest.mark.parametrize("mutation", ["commit", "cache"])
def test_exact_loader_rejects_commit_or_cache_mutation(mutation: str) -> None:
    namespace, _, _ = _exact_runner_namespace()
    loader_type = namespace["_ExactCommitLoader"]
    loader = object.__new__(loader_type)
    loader._commit = "a" * 40
    loader._cache = {}
    if mutation == "commit":
        loader._commit = "b" * 40
    else:
        loader._cache = []
    with pytest.raises(PermissionError, match="commit/cache root"):
        qualification._validate_exact_loader_root(loader, loader_type, "a" * 40)


@pytest.mark.parametrize("mutation", ["spec", "dunder"])
def test_exact_loader_rejects_module_spec_or_dunder_mutation(mutation: str) -> None:
    namespace, _, _ = _exact_runner_namespace()
    loader_type = namespace["_ExactCommitLoader"]
    loader = object.__new__(loader_type)
    loader._commit = "a" * 40
    loader._cache = {}
    fullname = "world_model.training.rgbd_known_action_qualification"
    relative = "world_model/training/rgbd_known_action_qualification.py"
    origin = f"git:{loader._commit}:{relative}"
    spec = qualification.importlib.util.spec_from_loader(
        fullname,
        loader,
        origin=origin,
        is_package=False,
    )
    assert spec is not None
    module = qualification.types.ModuleType(fullname)
    module.__package__ = "world_model.training"
    module.__file__ = os.fspath(qualification.REPOSITORY_ROOT / relative)
    module.__cached__ = None
    module.__loader__ = loader
    module.__spec__ = spec
    if mutation == "spec":
        spec.origin = "git:wrong:origin"
    else:
        module.__loader__ = object()
    with pytest.raises(PermissionError, match="dunder/spec binding"):
        qualification._validate_exact_loaded_module(
            module,
            fullname=fullname,
            relative=relative,
            is_package=False,
            loader=loader,
            commit="a" * 40,
        )


def test_exact_loader_loads_lightweight_leaf_without_heavy_parent_imports() -> None:
    _, loader, module, fake_sys = _lightweight_qualification_module()
    assert loader._cache[qualification.__name__][1:] == (
        "world_model/training/rgbd_known_action_qualification.py",
        False,
    )
    assert set(fake_sys.modules) == {qualification.__name__}
    assert module.torch is None
    assert module.OrpheusConfig is None
    assert module.load_config is None
    assert module._RUNTIME_DEPENDENCIES_ACTIVE is False
    assert module.__version__ == "0.1.0"
    assert module.SPECIFICATION_VERSION == "1.60"
    assert module.SIMULATOR_VERSION == "sphere_world_v7"


@pytest.mark.parametrize(
    "forbidden",
    [
        "import torch\n",
        "import world_model\n",
        "if True:\n    import torch\n",
    ],
)
def test_exact_loader_rejects_heavy_top_level_qualification_import(
    forbidden: str,
) -> None:
    path = qualification.REPOSITORY_ROOT / "world_model/training/rgbd_known_action_qualification.py"
    source = path.read_text(encoding="utf-8")
    insertion = source.index("import argparse")
    altered = (source[:insertion] + forbidden + source[insertion:]).encode("utf-8")
    with pytest.raises(PermissionError, match="recursively imports"):
        _lightweight_qualification_module(source=altered)


@pytest.mark.parametrize(
    "injected",
    [
        'exec("__import__(\\"torch\\")")\n',
        "_certificate_assignment = object()\n",
        "def _certificate_default(value=object()):\n    return value\n",
        "@staticmethod\ndef _certificate_decorated():\n    return None\n",
        "class _CertificateClass:\n    marker = object()\n",
    ],
)
def test_exact_loader_rejects_changed_module_execution_surface(injected: str) -> None:
    path = qualification.REPOSITORY_ROOT / "world_model/training/rgbd_known_action_qualification.py"
    source = path.read_text(encoding="utf-8")
    insertion = source.index("import argparse")
    altered = (source[:insertion] + injected + source[insertion:]).encode("utf-8")
    with pytest.raises(PermissionError, match="module-execution surface fingerprint"):
        _lightweight_qualification_module(source=altered)


def test_exact_loader_rejects_dynamic_import_inside_import_time_called_helper() -> None:
    path = qualification.REPOSITORY_ROOT / "world_model/training/rgbd_known_action_qualification.py"
    source = path.read_text(encoding="utf-8")
    original = "def canonical_sha256(value: Any) -> str:\n    return "
    altered = source.replace(
        original,
        'def canonical_sha256(value: Any) -> str:\n    __import__("torch")\n    return ',
        1,
    ).encode("utf-8")
    with pytest.raises(PermissionError, match="module-execution surface fingerprint"):
        _lightweight_qualification_module(source=altered)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "fingerprint sentinel binding differs"),
        ("duplicate", "fingerprint sentinel binding differs"),
        ("malformed", "fingerprint sentinel is malformed"),
    ],
)
def test_exact_loader_rejects_missing_duplicate_or_malformed_ast_sentinel(
    mutation: str,
    message: str,
) -> None:
    path = qualification.REPOSITORY_ROOT / "world_model/training/rgbd_known_action_qualification.py"
    source = path.read_text(encoding="utf-8")
    key = '"_LIGHTWEIGHT_QUALIFICATION_EXECUTION_SHA256"'
    sentinel_line = next(line for line in source.splitlines(keepends=True) if key in line)
    if mutation == "missing":
        replacement = ""
    elif mutation == "duplicate":
        replacement = sentinel_line + sentinel_line
    else:
        start = sentinel_line.index('": "') + len('": "')
        replacement = sentinel_line[:start] + "g" * 64 + sentinel_line[start + 64 :]
    altered = source.replace(sentinel_line, replacement, 1).encode("utf-8")
    with pytest.raises(PermissionError, match=message):
        _lightweight_qualification_module(source=altered)


@pytest.mark.parametrize("failure_phase", ["parse", "execute"])
def test_exact_loader_rolls_back_all_new_heavy_modules_on_failure(
    failure_phase: str,
) -> None:
    namespace, _, _ = _exact_runner_namespace()
    loader_type = namespace["_ExactCommitLoader"]
    loader = object.__new__(loader_type)
    loader._commit = "a" * 40
    relative = "world_model/training/rgbd_known_action_qualification.py"
    source = (qualification.REPOSITORY_ROOT / relative).read_bytes()
    loader._cache = {qualification.__name__: (source, relative, False)}
    fake_modules: dict[str, Any] = {}
    fake_sys = qualification.types.SimpleNamespace(modules=fake_modules)

    def poison() -> None:
        fake_modules["torch.rollback"] = object()
        fake_modules["world_model.rollback"] = object()

    def poisoned_parse(*args: Any, **kwargs: Any) -> Any:
        poison()
        raise RuntimeError("injected parser failure")

    def poisoned_exec(*args: Any, **kwargs: Any) -> None:
        poison()
        raise RuntimeError("injected execution failure")

    call_kwargs: dict[str, Any] = {
        "_sys": fake_sys,
        "_modules": fake_modules,
    }
    if failure_phase == "parse":
        call_kwargs["_parse"] = poisoned_parse
    else:
        call_kwargs["_exec_module_method"] = poisoned_exec
    with pytest.raises(RuntimeError, match="injected"):
        loader.load_lightweight_qualification(
            qualification.__name__,
            **call_kwargs,
        )
    assert fake_modules == {}


def test_checkpoint_activation_rejects_imported_project_parent_before_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, module, _ = _lightweight_qualification_module()
    attempted: list[str] = []

    def forbidden_import(name: str) -> Any:
        attempted.append(name)
        pytest.fail("contaminated checkpoint recovery reached Torch import")

    monkeypatch.setitem(
        module._activate_recovery_checkpoint_dependency.__kwdefaults__,
        "_import_module",
        forbidden_import,
    )
    with pytest.raises(PermissionError, match="project package isolation"):
        module._activate_recovery_checkpoint_dependency()
    assert attempted == []
    assert module.torch is None


def _lightweight_qualification_module(
    *,
    source: bytes | None = None,
) -> tuple[Any, Any, Any, Any]:
    namespace, _, _ = _exact_runner_namespace()
    loader_type = namespace["_ExactCommitLoader"]
    loader = object.__new__(loader_type)
    loader._commit = "a" * 40
    relative = "world_model/training/rgbd_known_action_qualification.py"
    exact_source = (
        (qualification.REPOSITORY_ROOT / relative).read_bytes() if source is None else source
    )
    loader._cache = {qualification.__name__: (exact_source, relative, False)}
    fake_sys = qualification.types.SimpleNamespace(modules={})
    previous = qualification.sys.modules[qualification.__name__]

    def module_from_spec(spec: Any) -> Any:
        module = qualification.importlib.util.module_from_spec(spec)
        qualification.sys.modules[qualification.__name__] = module
        return module

    try:
        module = loader.load_lightweight_qualification(
            qualification.__name__,
            _module_from_spec=module_from_spec,
            _sys=fake_sys,
            _modules=fake_sys.modules,
        )
    finally:
        qualification.sys.modules[qualification.__name__] = previous
    return namespace, loader, module, fake_sys


def _bind_qualification_to_exact_runner_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, Any, Any, Any, Any]:
    del monkeypatch
    namespace, loader, module, fake_sys = _lightweight_qualification_module()
    spec = module.__spec__
    assert spec is not None
    fake_modules = {
        name: module.__dict__[name]
        for name in (
            "argparse",
            "ast",
            "builtins",
            "contextlib",
            "copy",
            "hashlib",
            "importlib",
            "io",
            "json",
            "math",
            "os",
            "pathlib",
            "resource",
            "secrets",
            "stat",
            "subprocess",
            "sys",
            "sysconfig",
            "tempfile",
            "threading",
            "time",
            "types",
        )
    }
    fake_modules[qualification.__name__] = module
    fake_sys.modules = fake_modules
    return namespace["_qualification_callables"], loader, spec, module, fake_sys


def test_qualification_preflight_accepts_exact_transitive_helper_closure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper, loader, _, module, fake_sys = _bind_qualification_to_exact_runner_loader(monkeypatch)
    callables = helper(module, loader, _sys=fake_sys)
    assert len(callables) == 10
    assert callables[0] is module.capture_published_source


def test_qualification_preflight_rejects_durable_replace_default_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper, loader, _, module, fake_sys = _bind_qualification_to_exact_runner_loader(monkeypatch)
    defaults = module._pinned_durable_replace.__kwdefaults__
    assert type(defaults) is dict and defaults["_replace"] is module.os.replace
    defaults["_replace"] = lambda *args, **kwargs: None
    with pytest.raises(PermissionError, match="qualification captured default"):
        helper(module, loader, _sys=fake_sys)


def test_authenticated_durable_replace_default_survives_module_attribute_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper, loader, _, module, fake_sys = _bind_qualification_to_exact_runner_loader(monkeypatch)
    defaults = module._pinned_durable_replace.__kwdefaults__
    assert type(defaults) is dict
    captured_replace = defaults["_replace"]
    assert captured_replace is module.os.replace
    assert len(helper(module, loader, _sys=fake_sys)) == 10

    pin = module._acquire_pinned_directory(
        (tmp_path / "captured-recovery-replace").resolve(),
        create=True,
        canonical=False,
    )
    path = pin.path / module.DEVELOPMENT_LEDGER_NAME
    replacement_attempts: list[tuple[Any, ...]] = []

    def substituted_replace(*args: Any, **kwargs: Any) -> None:
        replacement_attempts.append((*args, kwargs))
        pytest.fail("recovery replacement used the mutable module attribute")

    try:
        module._pinned_durable_create(pin, path, b"captured predecessor")
        predecessor_inventory = module._pinned_recovery_inventory_binding(pin)
        predecessor_identity = module._pinned_full_file_identity(pin, path)
        with monkeypatch.context() as patch:
            patch.setattr(module.os, "replace", substituted_replace)
            assert defaults["_replace"] is captured_replace
            assert module.os.replace is substituted_replace
            reduced, full = module._pinned_durable_replace(
                pin,
                path,
                b"captured replacement",
                _return_full_identity=True,
                _recovery_expected_target_identity=predecessor_identity,
                _recovery_expected_inventory=predecessor_inventory,
            )
        assert module.os.replace is captured_replace
        assert replacement_attempts == []
        assert path.read_bytes() == b"captured replacement"
        assert full == module._pinned_full_file_identity(pin, path)
        assert reduced == (full[0], full[1], full[3], full[4], full[6])
    finally:
        module._release_pinned_directory(pin)


@pytest.mark.parametrize("mutation", ["cached", "search_locations", "helper", "module", "constant"])
def test_qualification_preflight_rejects_spec_helper_module_or_constant_substitution(
    mutation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper, loader, spec, module, fake_sys = _bind_qualification_to_exact_runner_loader(monkeypatch)
    if mutation == "cached":
        spec.cached = "/tmp/substituted.pyc"
    elif mutation == "search_locations":
        spec.submodule_search_locations = []
    elif mutation == "helper":
        monkeypatch.setattr(module, "_git_bytes", lambda *args, **kwargs: b"")
    elif mutation == "module":
        monkeypatch.setattr(
            module,
            "subprocess",
            qualification.types.ModuleType("substituted_subprocess"),
        )
    else:
        monkeypatch.setattr(module, "_TRUSTED_GIT", "/tmp/substituted-git")
    with pytest.raises(PermissionError, match="qualification"):
        helper(module, loader, _sys=fake_sys)


def test_outer_authorization_rejects_regular_named_and_duplicate_descriptors(
    tmp_path: Path,
) -> None:
    _, record = _authorization_bundle(nonce="7" * 64)
    regular = tmp_path / "authorization.json"
    regular.write_bytes(qualification._canonical_json(record))
    descriptor = os.open(regular, os.O_RDONLY)
    with pytest.raises(PermissionError, match="anonymous pipe"):
        qualification._consume_outer_authorization_record(str(descriptor))
    with pytest.raises(OSError):
        os.fstat(descriptor)

    fifo = tmp_path / "named-authorization.fifo"
    os.mkfifo(fifo)
    descriptor = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
    with pytest.raises(PermissionError, match="anonymous pipe"):
        qualification._consume_outer_authorization_record(str(descriptor))
    with pytest.raises(OSError):
        os.fstat(descriptor)

    contents = qualification._canonical_json(record)
    read_fd, write_fd = os.pipe()
    duplicate_fd = os.dup(read_fd)
    try:
        os.write(write_fd, contents)
    finally:
        os.close(write_fd)
    try:
        qualification._consume_outer_authorization_record(str(read_fd))
        with pytest.raises((ValueError, json.JSONDecodeError)):
            qualification._consume_outer_authorization_record(str(duplicate_fd))
    finally:
        with suppress(OSError):
            os.close(read_fd)
        with suppress(OSError):
            os.close(duplicate_fd)
        _forget_authorization_record(record)


def test_direct_registrar_consumes_pipe_before_rejecting_nonisolated_module_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _fake_source()
    argv = ["--phase", "development"]
    receipt, record = _authorization_bundle(argv=argv, source=source, nonce="6" * 64)
    preloader, runtime = qualification._expected_isolated_sys_paths()
    flags = {
        "isolated": qualification.sys.flags.isolated,
        "no_site": qualification.sys.flags.no_site,
        "ignore_environment": qualification.sys.flags.ignore_environment,
        "no_user_site": qualification.sys.flags.no_user_site,
        "safe_path": getattr(qualification.sys.flags, "safe_path", None),
    }
    environment = {
        "schema": "rgbd_known_action_bootstrap_environment_v1",
        "python_version": qualification.sys.version,
        "python_hexversion": qualification.sys.hexversion,
        "implementation": qualification.sys.implementation.name,
        "executable": qualification.sys.executable,
        "flags": flags,
        "preloader_sys_path": list(preloader),
        "preloader_sys_path_sha256": qualification.canonical_sha256(list(preloader)),
        "runtime_sys_path": list(runtime),
        "runtime_sys_path_sha256": qualification.canonical_sha256(list(runtime)),
        "runner_blob_sha256": source["publication_surface_blobs"]["runner"]["blob_sha256"],
        "bootstrap_literal_sha256": "7" * 64,
        "runner_preflight_sha256": "8" * 64,
        "preloaded_world_model": [],
    }
    canonical_accessed = False

    def forbidden_canonical_access() -> Path:
        nonlocal canonical_accessed
        canonical_accessed = True
        raise AssertionError("registrar crossed the canonical run boundary")

    monkeypatch.setattr(qualification, "_canonical_run_directory", forbidden_canonical_access)
    try:
        with _anonymous_pipe(record) as (fd_text, descriptor):
            with pytest.raises(PermissionError, match="isolation flags"):
                qualification._register_outer_runner_authority(
                    authorization_fd_text=fd_text,
                    bootstrap_environment_hex=qualification._canonical_json(environment).hex(),
                    receipt=receipt,
                    argv=argv,
                    source_provenance=source,
                )
            with pytest.raises(OSError):
                os.fstat(descriptor)
        assert canonical_accessed is False
    finally:
        _forget_authorization_record(record)


@pytest.mark.parametrize("mutation", ["argv", "source"])
def test_outer_authorization_rejects_mutated_argv_or_source_before_canonical_access(
    mutation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _fake_source()
    argv = ["--phase", "development"]
    receipt, record = _authorization_bundle(
        argv=argv,
        source=source,
        nonce=("4" if mutation == "argv" else "5") * 64,
    )
    supplied_argv = list(argv)
    supplied_source = source
    if mutation == "argv":
        supplied_argv = ["--phase", "qualification"]
    else:
        supplied_source = json.loads(json.dumps(source))
        supplied_source["commit"] = "c" * 40
        supplied_source["upstream_commit"] = "c" * 40
    monkeypatch.setattr(
        qualification,
        "_canonical_run_directory",
        lambda: pytest.fail("mutated authority accessed the canonical run directory"),
    )
    try:
        with (
            _anonymous_pipe(record) as (fd_text, _),
            pytest.raises((PermissionError, ValueError)),
        ):
            qualification._register_outer_runner_authority(
                authorization_fd_text=fd_text,
                bootstrap_environment_hex="00",
                receipt=receipt,
                argv=supplied_argv,
                source_provenance=supplied_source,
            )
    finally:
        _forget_authorization_record(record)


@pytest.mark.parametrize("mutation", ["parent", "runner_blob"])
def test_outer_authorization_rejects_parent_or_runner_blob_mutation_before_canonical_access(
    mutation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _fake_source()
    argv = ["--phase", "development"]
    receipt, record = _authorization_bundle(
        argv=argv,
        source=source,
        nonce=("2" if mutation == "parent" else "3") * 64,
    )
    if mutation == "parent":
        record["expected_child_parent_pid"] += 1
    else:
        record["runner_blob_sha256"] = "0" * 64
    body = dict(record)
    body.pop("record_sha256")
    record["record_sha256"] = qualification.canonical_sha256(body)
    receipt["authorization_record_sha256"] = qualification.canonical_sha256(record)
    monkeypatch.setattr(
        qualification,
        "_canonical_run_directory",
        lambda: pytest.fail("mutated authority accessed the canonical run directory"),
    )
    try:
        with (
            _anonymous_pipe(record) as (fd_text, _),
            pytest.raises(PermissionError, match="cross-binding"),
        ):
            qualification._register_outer_runner_authority(
                authorization_fd_text=fd_text,
                bootstrap_environment_hex="00",
                receipt=receipt,
                argv=argv,
                source_provenance=source,
            )
    finally:
        _forget_authorization_record(record)


def test_bootstrap_environment_literal_mutation_rejects_before_canonical_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, runner_bytes, path = _exact_runner_namespace()
    contract = qualification._published_runner_contract(runner_bytes, path)
    environment = {"bootstrap_literal_sha256": "0" * 64}
    monkeypatch.setattr(
        qualification,
        "_canonical_run_directory",
        lambda: pytest.fail("mutated bootstrap accessed the canonical run directory"),
    )
    with pytest.raises(PermissionError, match="binding differs"):
        qualification._validate_bootstrap_literal_binding(
            environment,
            contract["bootstrap_sha256"],
        )


def test_runner_authority_source_has_no_mutable_receipt_shortcut_and_binds_exact_loader() -> None:
    runner_path = qualification.REPOSITORY_ROOT / qualification.PUBLICATION_SURFACE_PATHS["runner"]
    runner_source = runner_path.read_text(encoding="utf-8")
    runner_tree = ast.parse(runner_source)
    assert "_CONSUMED_OUTER_RECEIPT" not in runner_source
    assert "def _rgbd_known_action_bootstrap():" in runner_source
    assert 'del _bootstrap_globals["_rgbd_known_action_bootstrap"]' in runner_source
    assert "_y.argv = [_path, *_user_argv]" in runner_source
    assert "_authorization_fd_text=_authorization_fd_text" in runner_source
    assert "_bootstrap_security=_main_capability" in runner_source
    assert "not cryptographic\nattestation against a malicious same-user launcher" in runner_source
    internal = next(
        node
        for node in runner_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_internal_main"
    )
    assert [argument.arg for argument in internal.args.args][-3:] == [
        "authorization_fd_text",
        "bootstrap_environment_hex",
        "bootstrap_security",
    ]
    outer = next(
        node
        for node in runner_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_outer_main"
    )
    assert isinstance(outer.body[0], ast.Expr)
    assert isinstance(outer.body[0].value, ast.Call)
    assert isinstance(outer.body[0].value.func, ast.Name)
    assert outer.body[0].value.func.id == "_preflight"
    register_source = Path(qualification.__file__).read_text(encoding="utf-8")
    register_tree = ast.parse(register_source)
    register = next(
        node
        for node in register_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_register_outer_runner_authority"
    )
    first_statement = (
        register.body[1] if isinstance(register.body[0], ast.Expr) else register.body[0]
    )
    assert isinstance(first_statement, ast.Assign)
    assert isinstance(first_statement.value, ast.Call)
    assert isinstance(first_statement.value.func, ast.Name)
    assert first_statement.value.func.id == "_consume_outer_authorization_record"
    for required in (
        "frame.f_globals is not trusted_namespace",
        "sys.meta_path[0] is not loader",
        "tuple(sys.meta_path[1:]) != meta_path_snapshot",
        "_validate_exact_loader_cache(",
        "_published_runner_contract(runner_bytes, expected_path)",
        "capture_published_source(REPOSITORY_ROOT)",
    ):
        assert required in register_source


def test_durable_create_replace_reconcile_and_temp_fstat_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _pinned_run(tmp_path) as pin:
        artifact = pin.path / qualification.DEVELOPMENT_REPORT_NAME
        original_fsync = qualification.os.fsync
        directory_failures = 0

        def fail_first_directory_fsync(descriptor: int) -> None:
            nonlocal directory_failures
            if descriptor == pin.directory_fd and directory_failures == 0:
                directory_failures += 1
                raise OSError("injected post-publication directory fsync fault")
            original_fsync(descriptor)

        monkeypatch.setattr(qualification.os, "fsync", fail_first_directory_fsync)
        qualification._pinned_durable_create(pin, artifact, b"first")
        assert (
            qualification._pinned_read_bytes(pin, artifact, label="reconciled create") == b"first"
        )

        directory_failures = 0
        qualification._pinned_durable_replace(pin, artifact, b"second")
        assert (
            qualification._pinned_read_bytes(pin, artifact, label="reconciled replace") == b"second"
        )

        monkeypatch.setattr(qualification.os, "fsync", original_fsync)
        original_fstat = qualification.os.fstat
        injected = False

        def fail_first_empty_regular_fstat(descriptor: int) -> os.stat_result:
            nonlocal injected
            metadata = original_fstat(descriptor)
            if (
                not injected
                and descriptor not in {pin.parent_fd, pin.directory_fd}
                and qualification.stat.S_ISREG(metadata.st_mode)
                and metadata.st_size == 0
            ):
                injected = True
                raise OSError("injected temporary fstat fault")
            return metadata

        monkeypatch.setattr(qualification.os, "fstat", fail_first_empty_regular_fstat)
        with pytest.raises(OSError, match="temporary fstat"):
            qualification._pinned_durable_replace(pin, artifact, b"third")
        assert qualification._pinned_inventory(pin) == frozenset({artifact.name})
        assert (
            qualification._pinned_read_bytes(pin, artifact, label="unchanged replacement")
            == b"second"
        )


def test_fake_development_checkpoint_and_report_roundtrip_use_weights_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_weights_only: list[bool] = []
    original_load = qualification.torch.load

    def guarded_load(*args: Any, **kwargs: Any) -> Any:
        observed_weights_only.append(kwargs.get("weights_only"))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(qualification.torch, "load", guarded_load)
    source = _fake_source()
    with _pinned_run(tmp_path) as pin:
        evaluator = _FakeEvaluator()
        assert (
            qualification._execute_development_for_tests(
                directory_pin=pin,
                evaluator=evaluator,
                source_provenance=source,
            )
            == 0
        )
        assert len(evaluator.calls) == 16
        assert [ordinal for _, batch in evaluator.calls for ordinal in batch] == list(range(64))
        paths = qualification._artifact_paths(pin)
        report_bytes = qualification._pinned_read_bytes(
            pin,
            paths["development_report"],
            label="development report",
        )
        report = qualification._strict_json_loads(report_bytes, label="development report")
        with pytest.raises(ValueError, match="execution_mode"):
            qualification._validate_report(
                report,
                stage="development",
                expected_source=source,
            )
        validated = qualification._validate_report(
            report,
            stage="development",
            expected_source=source,
            expected_execution_mode="fake_test",
        )
        assert validated["passed"] is True
        assert validated["execution_mode"] == "fake_test"
        assert "formal_authorization" not in validated
        assert validated["checkpoint"]["model_state_entry_count"] == 0
        checkpoint = qualification._pinned_read_bytes(
            pin,
            paths["checkpoint"],
            label="development checkpoint",
            maximum=qualification.MAX_CHECKPOINT_BYTES,
        )
        payload = qualification._validate_checkpoint_payload(
            checkpoint,
            expected_source=source,
            expected_execution_mode="fake_test",
        )
        assert payload["model_state"] == {}
        assert payload["execution_mode"] == "fake_test"
        assert "formal_authorization" not in payload
        assert payload["optimizer_state"] is None
        assert payload["rng_state"] is None
        formal_authorization = _fake_formal_authorization_binding()
        formal_checkpoint = qualification._checkpoint_bytes(
            development_result=validated["results"][0],
            source_provenance=source,
            execution_mode="formal",
            formal_authorization=formal_authorization,
        )
        formal_record, formal_payload = qualification._checkpoint_record_from_bytes(
            formal_checkpoint,
            source_provenance=source,
            execution_mode="formal",
            development_results=validated["results"],
            formal_authorization=formal_authorization,
        )
        assert formal_payload["formal_authorization"] == formal_authorization
        formal_report = qualification._report_root(
            stage="development",
            execution_mode="formal",
            source_provenance=source,
            results=validated["results"],
            pending_ledger_sha256=None,
            checkpoint=formal_record,
            formal_authorization=formal_authorization,
        )
        assert formal_report["formal_authorization"] == formal_authorization
        formal_bindings = qualification._ledger_bindings(
            stage="development",
            execution_mode="formal",
            directory_pin=pin,
            source_provenance=source,
            formal_authorization=formal_authorization,
        )
        assert formal_bindings["formal_authorization"] == formal_authorization
        with pytest.raises(PermissionError, match="fake checkpoint"):
            qualification._checkpoint_bytes(
                development_result=validated["results"][0],
                source_provenance=source,
                execution_mode="fake_test",
                formal_authorization=formal_authorization,
            )
        with pytest.raises(PermissionError, match="fake report"):
            qualification._report_root(
                stage="development",
                execution_mode="fake_test",
                source_provenance=source,
                results=validated["results"],
                pending_ledger_sha256=None,
                checkpoint=validated["checkpoint"],
                formal_authorization=formal_authorization,
            )
        with pytest.raises(PermissionError, match="fake ledger"):
            qualification._ledger_bindings(
                stage="development",
                execution_mode="fake_test",
                directory_pin=pin,
                source_provenance=source,
                formal_authorization=formal_authorization,
            )
        ledger = qualification._parse_ledger_bytes(
            qualification._pinned_read_bytes(
                pin,
                paths["development_ledger"],
                label="development ledger",
            ),
            label="development ledger",
        )
        assert "formal_authorization" not in ledger["bindings"]
        reviewed = _development_hashes(pin)
        with pytest.raises(PermissionError, match="formal review"):
            qualification._review_development_bundle(
                directory_pin=pin,
                reviewed_checkpoint_sha256=reviewed["checkpoint_sha256"],
                reviewed_report_sha256=reviewed["report_sha256"],
                reviewed_development_ledger_sha256=reviewed["ledger_sha256"],
                expected_source=source,
            )
    assert observed_weights_only and set(observed_weights_only) == {True}


def test_report_schema_rejects_extra_keys_and_checkpoint_rejects_junk(tmp_path: Path) -> None:
    source = _fake_source()
    with _pinned_run(tmp_path) as pin:
        qualification._execute_development_for_tests(
            directory_pin=pin,
            evaluator=_FakeEvaluator(),
            source_provenance=source,
        )
        report = qualification._strict_json_loads(
            qualification._pinned_read_bytes(
                pin,
                qualification._artifact_paths(pin)["development_report"],
                label="development report",
            ),
            label="development report",
        )
        report["unexpected"] = True
        with pytest.raises(ValueError, match="schema differs"):
            qualification._validate_report(
                report,
                stage="development",
                expected_source=source,
                expected_execution_mode="fake_test",
            )
    with pytest.raises(pickle.UnpicklingError):
        qualification._load_checkpoint_payload(b"not a torch checkpoint")


def test_external_three_hash_seal_and_every_protected_boundary_revalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _fake_source()
    with _pinned_run(tmp_path) as pin:
        assert (
            qualification._execute_development_for_tests(
                directory_pin=pin,
                evaluator=_FakeEvaluator(),
                source_provenance=source,
            )
            == 0
        )
        reviewed = _development_hashes(pin)
        original = qualification._revalidate_reviewed_development
        original_prepare = qualification._AccessLedger.prepare_publication
        calls: list[bool] = []
        intent_durable = False

        def counted(*args: Any, **kwargs: Any) -> Any:
            assert intent_durable is False
            calls.append(kwargs["qualification_active"])
            return original(*args, **kwargs)

        def record_intent(
            ledger: qualification._AccessLedger,
            report_template: dict[str, Any],
        ) -> tuple[str, dict[str, Any]]:
            nonlocal intent_durable
            result = original_prepare(ledger, report_template)
            intent_durable = True
            return result

        monkeypatch.setattr(qualification, "_revalidate_reviewed_development", counted)
        monkeypatch.setattr(
            qualification._AccessLedger,
            "prepare_publication",
            record_intent,
        )
        evaluator = _FakeEvaluator()
        assert (
            qualification._execute_qualification_for_tests(
                directory_pin=pin,
                evaluator=evaluator,
                source_provenance=source,
                reviewed_checkpoint_sha256=reviewed["checkpoint_sha256"],
                reviewed_report_sha256=reviewed["report_sha256"],
                reviewed_development_ledger_sha256=reviewed["ledger_sha256"],
            )
            == 0
        )
        assert len(evaluator.calls) == 48
        assert [split for split, _ in evaluator.calls] == (
            ["selector"] * 16 + ["confirmation"] * 16 + ["final_test"] * 16
        )
        assert intent_durable is True
        expected_active_revalidations = 3 * 37 + 1 + 1
        assert calls == [False] + [True] * expected_active_revalidations
        report = qualification._strict_json_loads(
            qualification._pinned_read_bytes(
                pin,
                qualification._artifact_paths(pin)["qualification_report"],
                label="qualification report",
            ),
            label="qualification report",
        )
        validated = qualification._validate_report(
            report,
            stage="qualification",
            expected_source=source,
            expected_execution_mode="fake_test",
        )
        assert validated["passed"] is True
        assert validated["reviewed_development"] == reviewed


def test_reviewed_bundle_mutation_fails_at_next_protected_boundary(tmp_path: Path) -> None:
    source = _fake_source()
    with _pinned_run(tmp_path) as pin:
        qualification._execute_development_for_tests(
            directory_pin=pin,
            evaluator=_FakeEvaluator(),
            source_provenance=source,
        )
        reviewed = _development_hashes(pin)
        report_path = qualification._artifact_paths(pin)["development_report"]

        def mutate(call: int, _request: qualification.BatchEvaluationRequest) -> None:
            if call == 1:
                descriptor = os.open(report_path, os.O_WRONLY | os.O_TRUNC)
                try:
                    os.write(descriptor, b"{}\n")
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)

        with pytest.raises(PermissionError, match="reviewed development"):
            qualification._execute_qualification_for_tests(
                directory_pin=pin,
                evaluator=_FakeEvaluator(callback=mutate),
                source_provenance=source,
                reviewed_checkpoint_sha256=reviewed["checkpoint_sha256"],
                reviewed_report_sha256=reviewed["report_sha256"],
                reviewed_development_ledger_sha256=reviewed["ledger_sha256"],
            )
        ledger = qualification._strict_json_loads(
            qualification._pinned_read_bytes(
                pin,
                qualification._artifact_paths(pin)["qualification_ledger"],
                label="qualification error ledger",
            ),
            label="qualification error ledger",
        )
        assert ledger["status"] == "terminal_error"


def test_final_review_failure_precedes_normal_intent_and_cannot_publish_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _fake_source()
    with _pinned_run(tmp_path) as pin:
        qualification._execute_development_for_tests(
            directory_pin=pin,
            evaluator=_FakeEvaluator(),
            source_provenance=source,
        )
        reviewed = _development_hashes(pin)
        paths = qualification._artifact_paths(pin)
        original_revalidate = qualification._revalidate_reviewed_development
        original_prepare = qualification._AccessLedger.prepare_publication
        active_revalidations = 0
        normal_intent_calls = 0

        def fail_at_final_review(*args: Any, **kwargs: Any) -> Any:
            nonlocal active_revalidations
            if kwargs["qualification_active"] is True:
                active_revalidations += 1
                expected_active_revalidations = 3 * 37 + 1 + 1
                if active_revalidations == expected_active_revalidations:
                    descriptor = os.open(
                        paths["development_report"],
                        os.O_WRONLY | os.O_TRUNC,
                    )
                    try:
                        os.write(descriptor, b"{}\n")
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
            return original_revalidate(*args, **kwargs)

        def count_normal_intent(
            ledger: qualification._AccessLedger,
            report_template: dict[str, Any],
        ) -> tuple[str, dict[str, Any]]:
            nonlocal normal_intent_calls
            normal_intent_calls += 1
            return original_prepare(ledger, report_template)

        monkeypatch.setattr(
            qualification,
            "_revalidate_reviewed_development",
            fail_at_final_review,
        )
        monkeypatch.setattr(
            qualification._AccessLedger,
            "prepare_publication",
            count_normal_intent,
        )
        with pytest.raises(PermissionError, match="reviewed development"):
            qualification._execute_qualification_for_tests(
                directory_pin=pin,
                evaluator=_FakeEvaluator(),
                source_provenance=source,
                reviewed_checkpoint_sha256=reviewed["checkpoint_sha256"],
                reviewed_report_sha256=reviewed["report_sha256"],
                reviewed_development_ledger_sha256=reviewed["ledger_sha256"],
            )
        assert active_revalidations == 3 * 37 + 1 + 1
        assert normal_intent_calls == 0

        report_bytes = qualification._pinned_read_bytes(
            pin,
            paths["qualification_report"],
            label="final-review failure report",
        )
        report = qualification._validate_report(
            qualification._strict_json_loads(
                report_bytes,
                label="final-review failure report",
            ),
            stage="qualification",
            expected_source=source,
            expected_execution_mode="fake_test",
        )
        ledger_bytes = qualification._pinned_read_bytes(
            pin,
            paths["qualification_ledger"],
            label="final-review failure ledger",
        )
        terminal = qualification._validate_terminal_ledger(
            ledger_bytes,
            stage="qualification",
            expected_bindings=qualification._ledger_bindings(
                stage="qualification",
                execution_mode="fake_test",
                directory_pin=pin,
                source_provenance=source,
                reviewed_development=reviewed,
            ),
            expected_results=report["results"],
            expected_outcome="error",
            expected_error_report_sha256=qualification.sha256_bytes(report_bytes),
            expected_pending_ledger_sha256=report["pending_ledger_sha256"],
        )
        assert report["passed"] is False
        assert report["outcome"] == "error"
        assert report["error"]["type"] == "PermissionError"
        assert terminal["status"] == "terminal_error"
        assert terminal["publication"]["kind"] == "error"
        assert terminal["publication"]["predecessor_status"] == "final_test_passed"


def test_evaluator_exception_publishes_terminal_error_ledger_and_report(tmp_path: Path) -> None:
    source = _fake_source()
    with _pinned_run(tmp_path) as pin:
        with pytest.raises(RuntimeError, match="injected evaluator failure"):
            qualification._execute_development_for_tests(
                directory_pin=pin,
                evaluator=_FakeEvaluator(raise_on_call=1),
                source_provenance=source,
            )
        paths = qualification._artifact_paths(pin)
        report_bytes = qualification._pinned_read_bytes(
            pin,
            paths["development_report"],
            label="terminal error report",
        )
        report = qualification._validate_report(
            qualification._strict_json_loads(report_bytes, label="terminal error report"),
            stage="development",
            expected_source=source,
            expected_execution_mode="fake_test",
        )
        ledger = qualification._validate_terminal_ledger(
            qualification._pinned_read_bytes(
                pin,
                paths["development_ledger"],
                label="terminal error ledger",
            ),
            stage="development",
            expected_bindings=qualification._ledger_bindings(
                stage="development",
                execution_mode="fake_test",
                directory_pin=pin,
                source_provenance=source,
            ),
        )
        assert report["outcome"] == "error"
        assert report["pending_ledger_sha256"] == ledger["publication"]["pending_ledger_sha256"]
        assert ledger["status"] == "terminal_error"
        assert ledger["error"]["report_sha256"] == qualification.sha256_bytes(report_bytes)
        assert ledger["splits"]["development"]["public_active_batch"] == [0, 1, 2, 3]


@pytest.mark.parametrize("cutpoint", ["checkpoint", "intent", "report"])
def test_normal_publication_recovers_from_every_durable_cutpoint(
    tmp_path: Path,
    cutpoint: str,
) -> None:
    with _pinned_run(tmp_path) as pin:
        ledger, source, result, _checkpoint, template = (
            _complete_fake_development_before_publication(pin)
        )
        report_path = qualification._artifact_paths(pin)["development_report"]
        intended_report: dict[str, Any] | None = None
        published_identity: tuple[int, int, int, int, int] | None = None
        if cutpoint in {"intent", "report"}:
            _, intended_report = ledger.prepare_publication(template)
        if cutpoint == "report":
            ledger.publish_prepared_report()
            published_identity = qualification._pinned_file_identity(pin, report_path)

        _forget_fake_process_state()
        terminal_digest = qualification._recover_existing_attempt_for_tests(
            directory_pin=pin,
            stage="development",
            source_provenance=source,
        )

        report_bytes = qualification._pinned_read_bytes(
            pin,
            report_path,
            label=f"recovered normal report after {cutpoint}",
        )
        report = qualification._validate_report(
            qualification._strict_json_loads(
                report_bytes,
                label=f"recovered normal report after {cutpoint}",
            ),
            stage="development",
            expected_source=source,
            expected_execution_mode="fake_test",
        )
        ledger_bytes = qualification._pinned_read_bytes(
            pin,
            qualification._artifact_paths(pin)["development_ledger"],
            label=f"recovered normal ledger after {cutpoint}",
        )
        terminal = qualification._validate_terminal_ledger(
            ledger_bytes,
            stage="development",
            expected_bindings=qualification._ledger_bindings(
                stage="development",
                execution_mode="fake_test",
                directory_pin=pin,
                source_provenance=source,
            ),
            expected_results=[result],
            expected_outcome="passed",
            expected_normal_report_sha256=qualification.sha256_bytes(report_bytes),
            expected_pending_ledger_sha256=report["pending_ledger_sha256"],
        )
        assert terminal_digest == qualification.sha256_bytes(ledger_bytes)
        assert report["outcome"] == "passed"
        assert report["results"] == [result]
        assert terminal["publication"]["report_template"] == template
        assert terminal["publication"]["report_sha256"] == qualification.sha256_bytes(report_bytes)
        if intended_report is not None:
            assert report == intended_report
        if published_identity is not None:
            assert qualification._pinned_file_identity(pin, report_path) == published_identity


@pytest.mark.parametrize(
    "cutpoint",
    ["initial_ledger", "active_batch", "error_intent", "error_report"],
)
def test_error_publication_recovers_without_reopening_evidence(
    tmp_path: Path,
    cutpoint: str,
) -> None:
    with _pinned_run(tmp_path) as pin:
        ledger, bindings = _new_development_ledger(pin)
        source = _fake_source()
        report_path = qualification._artifact_paths(pin)["development_report"]
        if cutpoint != "initial_ledger":
            ledger.begin_split("development")
            ledger.reserve_batch("development", (0, 1, 2, 3))
        intended_report: dict[str, Any] | None = None
        published_identity: tuple[int, int, int, int, int] | None = None
        if cutpoint in {"error_intent", "error_report"}:
            template = qualification._report_root(
                stage="development",
                execution_mode="fake_test",
                source_provenance=source,
                results=[],
                pending_ledger_sha256=None,
                error={"type": "RuntimeError", "message": "injected stopped process"},
                opened_splits=["development"],
            )
            _, intended_report = ledger.prepare_error_publication(template)
        if cutpoint == "error_report":
            ledger.publish_prepared_report()
            published_identity = qualification._pinned_file_identity(pin, report_path)

        _forget_fake_process_state()
        terminal_digest = qualification._recover_existing_attempt_for_tests(
            directory_pin=pin,
            stage="development",
            source_provenance=source,
        )

        report_bytes = qualification._pinned_read_bytes(
            pin,
            report_path,
            label=f"recovered error report after {cutpoint}",
        )
        report = qualification._validate_report(
            qualification._strict_json_loads(
                report_bytes,
                label=f"recovered error report after {cutpoint}",
            ),
            stage="development",
            expected_source=source,
            expected_execution_mode="fake_test",
        )
        ledger_bytes = qualification._pinned_read_bytes(
            pin,
            qualification._artifact_paths(pin)["development_ledger"],
            label=f"recovered error ledger after {cutpoint}",
        )
        terminal = qualification._validate_terminal_ledger(
            ledger_bytes,
            stage="development",
            expected_bindings=bindings,
            expected_results=[],
            expected_outcome="error",
            expected_error_report_sha256=qualification.sha256_bytes(report_bytes),
            expected_pending_ledger_sha256=report["pending_ledger_sha256"],
        )
        assert terminal_digest == qualification.sha256_bytes(ledger_bytes)
        assert report["outcome"] == "error"
        assert terminal["status"] == "terminal_error"
        assert terminal["publication"]["report_sha256"] == qualification.sha256_bytes(report_bytes)
        if cutpoint == "initial_ledger":
            assert report["opened_splits"] == []
            assert report["error"]["type"] == "InterruptedRun"
        else:
            assert report["opened_splits"] == ["development"]
            assert terminal["splits"]["development"]["public_active_batch"] == [0, 1, 2, 3]
        if intended_report is not None:
            assert report == intended_report
        elif cutpoint == "active_batch":
            assert report["error"]["type"] == "InterruptedRun"
        if published_identity is not None:
            assert qualification._pinned_file_identity(pin, report_path) == published_identity


@pytest.mark.parametrize("outcome", ["normal", "error"])
def test_existing_terminal_bundle_is_recognized_idempotently(
    tmp_path: Path,
    outcome: str,
) -> None:
    source = _fake_source()
    with _pinned_run(tmp_path) as pin:
        if outcome == "normal":
            qualification._execute_development_for_tests(
                directory_pin=pin,
                evaluator=_FakeEvaluator(),
                source_provenance=source,
            )
        else:
            with pytest.raises(RuntimeError, match="injected evaluator failure"):
                qualification._execute_development_for_tests(
                    directory_pin=pin,
                    evaluator=_FakeEvaluator(raise_on_call=1),
                    source_provenance=source,
                )
        inventory = qualification._pinned_inventory(pin)
        before = {
            name: (
                qualification._pinned_read_bytes(
                    pin,
                    pin.path / name,
                    label=f"terminal {outcome} {name}",
                ),
                qualification._pinned_file_identity(pin, pin.path / name),
            )
            for name in inventory
        }
        ledger_bytes = before[qualification.DEVELOPMENT_LEDGER_NAME][0]
        expected_digest = qualification.sha256_bytes(ledger_bytes)

        _forget_fake_process_state()
        for _ in range(2):
            assert (
                qualification._recover_existing_attempt_for_tests(
                    directory_pin=pin,
                    stage="development",
                    source_provenance=source,
                )
                == expected_digest
            )
            assert qualification._pinned_inventory(pin) == inventory
            for name, (contents, identity) in before.items():
                assert (
                    qualification._pinned_read_bytes(
                        pin,
                        pin.path / name,
                        label=f"recognized terminal {outcome} {name}",
                    )
                    == contents
                )
                assert qualification._pinned_file_identity(pin, pin.path / name) == identity


def test_publication_recovery_survives_release_and_reacquire_pin(tmp_path: Path) -> None:
    path = (tmp_path / "restart-run").resolve()
    pin = qualification._acquire_pinned_directory(path, create=True, canonical=False)
    source: dict[str, Any]
    intended_report: dict[str, Any]
    try:
        ledger, source, _result, _checkpoint, template = (
            _complete_fake_development_before_publication(pin)
        )
        _, intended_report = ledger.prepare_publication(template)
    finally:
        qualification._release_pinned_directory(pin)
    _forget_fake_process_state()

    restarted = qualification._acquire_pinned_directory(
        path,
        create=False,
        canonical=False,
    )
    try:
        terminal_digest = qualification._recover_existing_attempt_for_tests(
            directory_pin=restarted,
            stage="development",
            source_provenance=source,
        )
        report_bytes = qualification._pinned_read_bytes(
            restarted,
            qualification._artifact_paths(restarted)["development_report"],
            label="restarted recovered report",
        )
        report = qualification._validate_report(
            qualification._strict_json_loads(report_bytes, label="restarted recovered report"),
            stage="development",
            expected_source=source,
            expected_execution_mode="fake_test",
        )
        ledger_bytes = qualification._pinned_read_bytes(
            restarted,
            qualification._artifact_paths(restarted)["development_ledger"],
            label="restarted recovered ledger",
        )
        assert report == intended_report
        assert terminal_digest == qualification.sha256_bytes(ledger_bytes)
    finally:
        qualification._release_pinned_directory(restarted)


def test_recovery_rejects_mismatched_existing_report_without_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _pinned_run(tmp_path) as pin:
        ledger, source, _result, _checkpoint, template = (
            _complete_fake_development_before_publication(pin)
        )
        ledger.prepare_publication(template)
        ledger.publish_prepared_report()
        _forget_fake_process_state()
        report_path = qualification._artifact_paths(pin)["development_report"]
        descriptor = os.open(report_path, os.O_WRONLY | os.O_TRUNC)
        try:
            os.write(descriptor, b'{"mismatch":true}\n')
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        mismatch = qualification._pinned_read_bytes(pin, report_path, label="mismatched report")
        identity = qualification._pinned_file_identity(pin, report_path)
        monkeypatch.setattr(
            qualification,
            "_formal_scene_module",
            lambda: pytest.fail("recovery accessed the formal scene"),
        )

        with pytest.raises(PermissionError, match="differs from durable ledger intent"):
            qualification._recover_existing_attempt_for_tests(
                directory_pin=pin,
                stage="development",
                source_provenance=source,
            )
        assert (
            qualification._pinned_read_bytes(
                pin,
                report_path,
                label="unchanged mismatched report",
            )
            == mismatch
        )
        assert qualification._pinned_file_identity(pin, report_path) == identity


def test_recovery_rejects_extra_entry_before_artifact_or_scene_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _pinned_run(tmp_path) as pin:
        _ledger, _bindings = _new_development_ledger(pin)
        source = _fake_source()
        _forget_fake_process_state()
        extra = pin.path / "unexpected.json"
        qualification._pinned_durable_create(pin, extra, b"{}\n")
        monkeypatch.setattr(
            qualification,
            "_formal_scene_module",
            lambda: pytest.fail("recovery accessed the formal scene"),
        )

        with pytest.raises(PermissionError, match="neither fresh nor recoverable"):
            qualification._recover_existing_attempt_for_tests(
                directory_pin=pin,
                stage="development",
                source_provenance=source,
            )
        assert qualification._pinned_inventory(pin) == frozenset(
            {qualification.DEVELOPMENT_LEDGER_NAME, extra.name}
        )
        assert not qualification._pinned_exists(
            pin,
            qualification._artifact_paths(pin)["development_report"],
            label="extra-entry rejected report",
        )


def test_recovery_rejects_corrupt_checkpoint_before_report_or_scene_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _pinned_run(tmp_path) as pin:
        _ledger, source, _result, _checkpoint, _template = (
            _complete_fake_development_before_publication(pin)
        )
        _forget_fake_process_state()
        paths = qualification._artifact_paths(pin)
        ledger_before = qualification._pinned_read_bytes(
            pin,
            paths["development_ledger"],
            label="pre-corruption ledger",
        )
        descriptor = os.open(paths["checkpoint"], os.O_WRONLY | os.O_TRUNC)
        try:
            os.write(descriptor, b"corrupt checkpoint")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        monkeypatch.setattr(
            qualification,
            "_formal_scene_module",
            lambda: pytest.fail("recovery accessed the formal scene"),
        )

        with pytest.raises((RuntimeError, ValueError, EOFError, pickle.UnpicklingError)):
            qualification._recover_existing_attempt_for_tests(
                directory_pin=pin,
                stage="development",
                source_provenance=source,
            )
        assert (
            qualification._pinned_read_bytes(
                pin,
                paths["development_ledger"],
                label="unchanged corrupt-checkpoint ledger",
            )
            == ledger_before
        )
        assert not qualification._pinned_exists(
            pin,
            paths["development_report"],
            label="corrupt-checkpoint rejected report",
        )


@pytest.mark.parametrize("corruption", ["tampered", "noncanonical"])
def test_recovery_rejects_tampered_or_noncanonical_intent_before_report_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    with _pinned_run(tmp_path) as pin:
        ledger, source, _result, _checkpoint, template = (
            _complete_fake_development_before_publication(pin)
        )
        ledger.prepare_publication(template)
        _forget_fake_process_state()
        paths = qualification._artifact_paths(pin)
        ledger_path = paths["development_ledger"]
        original = qualification._pinned_read_bytes(pin, ledger_path, label="original intent")
        if corruption == "tampered":
            envelope = qualification._strict_json_loads(original, label="original intent")
            envelope.pop("record_sha256")
            envelope["publication"]["report_template"]["passed"] = False
            corrupted = qualification._ledger_bytes(envelope)
        else:
            corrupted = original + b" "
        descriptor = os.open(ledger_path, os.O_WRONLY | os.O_TRUNC)
        try:
            os.write(descriptor, corrupted)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        identity = qualification._pinned_file_identity(pin, ledger_path)
        monkeypatch.setattr(
            qualification,
            "_formal_scene_module",
            lambda: pytest.fail("recovery accessed the formal scene"),
        )

        with pytest.raises((PermissionError, ValueError)):
            qualification._recover_existing_attempt_for_tests(
                directory_pin=pin,
                stage="development",
                source_provenance=source,
            )
        assert (
            qualification._pinned_read_bytes(
                pin,
                ledger_path,
                label=f"unchanged {corruption} intent",
            )
            == corrupted
        )
        assert qualification._pinned_file_identity(pin, ledger_path) == identity
        assert not qualification._pinned_exists(
            pin,
            paths["development_report"],
            label=f"{corruption} intent rejected report",
        )


def test_recovery_replays_rehashed_intent_predecessor_before_report_access(
    tmp_path: Path,
) -> None:
    with _pinned_run(tmp_path) as pin:
        ledger, source, _result, _checkpoint, template = (
            _complete_fake_development_before_publication(pin)
        )
        ledger.prepare_publication(template)
        paths = qualification._artifact_paths(pin)
        ledger_path = paths["development_ledger"]
        envelope = qualification._strict_json_loads(
            qualification._pinned_read_bytes(pin, ledger_path, label="original predecessor intent"),
            label="original predecessor intent",
        )
        envelope.pop("record_sha256")
        envelope["publication"]["predecessor_ledger_sha256"] = "b" * 64
        rebound = qualification._ledger_bytes(envelope)
        descriptor = os.open(ledger_path, os.O_WRONLY | os.O_TRUNC)
        try:
            os.write(descriptor, rebound)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _forget_fake_process_state()

        with pytest.raises(ValueError, match="predecessor"):
            qualification._recover_existing_attempt_for_tests(
                directory_pin=pin,
                stage="development",
                source_provenance=source,
            )
        assert not qualification._pinned_exists(
            pin,
            paths["development_report"],
            label="predecessor-rebound rejected report",
        )


@pytest.mark.parametrize("mutation", ["corrupt", "replacement"])
def test_recovered_normal_intent_validates_checkpoint_before_report_access(
    tmp_path: Path,
    mutation: str,
) -> None:
    with _pinned_run(tmp_path) as pin:
        ledger, source, _result, _checkpoint, template = (
            _complete_fake_development_before_publication(pin)
        )
        ledger.prepare_publication(template)
        paths = qualification._artifact_paths(pin)
        checkpoint_path = paths["checkpoint"]
        checkpoint_bytes = qualification._pinned_read_bytes(
            pin,
            checkpoint_path,
            label="pre-mutation checkpoint",
            maximum=qualification.MAX_CHECKPOINT_BYTES,
        )
        replacement = (
            b"corrupt checkpoint"
            if mutation == "corrupt"
            else checkpoint_bytes + b"post-intent replacement"
        )
        qualification._pinned_durable_replace(pin, checkpoint_path, replacement)
        _forget_fake_process_state()

        with pytest.raises((PermissionError, RuntimeError, ValueError, pickle.UnpicklingError)):
            qualification._recover_existing_attempt_for_tests(
                directory_pin=pin,
                stage="development",
                source_provenance=source,
            )
        assert not qualification._pinned_exists(
            pin,
            paths["development_report"],
            label=f"{mutation} checkpoint rejected report",
        )


@pytest.mark.parametrize(
    ("cutpoint", "mutate_reviewed", "expected_outcome"),
    [
        ("unbound", False, "passed"),
        ("intent", False, "passed"),
        ("unbound", True, "error"),
        ("intent", True, "rejected"),
    ],
)
def test_qualification_restart_validates_reviewed_trio_and_terminal_outcome(
    tmp_path: Path,
    cutpoint: str,
    mutate_reviewed: bool,
    expected_outcome: str,
) -> None:
    source = _fake_source()
    with _pinned_run(tmp_path) as pin:
        assert (
            qualification._execute_development_for_tests(
                directory_pin=pin,
                evaluator=_FakeEvaluator(),
                source_provenance=source,
            )
            == 0
        )
        reviewed = _development_hashes(pin)
        ledger, results, template = _complete_fake_qualification_before_publication(
            pin,
            source=source,
            reviewed=reviewed,
        )
        if cutpoint == "intent":
            ledger.prepare_publication(template)
        paths = qualification._artifact_paths(pin)
        if mutate_reviewed:
            qualification._pinned_durable_replace(
                pin,
                paths["development_report"],
                b"{}\n",
            )
        _forget_fake_process_state()

        if expected_outcome == "rejected":
            with pytest.raises((PermissionError, ValueError)):
                qualification._recover_existing_attempt_for_tests(
                    directory_pin=pin,
                    stage="qualification",
                    source_provenance=source,
                    reviewed_development=reviewed,
                )
            assert not qualification._pinned_exists(
                pin,
                paths["qualification_report"],
                label="post-intent reviewed mutation rejected report",
            )
            return

        terminal_digest = qualification._recover_existing_attempt_for_tests(
            directory_pin=pin,
            stage="qualification",
            source_provenance=source,
            reviewed_development=reviewed,
        )
        report_bytes = qualification._pinned_read_bytes(
            pin,
            paths["qualification_report"],
            label=f"recovered qualification {expected_outcome} report",
        )
        report = qualification._validate_report(
            qualification._strict_json_loads(
                report_bytes,
                label=f"recovered qualification {expected_outcome} report",
            ),
            stage="qualification",
            expected_source=source,
            expected_execution_mode="fake_test",
        )
        ledger_bytes = qualification._pinned_read_bytes(
            pin,
            paths["qualification_ledger"],
            label=f"recovered qualification {expected_outcome} ledger",
        )
        assert report["outcome"] == expected_outcome
        assert report["results"] == results
        assert terminal_digest == qualification.sha256_bytes(ledger_bytes)
        inventory = qualification._pinned_inventory(pin)
        _forget_fake_process_state()
        for _ in range(2):
            assert (
                qualification._recover_existing_attempt_for_tests(
                    directory_pin=pin,
                    stage="qualification",
                    source_provenance=source,
                    reviewed_development=reviewed,
                )
                == terminal_digest
            )
            assert qualification._pinned_inventory(pin) == inventory
        if expected_outcome == "passed" and cutpoint == "intent":
            qualification._pinned_durable_replace(
                pin,
                paths["checkpoint"],
                b"post-terminal reviewed checkpoint mutation",
            )
            _forget_fake_process_state()
            with pytest.raises((PermissionError, ValueError, pickle.UnpicklingError)):
                qualification._recover_existing_attempt_for_tests(
                    directory_pin=pin,
                    stage="qualification",
                    source_provenance=source,
                    reviewed_development=reviewed,
                )


def test_recovery_receipt_schema_binds_phase_predecessor_report_and_rejects_fake() -> None:
    source = _fake_source()
    authorization = _fake_formal_authorization_binding()
    intent_receipt = _fake_recovery_receipt()
    intent_template = qualification._report_root(
        stage="development",
        execution_mode="formal",
        source_provenance=source,
        results=[],
        pending_ledger_sha256=None,
        formal_authorization=authorization,
        error={
            "type": "InterruptedRun",
            "message": "recovered a reserved attempt without opening public evidence",
        },
        opened_splits=[],
    )
    intent_publication = {
        "state": "error_intent",
        "kind": "error",
        "predecessor_status": "reserved",
        "predecessor_ledger_sha256": "a" * 64,
        "report_template": intent_template,
        "report_template_sha256": qualification.canonical_sha256(intent_template),
        "pending_ledger_sha256": None,
        "report_sha256": None,
        "intent_generation": 1,
        "recovery_receipts": [intent_receipt],
    }
    assert qualification._validated_publication_recovery_receipts(
        intent_publication,
        stage="development",
        execution_mode="formal",
        expected_source=source,
        expected_run_directory=intent_receipt["run_directory"],
    ) == [intent_receipt]
    with pytest.raises(PermissionError, match="fake or empty"):
        qualification._validated_publication_recovery_receipts(
            intent_publication,
            stage="development",
            execution_mode="fake_test",
            expected_source=source,
            expected_run_directory=intent_receipt["run_directory"],
        )

    normal_template = {"outcome": "passed", "checkpoint": None}
    terminal_receipt = _fake_recovery_receipt(
        placement="terminal",
        action="recover_existing_normal_intent",
        outcome="complete_passed",
        predecessor_sha256="b" * 64,
        predecessor_generation=1,
        report_sha256="c" * 64,
    )
    terminal_publication = {
        **intent_publication,
        "state": "normal_bound",
        "kind": "normal",
        "report_template": normal_template,
        "report_template_sha256": qualification.canonical_sha256(normal_template),
        "pending_ledger_sha256": "b" * 64,
        "report_sha256": "c" * 64,
        "recovery_receipts": [terminal_receipt],
    }
    assert qualification._validated_publication_recovery_receipts(
        terminal_publication,
        stage="development",
        execution_mode="formal",
        expected_source=source,
        expected_run_directory=terminal_receipt["run_directory"],
    ) == [terminal_receipt]

    rebound = dict(terminal_receipt)
    rebound["predecessor_generation"] = True
    rebound_body = dict(rebound)
    rebound_body.pop("receipt_sha256")
    rebound["receipt_sha256"] = qualification.canonical_sha256(rebound_body)
    terminal_publication["recovery_receipts"] = [rebound]
    with pytest.raises(ValueError, match="root binding"):
        qualification._validated_publication_recovery_receipts(
            terminal_publication,
            stage="development",
            execution_mode="formal",
            expected_source=source,
            expected_run_directory=terminal_receipt["run_directory"],
        )


@pytest.mark.parametrize("receipt_placement", ["intent", "terminal"])
def test_recovery_receipt_replays_legacy_and_recovered_intent_transition(
    receipt_placement: str,
) -> None:
    source = _fake_source()
    original_authorization = _fake_formal_authorization_binding()
    original_authorization["first_receipt_sha256"] = "f" * 64
    run_directory = _fake_recovery_receipt()["run_directory"]
    bindings = {
        "execution_mode": "formal",
        "source_provenance": source,
        "run_directory": run_directory,
        "formal_authorization": original_authorization,
    }
    base = {
        "schema": "rgbd_known_action_access_ledger_v2",
        "artifact_kind": "rgbd_known_action_exactly_once_access_ledger",
        "architecture_attempt": qualification.ARCHITECTURE_ATTEMPT,
        "stage": "development",
        "execution_mode": "formal",
        "order": ["development"],
        "bindings": bindings,
        "batch_size": qualification.BATCH_SIZE,
        "scenes_per_split": qualification.SCENES_PER_SPLIT,
        "splits": {"development": qualification._empty_split_state()},
        "attempt_reserved": True,
        "status": "reserved",
        "generation": 0,
        "publication": qualification._empty_publication_state(),
    }
    template = qualification._report_root(
        stage="development",
        execution_mode="formal",
        source_provenance=source,
        results=[],
        pending_ledger_sha256=None,
        formal_authorization=original_authorization,
        error={
            "type": "InterruptedRun",
            "message": "recovered a reserved attempt without opening public evidence",
        },
        opened_splits=[],
    )
    publication = {
        "state": "error_intent",
        "kind": "error",
        "predecessor_status": "reserved",
        "predecessor_ledger_sha256": qualification.sha256_bytes(qualification._ledger_bytes(base)),
        "report_template": template,
        "report_template_sha256": qualification.canonical_sha256(template),
        "pending_ledger_sha256": None,
        "report_sha256": None,
        "intent_generation": 1,
    }
    if receipt_placement == "intent":
        publication["recovery_receipts"] = [
            _fake_recovery_receipt(
                predecessor_sha256=publication["predecessor_ledger_sha256"],
                predecessor_generation=0,
            )
        ]
    intent = {
        **base,
        "status": "publication_intent_error",
        "generation": 1,
        "publication": publication,
    }
    pending_sha256 = qualification.sha256_bytes(qualification._ledger_bytes(intent))
    report = qualification._complete_report_template(
        template,
        pending_ledger_sha256=pending_sha256,
        stage="development",
        expected_source=source,
        expected_execution_mode="formal",
        expected_formal_authorization=original_authorization,
    )
    report_sha256 = qualification.sha256_bytes(qualification._report_bytes(report))
    terminal_publication = {
        **publication,
        "state": "error_bound",
        "pending_ledger_sha256": pending_sha256,
        "report_sha256": report_sha256,
    }
    if receipt_placement == "terminal":
        terminal_publication["recovery_receipts"] = [
            _fake_recovery_receipt(
                placement="terminal",
                action="recover_existing_error_intent",
                predecessor_sha256=pending_sha256,
                predecessor_generation=1,
                report_sha256=report_sha256,
            )
        ]
    terminal = {
        **intent,
        "status": "terminal_error",
        "generation": 2,
        "publication": terminal_publication,
        "error": {
            "type": report["error"]["type"],
            "message": report["error"]["message"],
            "report_sha256": report_sha256,
        },
    }
    validated = qualification._validate_terminal_ledger(
        qualification._ledger_bytes(terminal),
        stage="development",
        expected_bindings=bindings,
        expected_results=[],
        expected_outcome="error",
        expected_error_report_sha256=report_sha256,
        expected_pending_ledger_sha256=pending_sha256,
    )
    assert validated["bindings"]["formal_authorization"] == original_authorization
    assert validated["publication"]["recovery_receipts"][0]["placement"] == (receipt_placement)


@pytest.mark.parametrize(
    ("action", "outcome", "inventory_names", "message"),
    [
        (
            "prepare_completed_unbound",
            "complete_passed",
            None,
            "action/outcome",
        ),
        (
            "interrupt_nonterminal",
            "terminal_error",
            {
                qualification.DEVELOPMENT_LEDGER_NAME,
                qualification.CHECKPOINT_NAME,
            },
            "predecessor inventory",
        ),
    ],
)
def test_terminal_replay_rejects_canonically_rehashed_receipt_contradictions(
    action: str,
    outcome: str,
    inventory_names: set[str] | None,
    message: str,
) -> None:
    source = _fake_source()
    original_authorization = _fake_formal_authorization_binding()
    original_authorization["first_receipt_sha256"] = "f" * 64
    run_directory = _fake_recovery_receipt()["run_directory"]
    bindings = {
        "execution_mode": "formal",
        "source_provenance": source,
        "run_directory": run_directory,
        "formal_authorization": original_authorization,
    }
    base = {
        "schema": "rgbd_known_action_access_ledger_v2",
        "artifact_kind": "rgbd_known_action_exactly_once_access_ledger",
        "architecture_attempt": qualification.ARCHITECTURE_ATTEMPT,
        "stage": "development",
        "execution_mode": "formal",
        "order": ["development"],
        "bindings": bindings,
        "batch_size": qualification.BATCH_SIZE,
        "scenes_per_split": qualification.SCENES_PER_SPLIT,
        "splits": {"development": qualification._empty_split_state()},
        "attempt_reserved": True,
        "status": "reserved",
        "generation": 0,
        "publication": qualification._empty_publication_state(),
    }
    predecessor_sha256 = qualification.sha256_bytes(qualification._ledger_bytes(base))
    contradictory = _fake_recovery_receipt(
        action=action,
        outcome=outcome,
        predecessor_sha256=predecessor_sha256,
        inventory_names=inventory_names,
    )
    assert qualification._validated_ledger_recovery_receipt(contradictory) == contradictory
    template = qualification._report_root(
        stage="development",
        execution_mode="formal",
        source_provenance=source,
        results=[],
        pending_ledger_sha256=None,
        formal_authorization=original_authorization,
        error={
            "type": "InterruptedRun",
            "message": "recovered a reserved attempt without opening public evidence",
        },
        opened_splits=[],
    )
    intent = {
        **base,
        "status": "publication_intent_error",
        "generation": 1,
        "publication": {
            "state": "error_intent",
            "kind": "error",
            "predecessor_status": "reserved",
            "predecessor_ledger_sha256": predecessor_sha256,
            "report_template": template,
            "report_template_sha256": qualification.canonical_sha256(template),
            "pending_ledger_sha256": None,
            "report_sha256": None,
            "intent_generation": 1,
            "recovery_receipts": [contradictory],
        },
    }
    pending_sha256 = qualification.sha256_bytes(qualification._ledger_bytes(intent))
    report = qualification._complete_report_template(
        template,
        pending_ledger_sha256=pending_sha256,
        stage="development",
        expected_source=source,
        expected_execution_mode="formal",
        expected_formal_authorization=original_authorization,
    )
    report_sha256 = qualification.sha256_bytes(qualification._report_bytes(report))
    terminal = {
        **intent,
        "status": "terminal_error",
        "generation": 2,
        "publication": {
            **intent["publication"],
            "state": "error_bound",
            "pending_ledger_sha256": pending_sha256,
            "report_sha256": report_sha256,
        },
        "error": {
            "type": report["error"]["type"],
            "message": report["error"]["message"],
            "report_sha256": report_sha256,
        },
    }
    with pytest.raises(ValueError, match=message):
        qualification._validate_terminal_ledger(
            qualification._ledger_bytes(terminal),
            stage="development",
            expected_bindings=bindings,
            expected_results=[],
            expected_outcome="error",
            expected_error_report_sha256=report_sha256,
            expected_pending_ledger_sha256=pending_sha256,
        )


def test_formal_recovery_authority_is_inventory_bound_one_shot_and_content_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _fake_source()
    canonical = (tmp_path / "canonical-recovery").resolve()
    monkeypatch.setattr(qualification, "_canonical_run_directory", lambda: canonical)
    outer = _synthetic_outer_authority(stage="development", source=source)
    pin: qualification._PinnedDirectory | None = None
    recovery: qualification._LedgerRecoveryAuthorization | None = None
    try:
        pin = qualification._acquire_pinned_directory(
            canonical,
            create=True,
            canonical=True,
            outer_authority=outer,
        )
        ledger_path = pin.path / qualification.DEVELOPMENT_LEDGER_NAME
        qualification._pinned_durable_create(pin, ledger_path, b"opaque static ledger bytes")
        recovery = qualification._mint_ledger_recovery_authorization(
            outer,
            stage="development",
            directory_pin=pin,
            source_provenance=source,
        )
        monkeypatch.setattr(
            qualification,
            "_pinned_read_bytes",
            lambda *_args, **_kwargs: pytest.fail(
                "recovery authorization read governed artifact contents"
            ),
        )
        with pytest.raises(PermissionError, match="outer-runner authority"):
            qualification._consume_ledger_recovery_authorization(
                recovery,
                stage="qualification",
                directory_pin=pin,
                source_provenance=source,
                ledger_path=ledger_path,
            )
        inventory, runner_authorization = qualification._consume_ledger_recovery_authorization(
            recovery,
            stage="development",
            directory_pin=pin,
            source_provenance=source,
            ledger_path=ledger_path,
        )
        assert [entry["name"] for entry in inventory["entries"]] == [
            qualification.DEVELOPMENT_LEDGER_NAME
        ]
        assert runner_authorization == qualification._formal_authorization_binding(outer)
        with pytest.raises(PermissionError, match="replayed"):
            qualification._consume_ledger_recovery_authorization(
                recovery,
                stage="development",
                directory_pin=pin,
                source_provenance=source,
                ledger_path=ledger_path,
            )
    finally:
        if recovery is not None:
            qualification._revoke_ledger_recovery_authorization(recovery)
        if pin is not None and qualification._PIN_REGISTRY.get(id(pin)) is pin:
            qualification._release_pinned_directory(pin)
        if recovery is not None:
            qualification._LEDGER_RECOVERY_AUTHORIZATION_REGISTRY.pop(id(recovery), None)
        _forget_synthetic_outer_authority(outer)


@pytest.mark.parametrize(
    "mutation",
    ["wrong_pin", "wrong_thread", "inventory_change", "ledger_identity_change"],
)
def test_formal_recovery_authority_rejects_rebound_cut_before_consumption(
    mutation: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _fake_source()
    canonical = (tmp_path / f"canonical-recovery-{mutation}").resolve()
    monkeypatch.setattr(qualification, "_canonical_run_directory", lambda: canonical)
    outer = _synthetic_outer_authority(stage="development", source=source)
    pin: qualification._PinnedDirectory | None = None
    other_pin: qualification._PinnedDirectory | None = None
    recovery: qualification._LedgerRecoveryAuthorization | None = None
    try:
        pin = qualification._acquire_pinned_directory(
            canonical,
            create=True,
            canonical=True,
            outer_authority=outer,
        )
        ledger_path = pin.path / qualification.DEVELOPMENT_LEDGER_NAME
        qualification._pinned_durable_create(pin, ledger_path, b"opaque recovery ledger")
        recovery = qualification._mint_ledger_recovery_authorization(
            outer,
            stage="development",
            directory_pin=pin,
            source_provenance=source,
        )
        if mutation == "wrong_pin":
            other_pin = qualification._acquire_pinned_directory(
                (tmp_path / "other-recovery-pin").resolve(),
                create=True,
                canonical=False,
            )
            with pytest.raises(PermissionError, match="rebound or stale"):
                qualification._consume_ledger_recovery_authorization(
                    recovery,
                    stage="development",
                    directory_pin=other_pin,
                    source_provenance=source,
                    ledger_path=other_pin.path / qualification.DEVELOPMENT_LEDGER_NAME,
                )
        elif mutation == "wrong_thread":
            failures: list[BaseException] = []

            def cross_thread_consume() -> None:
                try:
                    qualification._consume_ledger_recovery_authorization(
                        recovery,
                        stage="development",
                        directory_pin=pin,
                        source_provenance=source,
                        ledger_path=ledger_path,
                    )
                except BaseException as error:
                    failures.append(error)

            thread = qualification.threading.Thread(target=cross_thread_consume)
            thread.start()
            thread.join()
            assert len(failures) == 1
            assert isinstance(failures[0], PermissionError)
        elif mutation == "inventory_change":
            qualification._pinned_durable_create(
                pin,
                pin.path / qualification.DEVELOPMENT_REPORT_NAME,
                b"post-authorization inventory mutation",
            )
            with pytest.raises(PermissionError, match="rebound or stale"):
                qualification._consume_ledger_recovery_authorization(
                    recovery,
                    stage="development",
                    directory_pin=pin,
                    source_provenance=source,
                    ledger_path=ledger_path,
                )
        else:
            qualification._pinned_durable_replace(
                pin,
                ledger_path,
                b"post-authorization ledger replacement",
            )
            with pytest.raises(PermissionError, match="rebound or stale"):
                qualification._consume_ledger_recovery_authorization(
                    recovery,
                    stage="development",
                    directory_pin=pin,
                    source_provenance=source,
                    ledger_path=ledger_path,
                )
        assert qualification._LEDGER_RECOVERY_AUTHORIZATION_REGISTRY[id(recovery)] == (
            recovery,
            outer,
            "issued",
            None,
            "authorization_issued",
            None,
        )
    finally:
        if recovery is not None:
            qualification._revoke_ledger_recovery_authorization(recovery)
        if other_pin is not None and qualification._PIN_REGISTRY.get(id(other_pin)) is other_pin:
            qualification._release_pinned_directory(other_pin)
        if pin is not None and qualification._PIN_REGISTRY.get(id(pin)) is pin:
            qualification._release_pinned_directory(pin)
        if recovery is not None:
            qualification._LEDGER_RECOVERY_AUTHORIZATION_REGISTRY.pop(id(recovery), None)
        _forget_synthetic_outer_authority(outer)


def test_formal_recovery_receipt_is_exactly_once_and_action_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _consumed_formal_recovery_authority(
        tmp_path,
        monkeypatch,
        suffix="receipt-exactly-once",
    ) as (_, outer, _, recovery, ledger_path):
        predecessor = ledger_path.read_bytes()
        receipt = qualification._ledger_recovery_receipt(
            recovery,
            predecessor_ledger_bytes=predecessor,
            predecessor_generation=0,
            placement="intent",
            action="interrupt_nonterminal",
            outcome="terminal_error",
            report_sha256=None,
        )
        registration = qualification._LEDGER_RECOVERY_AUTHORIZATION_REGISTRY[id(recovery)]
        assert registration == (
            recovery,
            outer,
            "receipt_minted",
            qualification._ledger_recovery_receipt_registry_binding(receipt),
            "intent_receipt_minted",
            None,
        )
        with pytest.raises(PermissionError, match="consumed authority"):
            qualification._ledger_recovery_receipt(
                recovery,
                predecessor_ledger_bytes=predecessor,
                predecessor_generation=0,
                placement="intent",
                action="prepare_completed_unbound",
                outcome="complete_failed",
                report_sha256=None,
            )
        assert qualification._LEDGER_RECOVERY_AUTHORIZATION_REGISTRY[id(recovery)] == registration


@pytest.mark.parametrize("failure", ["reentrant", "binding"])
def test_formal_recovery_failed_mint_is_burned_and_cannot_retry(
    failure: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _consumed_formal_recovery_authority(
        tmp_path,
        monkeypatch,
        suffix=f"receipt-failure-{failure}",
    ) as (_, outer, _, recovery, ledger_path):
        predecessor = ledger_path.read_bytes()
        original_candidate = qualification._ledger_recovery_receipt_candidate
        original_binding = qualification._ledger_recovery_receipt_registry_binding

        if failure == "reentrant":

            def reentrant_candidate(*args: Any, **kwargs: Any) -> dict[str, Any]:
                qualification._ledger_recovery_receipt(*args, **kwargs)
                return original_candidate(*args, **kwargs)

            monkeypatch.setattr(
                qualification,
                "_ledger_recovery_receipt_candidate",
                reentrant_candidate,
            )
        else:
            monkeypatch.setattr(
                qualification,
                "_ledger_recovery_receipt_registry_binding",
                lambda _receipt: (_ for _ in ()).throw(
                    RuntimeError("injected receipt binding failure")
                ),
            )
        with pytest.raises((PermissionError, RuntimeError)):
            qualification._ledger_recovery_receipt(
                recovery,
                predecessor_ledger_bytes=predecessor,
                predecessor_generation=0,
                placement="intent",
                action="interrupt_nonterminal",
                outcome="terminal_error",
                report_sha256=None,
            )
        registration = qualification._LEDGER_RECOVERY_AUTHORIZATION_REGISTRY[id(recovery)]
        assert registration == (
            recovery,
            outer,
            "receipt_failed",
            None,
            "receipt_failed",
            None,
        )
        monkeypatch.setattr(
            qualification,
            "_ledger_recovery_receipt_candidate",
            original_candidate,
        )
        monkeypatch.setattr(
            qualification,
            "_ledger_recovery_receipt_registry_binding",
            original_binding,
        )
        with pytest.raises(PermissionError, match="consumed authority"):
            qualification._ledger_recovery_receipt(
                recovery,
                predecessor_ledger_bytes=predecessor,
                predecessor_generation=0,
                placement="intent",
                action="interrupt_nonterminal",
                outcome="terminal_error",
                report_sha256=None,
            )


def test_consumed_recovery_detects_same_inode_restored_mtime_changed_ctime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _consumed_formal_recovery_authority(
        tmp_path,
        monkeypatch,
        suffix="post-consumption-ctime",
    ) as (_, _, _, recovery, ledger_path):
        before = ledger_path.stat()
        contents = ledger_path.read_bytes()
        changed = bytes([contents[0] ^ 1]) + contents[1:]
        with ledger_path.open("r+b") as handle:
            handle.write(changed)
            handle.flush()
            os.fsync(handle.fileno())
        os.utime(
            ledger_path,
            ns=(before.st_atime_ns, before.st_mtime_ns),
        )
        after = ledger_path.stat()
        assert (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) == (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        assert after.st_ctime_ns != before.st_ctime_ns
        with pytest.raises(PermissionError, match="changed after authorization"):
            qualification._validated_live_recovery_inventory(
                recovery,
                expected_state="consumed",
                label="same-inode adversarial recapture",
            )


def test_consumed_qualification_recovery_detects_other_base_artifact_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _consumed_formal_recovery_authority(
        tmp_path,
        monkeypatch,
        stage="qualification",
        suffix="post-consumption-other-base",
    ) as (_, _, pin, recovery, _):
        report_path = pin.path / qualification.DEVELOPMENT_REPORT_NAME
        qualification._pinned_durable_replace(
            pin,
            report_path,
            b"replacement development report",
        )
        with pytest.raises(PermissionError, match="changed after authorization"):
            qualification._validated_live_recovery_inventory(
                recovery,
                expected_state="consumed",
                label="other-base adversarial recapture",
            )


def test_formal_entry_finally_revokes_unconsumed_recovery_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _fake_source()
    canonical = (tmp_path / "canonical-finally-recovery").resolve()
    canonical.mkdir()
    (canonical / qualification.DEVELOPMENT_LEDGER_NAME).write_bytes(b"opaque recovery ledger")
    monkeypatch.setattr(qualification, "_canonical_run_directory", lambda: canonical)
    outer = _synthetic_outer_authority(stage="development", source=source)
    monkeypatch.setattr(
        qualification,
        "_recover_existing_formal_attempt",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("injected recovery failure")),
    )
    monkeypatch.setattr(
        qualification,
        "_activate_runtime_dependencies",
        lambda: pytest.fail("failed recovery activated the fresh runtime"),
    )
    try:
        with pytest.raises(RuntimeError, match="injected recovery failure"):
            qualification.run_development(
                source_provenance=source,
                runner_authority=outer,
            )
        assert qualification._PIN_REGISTRY == {}
        assert qualification._FORMAL_PIN_AUTHORITY_REGISTRY == {}
        registrations = list(qualification._LEDGER_RECOVERY_AUTHORIZATION_REGISTRY.values())
        assert len(registrations) == 1
        assert registrations[0][1] is outer
        assert registrations[0][2] == "revoked"
        assert qualification._OUTER_RUNNER_AUTHORITY_REGISTRY[id(outer)] == (
            outer,
            "revoked",
        )
    finally:
        qualification._LEDGER_RECOVERY_AUTHORIZATION_REGISTRY.clear()
        _forget_synthetic_outer_authority(outer)


def test_formal_recovery_rejects_fresh_fake_and_forged_before_content_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _fake_source()
    canonical = (tmp_path / "fresh-canonical-recovery").resolve()
    monkeypatch.setattr(qualification, "_canonical_run_directory", lambda: canonical)
    outer = _synthetic_outer_authority(stage="development", source=source)
    pin: qualification._PinnedDirectory | None = None
    try:
        pin = qualification._acquire_pinned_directory(
            canonical,
            create=True,
            canonical=True,
            outer_authority=outer,
        )
        with pytest.raises(PermissionError, match="existing canonical attempt"):
            qualification._mint_ledger_recovery_authorization(
                outer,
                stage="development",
                directory_pin=pin,
                source_provenance=source,
            )
    finally:
        if pin is not None and qualification._PIN_REGISTRY.get(id(pin)) is pin:
            qualification._release_pinned_directory(pin)
        _forget_synthetic_outer_authority(outer)

    fake_path = (tmp_path / "fake-forged-recovery").resolve()
    fake_pin = qualification._acquire_pinned_directory(
        fake_path,
        create=True,
        canonical=False,
    )
    try:
        qualification._pinned_durable_create(
            fake_pin,
            fake_pin.path / qualification.DEVELOPMENT_LEDGER_NAME,
            b"opaque static ledger bytes",
        )
        forged = qualification._LedgerRecoveryAuthorization(
            stage="development",
            execution_mode="formal",
            directory_pin=fake_pin,
            run_directory_sha256="1" * 64,
            runner_authorization_bytes=qualification._canonical_json(
                _fake_formal_authorization_binding()
            ),
            source_provenance_sha256=qualification.canonical_sha256(source),
            inventory_bytes=qualification._canonical_json(_fake_recovery_inventory("development")),
            ledger_name=qualification.DEVELOPMENT_LEDGER_NAME,
            ledger_identity=(1, 2, 0o100600, 3, 4, 5, 1),
            outer_authority_identity=1,
            owner_thread=qualification.threading.get_ident(),
            nonce=object(),
        )
        monkeypatch.setattr(
            qualification,
            "_pinned_read_bytes",
            lambda *_args, **_kwargs: pytest.fail(
                "forged recovery read governed artifact contents"
            ),
        )
        with pytest.raises(PermissionError, match="forged or replayed"):
            qualification._recover_existing_formal_attempt(
                directory_pin=fake_pin,
                stage="development",
                source_provenance=source,
                authorization=forged,
            )
    finally:
        qualification._release_pinned_directory(fake_pin)


def test_lightweight_formal_interruption_recovery_avoids_all_heavy_imports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, module, _ = _lightweight_qualification_module()
    source = _fake_source()
    canonical = (tmp_path / "lightweight-interrupted-recovery").resolve()
    monkeypatch.setattr(module, "_canonical_run_directory", lambda: canonical)
    first = _synthetic_outer_authority_for_module(
        module,
        stage="development",
        source=source,
        marker="first",
    )
    pin = module._acquire_pinned_directory(
        canonical,
        create=True,
        canonical=True,
        outer_authority=first,
    )
    try:
        bindings = module._ledger_bindings(
            stage="development",
            execution_mode="formal",
            directory_pin=pin,
            source_provenance=source,
            formal_authorization=module._formal_authorization_binding(first),
        )
        record = {
            "schema": "rgbd_known_action_access_ledger_v2",
            "artifact_kind": "rgbd_known_action_exactly_once_access_ledger",
            "architecture_attempt": module.ARCHITECTURE_ATTEMPT,
            "stage": "development",
            "execution_mode": "formal",
            "order": ["development"],
            "bindings": bindings,
            "batch_size": module.BATCH_SIZE,
            "scenes_per_split": module.SCENES_PER_SPLIT,
            "splits": {"development": module._empty_split_state()},
            "attempt_reserved": True,
            "status": "reserved",
            "generation": 0,
            "publication": module._empty_publication_state(),
        }
        module._pinned_durable_create(
            pin,
            pin.path / module.DEVELOPMENT_LEDGER_NAME,
            module._ledger_bytes(record),
        )
    finally:
        module._release_pinned_directory(pin)
        _forget_synthetic_outer_authority_for_module(module, first)

    second = _synthetic_outer_authority_for_module(
        module,
        stage="development",
        source=source,
        marker="second",
    )
    imported: list[str] = []

    def forbidden_import(name: str, *args: Any, **kwargs: Any) -> Any:
        imported.append(name)
        pytest.fail(f"lightweight interrupted recovery imported {name}")

    monkeypatch.setattr(module.importlib, "import_module", forbidden_import)
    monkeypatch.setattr(
        module,
        "_require_scene_freeze",
        lambda: pytest.fail("interrupted recovery touched the scene freeze"),
    )
    monkeypatch.setattr(
        module,
        "_activate_runtime_dependencies",
        lambda: pytest.fail("interrupted recovery activated the fresh runtime"),
    )
    monkeypatch.setattr(
        module,
        "_activate_recovery_checkpoint_dependency",
        lambda: pytest.fail("interrupted recovery activated Torch"),
    )
    try:
        assert (
            module.run_development(
                source_provenance=source,
                runner_authority=second,
            )
            == 1
        )
        assert imported == []
        assert module.torch is None
        assert module.OrpheusConfig is None
        assert module.load_config is None
        assert module._RUNTIME_DEPENDENCIES_ACTIVE is False
        terminal = module._parse_ledger_bytes(
            (canonical / module.DEVELOPMENT_LEDGER_NAME).read_bytes(),
            label="temp-isolated interrupted terminal",
        )
        assert terminal["status"] == "terminal_error"
        assert terminal["publication"]["recovery_receipts"][0]["action"] == (
            "interrupt_nonterminal"
        )
        report = module._strict_json_loads(
            (canonical / module.DEVELOPMENT_REPORT_NAME).read_bytes(),
            label="temp-isolated interrupted report",
        )
        assert report["error"] == {
            "type": "InterruptedRun",
            "message": "recovered a reserved attempt without opening public evidence",
        }
        assert module._PIN_REGISTRY == {}
        assert all(
            registration[2] == "revoked"
            for registration in module._LEDGER_RECOVERY_AUTHORIZATION_REGISTRY.values()
        )
    finally:
        _forget_synthetic_outer_authority_for_module(module, second)
        module._LEDGER_RECOVERY_AUTHORIZATION_REGISTRY.clear()
        module._FORMAL_PIN_AUTHORITY_REGISTRY.clear()


def test_lightweight_completed_recovery_imports_only_torch_after_classification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, module, _ = _lightweight_qualification_module()
    source = _fake_source()
    canonical = (tmp_path / "lightweight-checkpoint-recovery").resolve()
    monkeypatch.setattr(module, "_canonical_run_directory", lambda: canonical)
    first = _synthetic_outer_authority_for_module(
        module,
        stage="development",
        source=source,
        marker="first",
    )
    pin = module._acquire_pinned_directory(
        canonical,
        create=True,
        canonical=True,
        outer_authority=first,
    )
    ledger: Any = None
    try:
        formal_authorization = module._formal_authorization_binding(first)
        bindings = module._ledger_bindings(
            stage="development",
            execution_mode="formal",
            directory_pin=pin,
            source_provenance=source,
            formal_authorization=formal_authorization,
        )
        seal = module._mint_runner_invocation_seal_from_outer(
            first,
            stage="development",
            directory_pin=pin,
            source_provenance=source,
        )
        authorization = module._authorize_ledger_creation(
            invocation_seal=seal,
            stage="development",
            directory_pin=pin,
            source_provenance=source,
            bindings=bindings,
            ledger_path=pin.path / module.DEVELOPMENT_LEDGER_NAME,
        )
        ledger = module._AccessLedger(
            pin.path / module.DEVELOPMENT_LEDGER_NAME,
            stage="development",
            bindings=bindings,
            directory_pin=pin,
            authorization=authorization,
        )
        result = _complete_lightweight_formal_split_for_recovery_fixture(
            module,
            ledger=ledger,
            split="development",
        )
        assert result["passed"] is True
        module.torch = qualification.torch
        checkpoint_bytes = module._checkpoint_bytes(
            development_result=result,
            source_provenance=source,
            execution_mode="formal",
            formal_authorization=formal_authorization,
        )
        module._pinned_durable_create(
            pin,
            pin.path / module.CHECKPOINT_NAME,
            checkpoint_bytes,
        )
        module.torch = None
    finally:
        if module._PIN_REGISTRY.get(id(pin)) is pin:
            module._release_pinned_directory(pin)
        if ledger is not None:
            module._LEDGER_REGISTRY.pop(id(ledger), None)
            module._LEDGER_SLOT_REGISTRY.pop(
                ("development", os.fspath(canonical)),
                None,
            )
        module._RUN_AUTHORIZATION_REGISTRY.clear()
        module._INVOCATION_SEAL_REGISTRY.clear()
        _forget_synthetic_outer_authority_for_module(module, first)

    second = _synthetic_outer_authority_for_module(
        module,
        stage="development",
        source=source,
        marker="second",
    )
    imported: list[str] = []
    isolated_modules = {module.__name__: module}
    isolated_sys = qualification.types.SimpleNamespace(modules=isolated_modules)

    def torch_only_import(name: str, *args: Any, **kwargs: Any) -> Any:
        imported.append(name)
        if name == "torch":
            registrations = list(module._LEDGER_RECOVERY_AUTHORIZATION_REGISTRY.values())
            assert len(registrations) == 1
            assert registrations[0][2] == "consumed"
            isolated_modules["torch"] = qualification.torch
            return qualification.torch
        pytest.fail(f"checkpoint recovery imported project/runtime module {name}")

    def forbidden_owner_import(name: str, *args: Any, **kwargs: Any) -> Any:
        pytest.fail(f"durable completed recovery imported project owner {name}")

    monkeypatch.setattr(module.importlib, "import_module", forbidden_owner_import)
    monkeypatch.setitem(
        module._activate_recovery_checkpoint_dependency.__kwdefaults__,
        "_sys",
        isolated_sys,
    )
    monkeypatch.setitem(
        module._activate_recovery_checkpoint_dependency.__kwdefaults__,
        "_import_module",
        torch_only_import,
    )
    monkeypatch.setattr(
        module,
        "_require_scene_freeze",
        lambda: pytest.fail("completed recovery touched the scene freeze"),
    )
    monkeypatch.setattr(
        module,
        "_activate_runtime_dependencies",
        lambda: pytest.fail("completed recovery activated the fresh runtime"),
    )
    try:
        assert (
            module.run_development(
                source_provenance=source,
                runner_authority=second,
            )
            == 0
        )
        assert imported == ["torch"]
        assert module.torch is qualification.torch
        assert module.OrpheusConfig is None
        assert module.load_config is None
        assert module._RUNTIME_DEPENDENCIES_ACTIVE is False
        terminal = module._parse_ledger_bytes(
            (canonical / module.DEVELOPMENT_LEDGER_NAME).read_bytes(),
            label="temp-isolated checkpoint terminal",
        )
        assert terminal["status"] == "complete_passed"
        assert [
            (receipt["placement"], receipt["action"], receipt["outcome"])
            for receipt in terminal["publication"]["recovery_receipts"]
        ] == [("intent", "prepare_completed_unbound", "complete_passed")]
        assert module._PIN_REGISTRY == {}
        assert all(
            registration[2] == "revoked"
            for registration in module._LEDGER_RECOVERY_AUTHORIZATION_REGISTRY.values()
        )
    finally:
        _forget_synthetic_outer_authority_for_module(module, second)
        module._LEDGER_RECOVERY_AUTHORIZATION_REGISTRY.clear()
        module._FORMAL_PIN_AUTHORITY_REGISTRY.clear()

    terminal_bytes = (canonical / module.DEVELOPMENT_LEDGER_NAME).read_bytes()
    recognizing = _synthetic_outer_authority_for_module(
        module,
        stage="development",
        source=source,
        marker="completed-terminal-recognition",
    )
    try:
        assert (
            module.run_development(
                source_provenance=source,
                runner_authority=recognizing,
            )
            == 0
        )
        assert imported == ["torch"]
        assert (canonical / module.DEVELOPMENT_LEDGER_NAME).read_bytes() == terminal_bytes
    finally:
        _forget_synthetic_outer_authority_for_module(module, recognizing)
        module._LEDGER_RECOVERY_AUTHORIZATION_REGISTRY.clear()
        module._FORMAL_PIN_AUTHORITY_REGISTRY.clear()


def test_lightweight_completed_failed_recovery_and_terminal_recognition_import_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, module, _ = _lightweight_qualification_module()
    source = _fake_source()
    canonical = (tmp_path / "lightweight-completed-failed").resolve()
    monkeypatch.setattr(module, "_canonical_run_directory", lambda: canonical)
    seeding = _synthetic_outer_authority_for_module(
        module,
        stage="development",
        source=source,
        marker="completed-failed-seed",
    )
    pin = module._acquire_pinned_directory(
        canonical,
        create=True,
        canonical=True,
        outer_authority=seeding,
    )
    ledger: Any = None
    try:
        formal_authorization = module._formal_authorization_binding(seeding)
        bindings = module._ledger_bindings(
            stage="development",
            execution_mode="formal",
            directory_pin=pin,
            source_provenance=source,
            formal_authorization=formal_authorization,
        )
        seal = module._mint_runner_invocation_seal_from_outer(
            seeding,
            stage="development",
            directory_pin=pin,
            source_provenance=source,
        )
        authorization = module._authorize_ledger_creation(
            invocation_seal=seal,
            stage="development",
            directory_pin=pin,
            source_provenance=source,
            bindings=bindings,
            ledger_path=pin.path / module.DEVELOPMENT_LEDGER_NAME,
        )
        ledger = module._AccessLedger(
            pin.path / module.DEVELOPMENT_LEDGER_NAME,
            stage="development",
            bindings=bindings,
            directory_pin=pin,
            authorization=authorization,
        )
        result = _complete_lightweight_formal_split_for_recovery_fixture(
            module,
            split="development",
            ledger=ledger,
            fail_metric=qualification.GATE_RULES[0].name,
        )
        assert result["passed"] is False
    finally:
        if module._PIN_REGISTRY.get(id(pin)) is pin:
            module._release_pinned_directory(pin)
        if ledger is not None:
            module._LEDGER_REGISTRY.pop(id(ledger), None)
            module._LEDGER_SLOT_REGISTRY.pop(("development", os.fspath(canonical)), None)
        module._RUN_AUTHORIZATION_REGISTRY.clear()
        module._INVOCATION_SEAL_REGISTRY.clear()
        _forget_synthetic_outer_authority_for_module(module, seeding)

    def forbidden_import(name: str, *args: Any, **kwargs: Any) -> Any:
        pytest.fail(f"completed-failed recovery imported {name}")

    monkeypatch.setattr(module.importlib, "import_module", forbidden_import)
    monkeypatch.setattr(
        module,
        "_activate_recovery_checkpoint_dependency",
        lambda: pytest.fail("completed-failed recovery activated Torch"),
    )
    monkeypatch.setattr(
        module,
        "_activate_runtime_dependencies",
        lambda: pytest.fail("completed-failed recovery activated runtime"),
    )
    recovering = _synthetic_outer_authority_for_module(
        module,
        stage="development",
        source=source,
        marker="completed-failed-recovery",
    )
    try:
        assert (
            module.run_development(
                source_provenance=source,
                runner_authority=recovering,
            )
            == 1
        )
        terminal_bytes = (canonical / module.DEVELOPMENT_LEDGER_NAME).read_bytes()
        terminal = module._parse_ledger_bytes(
            terminal_bytes,
            label="completed-failed terminal",
        )
        assert terminal["status"] == "complete_failed"
    finally:
        _forget_synthetic_outer_authority_for_module(module, recovering)
        module._LEDGER_RECOVERY_AUTHORIZATION_REGISTRY.clear()
        module._FORMAL_PIN_AUTHORITY_REGISTRY.clear()

    recognizing = _synthetic_outer_authority_for_module(
        module,
        stage="development",
        source=source,
        marker="completed-failed-recognition",
    )
    try:
        assert (
            module.run_development(
                source_provenance=source,
                runner_authority=recognizing,
            )
            == 1
        )
        assert (canonical / module.DEVELOPMENT_LEDGER_NAME).read_bytes() == terminal_bytes
    finally:
        _forget_synthetic_outer_authority_for_module(module, recognizing)
        module._LEDGER_RECOVERY_AUTHORIZATION_REGISTRY.clear()
        module._FORMAL_PIN_AUTHORITY_REGISTRY.clear()


def test_formal_startup_recovery_precedes_science_and_runner_config() -> None:
    qualification_source = (
        qualification.REPOSITORY_ROOT / "world_model/training/rgbd_known_action_qualification.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(qualification_source)
    functions = {
        node.name: ast.get_source_segment(qualification_source, node) or ""
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    for name in ("run_development", "run_qualification"):
        body = functions[name]
        assert body.index("_validate_outer_runner_authority") < body.index(
            "_acquire_pinned_directory"
        )
        assert body.index("_acquire_pinned_directory") < body.index(
            "_validated_invocation_inventory"
        )
        assert body.index("_validated_invocation_inventory") < body.index(
            "_mint_ledger_recovery_authorization"
        )
        assert body.index("_recover_existing_formal_attempt") < body.index("_require_scene_freeze")
        assert body.index("_require_scene_freeze") < body.index("_activate_runtime_dependencies")
        assert body.index("_activate_runtime_dependencies") < body.index("require_frozen_config")
        assert body.index("require_frozen_config") < body.index(
            "evaluator = _FormalKnownActionEvaluator("
        )

    recovery_body = functions["_recover_existing_formal_attempt"]
    for forbidden in (
        "assert_known_action_config",
        "_FormalKnownActionEvaluator",
        "_formal_scene_module",
        "_accepted_orbital_owner",
        "_capability_nonce_hex",
        "_authorize_ledger_creation",
    ):
        assert forbidden not in recovery_body
    assert ".adopt_formal_recoverable(" in recovery_body

    runner_source = (
        qualification.REPOSITORY_ROOT / "scripts/run_rgbd_known_action_qualification.py"
    ).read_text(encoding="utf-8")
    runner_tree = ast.parse(runner_source)
    internal_main = next(
        node
        for node in runner_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_internal_main"
    )
    internal_source = ast.get_source_segment(runner_source, internal_main) or ""
    assert "_require_config(" not in internal_source
    assert '_import("torch")' not in internal_source


@pytest.mark.parametrize("stage", ["development", "qualification"])
def test_fresh_formal_path_reaches_freeze_then_runtime_activation(
    stage: qualification.Stage,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, module, _ = _lightweight_qualification_module()
    source = _fake_source()
    canonical = (tmp_path / f"fresh-freeze-{stage}").resolve()
    canonical.mkdir()
    if stage == "qualification":
        for name in module.DEVELOPMENT_ARTIFACT_NAMES:
            (canonical / name).write_bytes(b"opaque reviewed development artifact")
    monkeypatch.setattr(module, "_canonical_run_directory", lambda: canonical)
    outer = _synthetic_outer_authority_for_module(
        module,
        stage=stage,
        source=source,
        marker=f"fresh-{stage}",
        reviewed=(
            {
                "checkpoint_sha256": "a" * 64,
                "report_sha256": "b" * 64,
                "ledger_sha256": "c" * 64,
            }
            if stage == "qualification"
            else None
        ),
    )
    events: list[str] = []

    def observed_freeze() -> None:
        events.append("freeze")

    def stop_after_activation() -> None:
        events.append("activation")
        raise RuntimeError("synthetic stop after ordered formal activation")

    monkeypatch.setattr(module, "_require_scene_freeze", observed_freeze)
    monkeypatch.setattr(module, "_activate_runtime_dependencies", stop_after_activation)
    monkeypatch.setattr(module, "_persist_exception_report", lambda **_kwargs: None)
    kwargs: dict[str, Any] = {
        "source_provenance": source,
        "runner_authority": outer,
    }
    if stage == "qualification":
        kwargs.update(
            {
                "reviewed_checkpoint_sha256": "a" * 64,
                "reviewed_report_sha256": "b" * 64,
                "reviewed_development_ledger_sha256": "c" * 64,
            }
        )
    try:
        with pytest.raises(RuntimeError, match="ordered formal activation"):
            (
                module.run_development(**kwargs)
                if stage == "development"
                else module.run_qualification(**kwargs)
            )
        assert events == ["freeze", "activation"]
        assert module.torch is None
        assert module.OrpheusConfig is None
        assert module.load_config is None
    finally:
        _forget_synthetic_outer_authority_for_module(module, outer)
        module._FORMAL_PIN_AUTHORITY_REGISTRY.clear()


def _seed_lightweight_reserved_formal_development(
    module: Any,
    *,
    canonical: Path,
    source: dict[str, Any],
) -> None:
    first = _synthetic_outer_authority_for_module(
        module,
        stage="development",
        source=source,
        marker="seed-reserved",
    )
    pin = module._acquire_pinned_directory(
        canonical,
        create=True,
        canonical=True,
        outer_authority=first,
    )
    try:
        bindings = module._ledger_bindings(
            stage="development",
            execution_mode="formal",
            directory_pin=pin,
            source_provenance=source,
            formal_authorization=module._formal_authorization_binding(first),
        )
        record = {
            "schema": "rgbd_known_action_access_ledger_v2",
            "artifact_kind": "rgbd_known_action_exactly_once_access_ledger",
            "architecture_attempt": module.ARCHITECTURE_ATTEMPT,
            "stage": "development",
            "execution_mode": "formal",
            "order": ["development"],
            "bindings": bindings,
            "batch_size": module.BATCH_SIZE,
            "scenes_per_split": module.SCENES_PER_SPLIT,
            "splits": {"development": module._empty_split_state()},
            "attempt_reserved": True,
            "status": "reserved",
            "generation": 0,
            "publication": module._empty_publication_state(),
        }
        module._pinned_durable_create(
            pin,
            pin.path / module.DEVELOPMENT_LEDGER_NAME,
            module._ledger_bytes(record),
        )
    finally:
        module._release_pinned_directory(pin)
        _forget_synthetic_outer_authority_for_module(module, first)


def _complete_lightweight_formal_split_for_recovery_fixture(
    module: Any,
    *,
    ledger: Any,
    split: str,
    fail_metric: str | None = None,
) -> dict[str, Any]:
    manifest = module._ManifestCapability(split=split, ledger=ledger)
    evaluator = _FakeEvaluator()
    evaluations: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    for start in range(0, module.SCENES_PER_SPLIT, module.BATCH_SIZE):
        ordinals = tuple(range(start, start + module.BATCH_SIZE))
        batch = manifest.begin_batch(ordinals)
        request = module.PublicBatchEvaluationRequest(
            split=split,
            ordinals=ordinals,
            tokens=batch.tokens,
            _manifest=manifest,
            _batch=batch,
        )
        evaluation = evaluator.evaluate_public_batch(request)
        if start == 0:
            with pytest.raises(
                PermissionError,
                match="formal materializer evaluation binding differs",
            ):
                manifest.complete_batch(batch, evaluation)
        validated = module._validated_public_batch_evaluation(
            evaluation,
            split=split,
            ordinals=ordinals,
        )
        body = {
            "schema": "rgbd_known_action_public_batch_receipt_v2",
            "phase": "public",
            "split": split,
            "ordinals": list(ordinals),
            "batch_index": start // module.BATCH_SIZE,
            "bundle_count": module.BATCH_SIZE,
            "candidate_count": module.BATCH_SIZE * module.CANDIDATE_COUNT,
            "public_evaluation": module.copy.deepcopy(validated),
            "public_evidence": module.copy.deepcopy(validated["public_evidence"]),
            "public_evidence_sha256": validated["public_evidence_sha256"],
            "public_metrics": module.copy.deepcopy(validated["public_metrics"]),
            "public_metrics_sha256": module.canonical_sha256(validated["public_metrics"]),
            "public_resources": module.copy.deepcopy(validated["public_resources"]),
            "public_resources_sha256": module.canonical_sha256(validated["public_resources"]),
            "public_result_sha256": module.canonical_sha256(validated),
        }
        receipt = {**body, "receipt_sha256": module.canonical_sha256(body)}
        ledger.commit_batch(split, receipt)
        batch_registration = module._BATCH_REGISTRY.pop(id(batch))
        token_registrations = tuple(module._TOKEN_REGISTRY.pop(id(token)) for token in batch.tokens)
        assert batch_registration.batch is batch
        assert all(
            registration == (token, batch_registration, "consumed")
            for token, registration in zip(
                batch.tokens,
                token_registrations,
                strict=True,
            )
        )
        batch_registration.status = "retired"
        manifest._next_ordinal += module.BATCH_SIZE
        manifest._active = None
        evaluations.append(validated)
        receipts.append(
            module._validate_batch_receipt(
                receipt,
                split=split,
                batch_index=start // module.BATCH_SIZE,
            )
        )
    finalization = evaluator.finalize_public_split(split, tuple(evaluations))
    validated_finalization = module._validated_public_split_finalization(
        finalization,
        split=split,
        batch_results=evaluations,
    )
    summary = module._public_split_summary(
        split=split,
        receipts=receipts,
        finalization=validated_finalization,
    )
    seal = module._public_split_seal(summary, split=split)
    # This lightweight recovery fixture has no sensor tensors, so it binds a
    # structurally valid synthetic formal receipt directly at the ledger cut.
    # The materializer-backed production close is exercised independently.
    ledger.complete_public_split(split, summary, seal)
    request = manifest.begin_private_scoring(public_seal=seal)
    truth_request = request.consume()
    private_receipt = _synthetic_formal_private_receipt(
        module,
        split=split,
        public_seal_sha256=truth_request["public_seal_sha256"],
        truth_request=truth_request,
        fail_metric=fail_metric,
    )
    result = module._split_result_from_private_receipt(
        split=split,
        state=ledger.record["splits"][split],
        private_receipt=private_receipt,
    )
    validated_receipt = module._validated_private_scoring_receipt(
        private_receipt,
        split=split,
        public_seal_sha256=truth_request["public_seal_sha256"],
    )
    validated_result = module._validated_split_result(result, split=split)
    truth_registration = module._TRUTH_AUTHORITY_REGISTRY[id(request._authority)]
    module._TRUTH_AUTHORITY_REGISTRY[id(request._authority)] = (
        request._authority,
        manifest,
        request,
        "receipt_bound",
        module.canonical_sha256(validated_receipt),
    )
    assert truth_registration[3] == "consumed"
    ledger.complete_split(
        split,
        validated_result,
        authority=request._authority,
        receipt=validated_receipt,
    )
    manifest._closed = True
    return validated_result


def _seed_lightweight_terminal_formal_development(
    module: Any,
    *,
    canonical: Path,
    source: dict[str, Any],
) -> dict[str, str]:
    outer = _synthetic_outer_authority_for_module(
        module,
        stage="development",
        source=source,
        marker="seed-terminal-development",
    )
    pin = module._acquire_pinned_directory(
        canonical,
        create=True,
        canonical=True,
        outer_authority=outer,
    )
    ledger: Any = None
    try:
        formal_authorization = module._formal_authorization_binding(outer)
        bindings = module._ledger_bindings(
            stage="development",
            execution_mode="formal",
            directory_pin=pin,
            source_provenance=source,
            formal_authorization=formal_authorization,
        )
        seal = module._mint_runner_invocation_seal_from_outer(
            outer,
            stage="development",
            directory_pin=pin,
            source_provenance=source,
        )
        authorization = module._authorize_ledger_creation(
            invocation_seal=seal,
            stage="development",
            directory_pin=pin,
            source_provenance=source,
            bindings=bindings,
            ledger_path=pin.path / module.DEVELOPMENT_LEDGER_NAME,
        )
        ledger = module._AccessLedger(
            pin.path / module.DEVELOPMENT_LEDGER_NAME,
            stage="development",
            bindings=bindings,
            directory_pin=pin,
            authorization=authorization,
        )
        result = _complete_lightweight_formal_split_for_recovery_fixture(
            module,
            ledger=ledger,
            split="development",
        )
        module.torch = qualification.torch
        checkpoint = module._save_review_checkpoint(
            pin,
            development_result=result,
            source_provenance=source,
            execution_mode="formal",
            formal_authorization=formal_authorization,
        )
        template = module._report_root(
            stage="development",
            execution_mode="formal",
            source_provenance=source,
            results=[result],
            pending_ledger_sha256=None,
            checkpoint=checkpoint,
            formal_authorization=formal_authorization,
        )
        ledger.prepare_publication(template)
        ledger.publish_prepared_report()
        ledger.finish()
        paths = module._artifact_paths(pin)
        return {
            "checkpoint_sha256": module.sha256_bytes(
                module._pinned_read_bytes(
                    pin,
                    paths["checkpoint"],
                    label="seed reviewed checkpoint",
                    maximum=module.MAX_CHECKPOINT_BYTES,
                )
            ),
            "report_sha256": module.sha256_bytes(
                module._pinned_read_bytes(
                    pin,
                    paths["development_report"],
                    label="seed reviewed report",
                )
            ),
            "ledger_sha256": module.sha256_bytes(
                module._pinned_read_bytes(
                    pin,
                    paths["development_ledger"],
                    label="seed reviewed ledger",
                )
            ),
        }
    finally:
        module.torch = None
        if module._PIN_REGISTRY.get(id(pin)) is pin:
            module._release_pinned_directory(pin)
        if ledger is not None:
            module._LEDGER_REGISTRY.pop(id(ledger), None)
            module._LEDGER_SLOT_REGISTRY.pop(("development", os.fspath(canonical)), None)
        module._RUN_AUTHORIZATION_REGISTRY.clear()
        module._INVOCATION_SEAL_REGISTRY.clear()
        _forget_synthetic_outer_authority_for_module(module, outer)


def test_lightweight_qualification_completed_split_recovery_is_owner_and_torch_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, module, _ = _lightweight_qualification_module()
    source = _fake_source()
    canonical = (tmp_path / "lightweight-qualification-completed-split").resolve()
    monkeypatch.setattr(module, "_canonical_run_directory", lambda: canonical)
    reviewed = _seed_lightweight_terminal_formal_development(
        module,
        canonical=canonical,
        source=source,
    )
    seeding = _synthetic_outer_authority_for_module(
        module,
        stage="qualification",
        source=source,
        marker="qualification-completed-split-seed",
        reviewed=reviewed,
    )
    pin = module._acquire_pinned_directory(
        canonical,
        create=True,
        canonical=True,
        outer_authority=seeding,
    )
    ledger: Any = None
    try:
        formal_authorization = module._formal_authorization_binding(seeding)
        bindings = module._ledger_bindings(
            stage="qualification",
            execution_mode="formal",
            directory_pin=pin,
            source_provenance=source,
            reviewed_development=reviewed,
            formal_authorization=formal_authorization,
        )
        seal = module._mint_runner_invocation_seal_from_outer(
            seeding,
            stage="qualification",
            directory_pin=pin,
            source_provenance=source,
            reviewed_development=reviewed,
        )
        authorization = module._authorize_ledger_creation(
            invocation_seal=seal,
            stage="qualification",
            directory_pin=pin,
            source_provenance=source,
            bindings=bindings,
            ledger_path=pin.path / module.QUALIFICATION_LEDGER_NAME,
            reviewed_development=reviewed,
        )
        ledger = module._AccessLedger(
            pin.path / module.QUALIFICATION_LEDGER_NAME,
            stage="qualification",
            bindings=bindings,
            directory_pin=pin,
            authorization=authorization,
        )
        selector = _complete_lightweight_formal_split_for_recovery_fixture(
            module,
            ledger=ledger,
            split="selector",
        )
        assert selector["passed"] is True
        assert ledger.record["splits"]["selector"]["status"] == "passed"
        assert ledger.record["splits"]["confirmation"]["status"] == "unopened"
    finally:
        if module._PIN_REGISTRY.get(id(pin)) is pin:
            module._release_pinned_directory(pin)
        if ledger is not None:
            module._LEDGER_REGISTRY.pop(id(ledger), None)
            module._LEDGER_SLOT_REGISTRY.pop(("qualification", os.fspath(canonical)), None)
        module._RUN_AUTHORIZATION_REGISTRY.clear()
        module._INVOCATION_SEAL_REGISTRY.clear()
        _forget_synthetic_outer_authority_for_module(module, seeding)

    def forbidden_import(name: str, *args: Any, **kwargs: Any) -> Any:
        pytest.fail(f"completed-split qualification recovery imported {name}")

    monkeypatch.setattr(module.importlib, "import_module", forbidden_import)
    monkeypatch.setattr(
        module,
        "_activate_recovery_checkpoint_dependency",
        lambda: pytest.fail("interrupted qualification recovery activated Torch"),
    )
    monkeypatch.setattr(
        module,
        "_activate_runtime_dependencies",
        lambda: pytest.fail("interrupted qualification recovery activated runtime"),
    )
    recovering = _synthetic_outer_authority_for_module(
        module,
        stage="qualification",
        source=source,
        marker="qualification-completed-split-recovery",
        reviewed=reviewed,
    )
    try:
        assert (
            module.run_qualification(
                source_provenance=source,
                runner_authority=recovering,
                reviewed_checkpoint_sha256=reviewed["checkpoint_sha256"],
                reviewed_report_sha256=reviewed["report_sha256"],
                reviewed_development_ledger_sha256=reviewed["ledger_sha256"],
            )
            == 1
        )
        terminal = module._parse_ledger_bytes(
            (canonical / module.QUALIFICATION_LEDGER_NAME).read_bytes(),
            label="qualification completed-split terminal",
        )
        assert terminal["status"] == "terminal_error"
        assert terminal["splits"]["selector"]["status"] == "passed"
        assert terminal["splits"]["confirmation"]["status"] == "unopened"
        report = module._strict_json_loads(
            (canonical / module.QUALIFICATION_REPORT_NAME).read_bytes(),
            label="qualification completed-split recovery report",
        )
        assert report["error"] == {
            "type": "InterruptedRun",
            "message": (
                "recovered after a completed protected split without opening the next split"
            ),
        }
    finally:
        _forget_synthetic_outer_authority_for_module(module, recovering)
        module._LEDGER_RECOVERY_AUTHORIZATION_REGISTRY.clear()
        module._FORMAL_PIN_AUTHORITY_REGISTRY.clear()


def test_post_mint_report_mutation_blocks_terminal_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, module, _ = _lightweight_qualification_module()
    source = _fake_source()
    canonical = (tmp_path / "post-mint-preterminal-mutation").resolve()
    monkeypatch.setattr(module, "_canonical_run_directory", lambda: canonical)
    _seed_lightweight_reserved_formal_development(
        module,
        canonical=canonical,
        source=source,
    )
    outer = _synthetic_outer_authority_for_module(
        module,
        stage="development",
        source=source,
        marker="post-mint",
    )
    original = module._validated_recovery_preterminal_inventory

    def mutate_then_validate(
        authorization: Any,
        ledger: Any,
        *,
        report_identity: tuple[int, ...],
    ) -> dict[str, Any]:
        report_path = canonical / module.DEVELOPMENT_REPORT_NAME
        current = report_path.read_bytes()
        changed = bytes([current[0] ^ 1]) + current[1:]
        module._pinned_durable_replace(ledger._pin, report_path, changed)
        return original(
            authorization,
            ledger,
            report_identity=report_identity,
        )

    monkeypatch.setattr(
        module,
        "_validated_recovery_preterminal_inventory",
        mutate_then_validate,
    )
    monkeypatch.setattr(
        module,
        "_activate_runtime_dependencies",
        lambda: pytest.fail("recovery activated the fresh runtime"),
    )
    try:
        with pytest.raises(PermissionError, match="preterminal recovery inventory"):
            module.run_development(
                source_provenance=source,
                runner_authority=outer,
            )
        ledger = module._parse_ledger_bytes(
            (canonical / module.DEVELOPMENT_LEDGER_NAME).read_bytes(),
            label="post-mint mutation intent ledger",
        )
        assert ledger["publication"]["state"] == "error_intent"
        assert ledger["status"] == "publication_intent_error"
    finally:
        _forget_synthetic_outer_authority_for_module(module, outer)
        module._LEDGER_RECOVERY_AUTHORIZATION_REGISTRY.clear()
        module._FORMAL_PIN_AUTHORITY_REGISTRY.clear()
        module._LEDGER_REGISTRY.clear()
        module._LEDGER_SLOT_REGISTRY.clear()


def test_post_mint_same_inode_ledger_mutation_blocks_intent_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, module, _ = _lightweight_qualification_module()
    source = _fake_source()
    canonical = (tmp_path / "post-mint-intent-target-mutation").resolve()
    monkeypatch.setattr(module, "_canonical_run_directory", lambda: canonical)
    _seed_lightweight_reserved_formal_development(
        module,
        canonical=canonical,
        source=source,
    )
    outer = _synthetic_outer_authority_for_module(
        module,
        stage="development",
        source=source,
        marker="intent-target-mutation",
    )
    original_replace = module._pinned_durable_replace
    original_write = module._write_descriptor
    changed_bytes: bytes | None = None
    writer_calls = 0
    observed_rejection = False

    def count_staging_writes(descriptor: int, contents: bytes) -> Any:
        nonlocal writer_calls
        writer_calls += 1
        return original_write(descriptor, contents)

    def mutate_target_then_replace(
        pin: Any,
        path: Path,
        contents: bytes,
        **kwargs: Any,
    ) -> Any:
        nonlocal changed_bytes, observed_rejection
        registry_pins = list(module._LEDGER_REGISTRY.values())
        assert len(registry_pins) == 1
        registry_pin = registry_pins[0]
        ledger = registry_pin.ledger
        authorization = ledger._recovery_authorization
        registration = module._LEDGER_RECOVERY_AUTHORIZATION_REGISTRY[id(authorization)]
        assert registration[0] is authorization
        assert registration[2] == "receipt_minted"
        assert registration[4] == "intent_receipt_minted"
        assert registration[5] is None
        authorized_inventory = module._validated_recovery_inventory_binding(
            module._strict_json_loads(
                authorization.inventory_bytes,
                label="intent-race authorized inventory",
            )
        )
        predecessor_last_identity = ledger._last_identity
        predecessor_recovery_identity = ledger._recovery_ledger_identity
        assert predecessor_recovery_identity == authorization.ledger_identity
        assert len(predecessor_recovery_identity) == 7
        assert all(type(value) is int for value in predecessor_recovery_identity)
        assert set(kwargs) == {
            "_return_full_identity",
            "_recovery_expected_target_identity",
            "_recovery_expected_inventory",
        }
        assert kwargs["_return_full_identity"] is True
        assert kwargs["_recovery_expected_target_identity"] == authorization.ledger_identity
        assert kwargs["_recovery_expected_inventory"] == authorized_inventory
        target_rows = [
            row
            for row in authorized_inventory["entries"]
            if row["name"] == authorization.ledger_name
        ]
        assert target_rows == [
            {
                "name": authorization.ledger_name,
                "identity": list(authorization.ledger_identity),
            }
        ]
        before = path.stat()
        current = path.read_bytes()
        changed_bytes = current.replace(b'"generation": 0', b'"generation": 1', 1)
        assert len(changed_bytes) == len(current) and changed_bytes != current
        with path.open("r+b") as handle:
            handle.write(changed_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        after = path.stat()
        assert (after.st_ino, after.st_size, after.st_mtime_ns) == (
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        assert after.st_ctime_ns != before.st_ctime_ns
        try:
            return original_replace(pin, path, contents, **kwargs)
        except PermissionError:
            observed_rejection = True
            assert writer_calls == 0
            assert ledger._last_identity == predecessor_last_identity
            assert ledger._recovery_ledger_identity == predecessor_recovery_identity
            assert module._LEDGER_REGISTRY[id(ledger)] is registry_pin
            assert module._LEDGER_REGISTRY[id(ledger)].file_identity == (predecessor_last_identity)
            raise

    monkeypatch.setattr(module, "_pinned_durable_replace", mutate_target_then_replace)
    monkeypatch.setattr(module, "_write_descriptor", count_staging_writes)
    try:
        with pytest.raises(PermissionError, match="target changed before staging"):
            module.run_development(
                source_provenance=source,
                runner_authority=outer,
            )
        assert changed_bytes is not None
        assert observed_rejection is True
        assert writer_calls == 0
        assert (canonical / module.DEVELOPMENT_LEDGER_NAME).read_bytes() == changed_bytes
        assert not (canonical / f".{module.DEVELOPMENT_LEDGER_NAME}.tmp").exists()
    finally:
        _forget_synthetic_outer_authority_for_module(module, outer)
        module._LEDGER_RECOVERY_AUTHORIZATION_REGISTRY.clear()
        module._FORMAL_PIN_AUTHORITY_REGISTRY.clear()
        module._LEDGER_REGISTRY.clear()
        module._LEDGER_SLOT_REGISTRY.clear()


def test_preterminal_os_replace_wrapper_mutation_cannot_replace_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, module, _ = _lightweight_qualification_module()
    source = _fake_source()
    canonical = (tmp_path / "preterminal-os-replace-mutation").resolve()
    monkeypatch.setattr(module, "_canonical_run_directory", lambda: canonical)
    _seed_lightweight_reserved_formal_development(
        module,
        canonical=canonical,
        source=source,
    )
    outer = _synthetic_outer_authority_for_module(
        module,
        stage="development",
        source=source,
        marker="preterminal-os-replace-mutation",
    )
    original_write = module._write_descriptor
    replace_defaults = module._pinned_durable_replace.__kwdefaults__
    assert type(replace_defaults) is dict
    captured_replace = replace_defaults["_replace"]
    replace_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    terminal_staged = False

    def spy_captured_replace(*args: Any, **kwargs: Any) -> None:
        replace_calls.append((args, kwargs))
        captured_replace(*args, **kwargs)

    def mutate_report_after_terminal_staging(
        descriptor: int,
        contents: bytes,
    ) -> Any:
        nonlocal terminal_staged
        written = original_write(descriptor, contents)
        parsed: dict[str, Any] | None = None
        with suppress(PermissionError, TypeError, ValueError):
            parsed = module._parse_ledger_bytes(
                contents,
                label="terminal-race staged candidate",
            )
        if (
            parsed is not None
            and parsed["status"] == "terminal_error"
            and parsed["publication"]["state"] == "error_bound"
        ):
            assert terminal_staged is False
            terminal_staged = True
            assert written.st_size == len(contents)
            report_path = canonical / module.DEVELOPMENT_REPORT_NAME
            current = report_path.read_bytes()
            changed = bytes([current[0] ^ 1]) + current[1:]
            with report_path.open("r+b") as handle:
                handle.write(changed)
                handle.flush()
                os.fsync(handle.fileno())
        return written

    monkeypatch.setattr(module, "_write_descriptor", mutate_report_after_terminal_staging)
    replace_defaults["_replace"] = spy_captured_replace
    try:
        with pytest.raises(PermissionError, match="inventory changed before replacement"):
            module.run_development(
                source_provenance=source,
                runner_authority=outer,
            )
        ledger = module._parse_ledger_bytes(
            (canonical / module.DEVELOPMENT_LEDGER_NAME).read_bytes(),
            label="preterminal wrapper intent ledger",
        )
        assert ledger["publication"]["state"] == "error_intent"
        assert terminal_staged is True
        assert len(replace_calls) == 1
        replace_args, replace_kwargs = replace_calls[0]
        assert replace_args == (
            f".{module.DEVELOPMENT_LEDGER_NAME}.tmp",
            module.DEVELOPMENT_LEDGER_NAME,
        )
        assert replace_kwargs["src_dir_fd"] == replace_kwargs["dst_dir_fd"]
        assert not (canonical / f".{module.DEVELOPMENT_LEDGER_NAME}.tmp").exists()
    finally:
        replace_defaults["_replace"] = captured_replace
        _forget_synthetic_outer_authority_for_module(module, outer)
        module._LEDGER_RECOVERY_AUTHORIZATION_REGISTRY.clear()
        module._FORMAL_PIN_AUTHORITY_REGISTRY.clear()
        module._LEDGER_REGISTRY.clear()
        module._LEDGER_SLOT_REGISTRY.clear()


def test_successful_recovery_terminal_transition_uses_returned_identity_without_base_reread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, module, _ = _lightweight_qualification_module()
    source = _fake_source()
    canonical = (tmp_path / "successful-terminal-transition").resolve()
    monkeypatch.setattr(module, "_canonical_run_directory", lambda: canonical)
    _seed_lightweight_reserved_formal_development(
        module,
        canonical=canonical,
        source=source,
    )
    outer = _synthetic_outer_authority_for_module(
        module,
        stage="development",
        source=source,
        marker="successful-terminal-transition",
    )
    original_durable_replace = module._pinned_durable_replace
    original_publish = module._AccessLedger._publish_prevalidated_recovery
    original_read = module._pinned_read_bytes
    replace_defaults = original_durable_replace.__kwdefaults__
    assert type(replace_defaults) is dict
    captured_replace = replace_defaults["_replace"]
    inside_terminal_replace = False
    terminal_renamed = False
    publisher_observed = False
    terminal_result: tuple[tuple[int, ...], tuple[int, ...]] | None = None
    terminal_predecessor_inventory: dict[str, Any] | None = None
    primitive_phases: list[bool] = []
    forbidden_reads: list[str] = []

    def observed_captured_replace(*args: Any, **kwargs: Any) -> None:
        nonlocal terminal_renamed
        primitive_phases.append(inside_terminal_replace)
        captured_replace(*args, **kwargs)
        if inside_terminal_replace:
            terminal_renamed = True

    def reject_postterminal_base_reads(
        pin: Any,
        path: Path,
        *args: Any,
        **kwargs: Any,
    ) -> bytes:
        if terminal_renamed and path.name != module.DEVELOPMENT_LEDGER_NAME:
            forbidden_reads.append(path.name)
            pytest.fail(f"postterminal publication reread governed base artifact {path.name}")
        return original_read(pin, path, *args, **kwargs)

    def observe_terminal_helper(
        pin: Any,
        path: Path,
        contents: bytes,
        **kwargs: Any,
    ) -> Any:
        nonlocal inside_terminal_replace, terminal_result, terminal_predecessor_inventory
        if kwargs.get("_recovery_terminal") is not True:
            return original_durable_replace(pin, path, contents, **kwargs)
        registry_pins = list(module._LEDGER_REGISTRY.values())
        assert len(registry_pins) == 1
        predecessor_pin = registry_pins[0]
        ledger = predecessor_pin.ledger
        predecessor_last_identity = ledger._last_identity
        predecessor_recovery_identity = ledger._recovery_ledger_identity
        predecessor_record = ledger._record
        assert kwargs["_recovery_expected_target_identity"] == predecessor_recovery_identity
        terminal_predecessor_inventory = kwargs["_recovery_expected_inventory"]
        inside_terminal_replace = True
        try:
            result = original_durable_replace(pin, path, contents, **kwargs)
        finally:
            inside_terminal_replace = False
        assert terminal_renamed is True
        assert ledger._last_identity == predecessor_last_identity
        assert ledger._recovery_ledger_identity == predecessor_recovery_identity
        assert ledger._record is predecessor_record
        assert module._LEDGER_REGISTRY[id(ledger)] is predecessor_pin
        assert module._LEDGER_REGISTRY[id(ledger)].file_identity == predecessor_last_identity
        assert result[1] == module._file_identity(path.stat())
        terminal_result = result
        return result

    def observe_publisher(self: Any, *args: Any, **kwargs: Any) -> str:
        nonlocal publisher_observed, terminal_renamed
        authorization = kwargs["authorization"]
        predecessor_generation = self._record["generation"]
        try:
            result = original_publish(self, *args, **kwargs)
            assert terminal_renamed is True
            assert terminal_result is not None
            assert terminal_predecessor_inventory is not None
            reduced_identity, full_identity = terminal_result
            disk_identity = module._file_identity(self.path.stat())
            assert disk_identity == full_identity
            assert reduced_identity == (
                full_identity[0],
                full_identity[1],
                full_identity[3],
                full_identity[4],
                full_identity[6],
            )
            registry_pin = module._LEDGER_REGISTRY[id(self)]
            assert self._last_identity == reduced_identity
            assert self._recovery_ledger_identity == full_identity
            assert registry_pin.file_identity == reduced_identity
            assert registry_pin.terminal is True
            assert registry_pin.generation == predecessor_generation + 1
            expected_entries = [
                {
                    "name": row["name"],
                    "identity": (
                        list(full_identity)
                        if row["name"] == authorization.ledger_name
                        else list(row["identity"])
                    ),
                }
                for row in terminal_predecessor_inventory["entries"]
            ]
            expected_final_inventory = {
                "schema": "rgbd_known_action_ledger_recovery_inventory_v1",
                "entries": expected_entries,
                "entries_sha256": module.canonical_sha256(expected_entries),
            }
            actual_entries = [
                {
                    "name": artifact.name,
                    "identity": list(module._file_identity(artifact.stat())),
                }
                for artifact in sorted(canonical.iterdir(), key=lambda item: item.name)
            ]
            actual_inventory = {
                "schema": "rgbd_known_action_ledger_recovery_inventory_v1",
                "entries": actual_entries,
                "entries_sha256": module.canonical_sha256(actual_entries),
            }
            assert self._recovery_final_inventory == expected_final_inventory
            assert self._recovery_final_inventory == actual_inventory
            registration = module._LEDGER_RECOVERY_AUTHORIZATION_REGISTRY[id(authorization)]
            assert registration[0] is authorization
            assert registration[2] == "receipt_minted"
            assert registration[4] == "terminal_durable"
            assert registration[5] == module.canonical_sha256(actual_inventory)
            publisher_observed = True
            return result
        finally:
            terminal_renamed = False

    replace_defaults["_replace"] = observed_captured_replace
    monkeypatch.setattr(module, "_pinned_read_bytes", reject_postterminal_base_reads)
    monkeypatch.setattr(module, "_pinned_durable_replace", observe_terminal_helper)
    monkeypatch.setattr(
        module._AccessLedger,
        "_publish_prevalidated_recovery",
        observe_publisher,
    )
    try:
        assert (
            module.run_development(
                source_provenance=source,
                runner_authority=outer,
            )
            == 1
        )
        assert publisher_observed is True
        assert primitive_phases == [False, True]
        assert forbidden_reads == []
        assert terminal_renamed is False
    finally:
        replace_defaults["_replace"] = captured_replace
        _forget_synthetic_outer_authority_for_module(module, outer)
        module._LEDGER_RECOVERY_AUTHORIZATION_REGISTRY.clear()
        module._FORMAL_PIN_AUTHORITY_REGISTRY.clear()
        module._LEDGER_REGISTRY.clear()
        module._LEDGER_SLOT_REGISTRY.clear()


def test_true_last_step_replace_crash_is_recognized_without_rewrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, module, _ = _lightweight_qualification_module()
    source = _fake_source()
    canonical = (tmp_path / "true-last-step-crash").resolve()
    monkeypatch.setattr(module, "_canonical_run_directory", lambda: canonical)
    _seed_lightweight_reserved_formal_development(
        module,
        canonical=canonical,
        source=source,
    )
    crashing = _synthetic_outer_authority_for_module(
        module,
        stage="development",
        source=source,
        marker="true-last-step-crash",
    )
    replace_defaults = module._pinned_durable_replace.__kwdefaults__
    assert type(replace_defaults) is dict
    original_os_replace = replace_defaults["_replace"]
    original_read = module._pinned_read_bytes
    replace_calls = 0
    postterminal_sentinel = False
    forbidden_reads: list[str] = []

    def replace_then_crash(*args: Any, **kwargs: Any) -> None:
        nonlocal postterminal_sentinel, replace_calls
        replace_calls += 1
        original_os_replace(*args, **kwargs)
        if replace_calls == 2:
            postterminal_sentinel = True
            raise RuntimeError("injected post-rename crash")

    def reject_crash_path_base_reads(
        pin: Any,
        path: Path,
        *args: Any,
        **kwargs: Any,
    ) -> bytes:
        if postterminal_sentinel and path.name != module.DEVELOPMENT_LEDGER_NAME:
            forbidden_reads.append(path.name)
            pytest.fail(f"postterminal crash path reread governed base artifact {path.name}")
        return original_read(pin, path, *args, **kwargs)

    replace_defaults["_replace"] = replace_then_crash
    monkeypatch.setattr(module, "_pinned_read_bytes", reject_crash_path_base_reads)
    try:
        with pytest.raises(RuntimeError, match="post-rename crash"):
            module.run_development(
                source_provenance=source,
                runner_authority=crashing,
            )
        assert postterminal_sentinel is True
        assert forbidden_reads == []
    finally:
        replace_defaults["_replace"] = original_os_replace
        _forget_synthetic_outer_authority_for_module(module, crashing)
        postterminal_sentinel = False
    terminal_path = canonical / module.DEVELOPMENT_LEDGER_NAME
    terminal_bytes = terminal_path.read_bytes()
    terminal_identity = module._file_identity(terminal_path.stat())
    terminal = module._parse_ledger_bytes(terminal_bytes, label="post-rename crash terminal")
    assert terminal["status"] == "terminal_error"
    module._LEDGER_RECOVERY_AUTHORIZATION_REGISTRY.clear()
    module._FORMAL_PIN_AUTHORITY_REGISTRY.clear()
    module._LEDGER_REGISTRY.clear()
    module._LEDGER_SLOT_REGISTRY.clear()
    recognizing = _synthetic_outer_authority_for_module(
        module,
        stage="development",
        source=source,
        marker="post-crash-recognition",
    )
    try:
        assert (
            module.run_development(
                source_provenance=source,
                runner_authority=recognizing,
            )
            == 1
        )
        assert terminal_path.read_bytes() == terminal_bytes
        assert module._file_identity(terminal_path.stat()) == terminal_identity
    finally:
        _forget_synthetic_outer_authority_for_module(module, recognizing)
        module._LEDGER_RECOVERY_AUTHORIZATION_REGISTRY.clear()
        module._FORMAL_PIN_AUTHORITY_REGISTRY.clear()
        module._LEDGER_REGISTRY.clear()
        module._LEDGER_SLOT_REGISTRY.clear()


def test_read_only_terminal_recognition_ends_with_exact_inventory_recapture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, module, _ = _lightweight_qualification_module()
    source = _fake_source()
    canonical = (tmp_path / "read-only-terminal-recapture").resolve()
    monkeypatch.setattr(module, "_canonical_run_directory", lambda: canonical)
    _seed_lightweight_reserved_formal_development(
        module,
        canonical=canonical,
        source=source,
    )
    terminalizing = _synthetic_outer_authority_for_module(
        module,
        stage="development",
        source=source,
        marker="terminalizing",
    )
    try:
        assert (
            module.run_development(
                source_provenance=source,
                runner_authority=terminalizing,
            )
            == 1
        )
    finally:
        _forget_synthetic_outer_authority_for_module(module, terminalizing)
    before = {
        path.name: (path.read_bytes(), module._file_identity(path.stat()))
        for path in canonical.iterdir()
    }
    labels: list[str] = []
    original = module._validated_live_recovery_inventory

    def record_recapture(
        authorization: Any,
        *,
        expected_state: str,
        label: str,
    ) -> dict[str, Any]:
        labels.append(label)
        return original(
            authorization,
            expected_state=expected_state,
            label=label,
        )

    monkeypatch.setattr(module, "_validated_live_recovery_inventory", record_recapture)
    recognizing = _synthetic_outer_authority_for_module(
        module,
        stage="development",
        source=source,
        marker="recognizing",
    )
    try:
        assert (
            module.run_development(
                source_provenance=source,
                runner_authority=recognizing,
            )
            == 1
        )
        assert labels[-1] == "completed read-only terminal development"
        after = {
            path.name: (path.read_bytes(), module._file_identity(path.stat()))
            for path in canonical.iterdir()
        }
        assert after == before
        assert all(
            registration[2] == "revoked"
            for registration in module._LEDGER_RECOVERY_AUTHORIZATION_REGISTRY.values()
        )
    finally:
        _forget_synthetic_outer_authority_for_module(module, recognizing)
        module._LEDGER_RECOVERY_AUTHORIZATION_REGISTRY.clear()
        module._FORMAL_PIN_AUTHORITY_REGISTRY.clear()
        module._LEDGER_REGISTRY.clear()
        module._LEDGER_SLOT_REGISTRY.clear()


def test_lightweight_error_intent_recovery_is_project_and_torch_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, module, _ = _lightweight_qualification_module()
    source = _fake_source()
    canonical = (tmp_path / "lightweight-error-intent").resolve()
    monkeypatch.setattr(module, "_canonical_run_directory", lambda: canonical)
    _seed_lightweight_reserved_formal_development(
        module,
        canonical=canonical,
        source=source,
    )
    crashing = _synthetic_outer_authority_for_module(
        module,
        stage="development",
        source=source,
        marker="error-intent-crash",
    )
    original_replace = module._pinned_durable_replace
    replace_calls = 0

    def crash_terminal_replace(*args: Any, **kwargs: Any) -> Any:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise RuntimeError("injected terminal replacement crash")
        return original_replace(*args, **kwargs)

    monkeypatch.setattr(module, "_pinned_durable_replace", crash_terminal_replace)
    try:
        with pytest.raises(RuntimeError, match="terminal replacement crash"):
            module.run_development(
                source_provenance=source,
                runner_authority=crashing,
            )
        intent = module._parse_ledger_bytes(
            (canonical / module.DEVELOPMENT_LEDGER_NAME).read_bytes(),
            label="crashed error intent",
        )
        assert intent["publication"]["state"] == "error_intent"
        assert (canonical / module.DEVELOPMENT_REPORT_NAME).is_file()
    finally:
        _forget_synthetic_outer_authority_for_module(module, crashing)
    monkeypatch.setattr(module, "_pinned_durable_replace", original_replace)
    module._LEDGER_RECOVERY_AUTHORIZATION_REGISTRY.clear()
    module._FORMAL_PIN_AUTHORITY_REGISTRY.clear()
    module._LEDGER_REGISTRY.clear()
    module._LEDGER_SLOT_REGISTRY.clear()
    recovering = _synthetic_outer_authority_for_module(
        module,
        stage="development",
        source=source,
        marker="error-intent-recovery",
    )
    imports: list[str] = []

    def forbidden_import(name: str, *args: Any, **kwargs: Any) -> Any:
        imports.append(name)
        pytest.fail(f"error-intent recovery imported {name}")

    monkeypatch.setattr(module.importlib, "import_module", forbidden_import)
    monkeypatch.setattr(
        module,
        "_activate_recovery_checkpoint_dependency",
        lambda: pytest.fail("error-intent recovery activated Torch"),
    )
    monkeypatch.setattr(
        module,
        "_activate_runtime_dependencies",
        lambda: pytest.fail("error-intent recovery activated the fresh runtime"),
    )
    try:
        assert (
            module.run_development(
                source_provenance=source,
                runner_authority=recovering,
            )
            == 1
        )
        assert imports == []
        terminal = module._parse_ledger_bytes(
            (canonical / module.DEVELOPMENT_LEDGER_NAME).read_bytes(),
            label="recovered error intent",
        )
        assert terminal["status"] == "terminal_error"
        assert [receipt["action"] for receipt in terminal["publication"]["recovery_receipts"]] == [
            "interrupt_nonterminal",
            "recover_existing_error_intent",
        ]
    finally:
        _forget_synthetic_outer_authority_for_module(module, recovering)
        module._LEDGER_RECOVERY_AUTHORIZATION_REGISTRY.clear()
        module._FORMAL_PIN_AUTHORITY_REGISTRY.clear()
        module._LEDGER_REGISTRY.clear()
        module._LEDGER_SLOT_REGISTRY.clear()


@pytest.mark.parametrize("checkpoint_field", ["sha256", "bytes"])
def test_reviewed_trio_verifier_binds_report_checkpoint_bytes_and_length(
    tmp_path: Path,
    checkpoint_field: str,
) -> None:
    source = _fake_source()
    with _pinned_run(tmp_path) as pin:
        qualification._execute_development_for_tests(
            directory_pin=pin,
            evaluator=_FakeEvaluator(),
            source_provenance=source,
        )
        paths = qualification._artifact_paths(pin)
        artifacts = {
            "checkpoint": qualification._pinned_read_bytes(
                pin,
                paths["checkpoint"],
                label="crossbind checkpoint",
                maximum=qualification.MAX_CHECKPOINT_BYTES,
            ),
            "report": qualification._pinned_read_bytes(
                pin,
                paths["development_report"],
                label="crossbind development report",
            ),
            "ledger": qualification._pinned_read_bytes(
                pin,
                paths["development_ledger"],
                label="crossbind development ledger",
            ),
        }
        report = qualification._strict_json_loads(
            artifacts["report"],
            label="crossbind development report",
        )
        if checkpoint_field == "sha256":
            report["checkpoint"]["sha256"] = "c" * 64
        else:
            report["checkpoint"]["bytes"] += 1
        artifacts["report"] = qualification._report_bytes(report)
        reviewed = {
            f"{name}_sha256": qualification.sha256_bytes(contents)
            for name, contents in artifacts.items()
        }

        with pytest.raises(ValueError, match="report/checkpoint byte binding"):
            qualification._validate_reviewed_development_artifacts(
                directory_pin=pin,
                reviewed_development=reviewed,
                expected_source=source,
                expected_execution_mode="fake_test",
                artifact_bytes=artifacts,
            )


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [qualification._TRUSTED_GIT, *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=False,
    )


def _prepare_clean_source_repository(
    tmp_path: Path,
) -> tuple[Path, dict[str, str], str]:
    root = (tmp_path / "repository").resolve()
    remote = (tmp_path / "remote.git").resolve()
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.name", "Known Action Test")
    _git(root, "config", "user.email", "known-action@example.invalid")
    surface = {
        "qualification": "qualification.py",
        "runner": "runner.py",
        "qualification_test": "test_qualification.py",
    }
    for name, relative in surface.items():
        (root / relative).write_text(f"# {name}\n", encoding="utf-8")
    _git(root, "add", *surface.values())
    _git(root, "commit", "-m", "initial")
    _git(root, "branch", "-M", qualification._APPROVED_BRANCH)
    remote.mkdir()
    _git(remote, "init", "--bare")
    _git(root, "remote", "add", qualification._APPROVED_REMOTE_NAME, os.fspath(remote))
    _git(root, "push", "-u", qualification._APPROVED_REMOTE_NAME, qualification._APPROVED_BRANCH)
    _git(
        root,
        "remote",
        "set-url",
        qualification._APPROVED_REMOTE_NAME,
        qualification._APPROVED_REMOTE_URL,
    )
    commit = _git(root, "rev-parse", "HEAD").stdout.decode("ascii").strip()
    return root, surface, commit


def test_source_capture_binds_clean_upstream_remote_and_exact_git_blobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_agent_path = "/private/tmp/private-known-action-agent.sock"
    monkeypatch.setenv("SSH_AUTH_SOCK", private_agent_path)
    root = (tmp_path / "repository").resolve()
    remote = (tmp_path / "remote.git").resolve()
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.name", "Known Action Test")
    _git(root, "config", "user.email", "known-action@example.invalid")
    surface = {
        "qualification": "qualification.py",
        "runner": "runner.py",
        "qualification_test": "test_qualification.py",
    }
    for name, relative in surface.items():
        (root / relative).write_text(f"# {name}\n", encoding="utf-8")
    _git(root, "add", *surface.values())
    _git(root, "commit", "-m", "initial")
    _git(root, "branch", "-M", qualification._APPROVED_BRANCH)
    remote.mkdir()
    _git(remote, "init", "--bare")
    _git(
        root,
        "remote",
        "add",
        qualification._APPROVED_REMOTE_NAME,
        os.fspath(remote),
    )
    _git(root, "push", "-u", qualification._APPROVED_REMOTE_NAME, qualification._APPROVED_BRANCH)
    _git(
        root,
        "remote",
        "set-url",
        qualification._APPROVED_REMOTE_NAME,
        qualification._APPROVED_REMOTE_URL,
    )
    commit = _git(root, "rev-parse", "HEAD").stdout.decode("ascii").strip()
    calls: list[dict[str, Any]] = []

    def mocked_transport(
        command: list[str],
        *,
        cwd: str,
        env: dict[str, str],
        timeout: int,
        known_hosts: bytes,
    ) -> subprocess.CompletedProcess[bytes]:
        calls.append(
            {
                "command": list(command),
                "cwd": cwd,
                "env": dict(env),
                "timeout": timeout,
                "known_hosts": known_hosts,
            }
        )
        return subprocess.CompletedProcess(
            command,
            0,
            f"{commit}\t{qualification._APPROVED_BRANCH_MERGE_REF}\n".encode("ascii"),
            b"",
        )

    monkeypatch.setattr(qualification, "_remote_transport", mocked_transport)

    source = qualification.capture_published_source(root, surface_paths=surface)
    assert source["commit"] == source["upstream_commit"]
    assert source["ahead"] == source["behind"] == 0
    assert all(
        binding["blob_sha256"] == binding["worktree_sha256"]
        for binding in source["publication_surface_blobs"].values()
    )
    assert len(calls) == 1
    profile = qualification._remote_transport_profile()
    assert calls[0] == {
        "command": profile["argv"],
        "cwd": "/",
        "env": profile["environment"],
        "timeout": 20,
        "known_hosts": qualification._PINNED_GITHUB_HOST_KEY.encode("ascii"),
    }
    remote_publication = source["remote_publication"]
    assert remote_publication["advertised_commit"] == source["commit"]
    assert remote_publication["advertised_commit"] == source["upstream_commit"]
    assert remote_publication["transport_profile"] == profile
    serialized_remote = json.dumps(remote_publication, sort_keys=True)
    assert private_agent_path not in serialized_remote
    assert "SSH_AUTH_SOCK" not in remote_publication["transport_profile"]["environment"]
    assert "-oIdentityAgent=SSH_AUTH_SOCK" in serialized_remote

    (root / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="clean worktree"):
        qualification.capture_published_source(root, surface_paths=surface)
    (root / "dirty.txt").unlink()
    (root / "qualification.py").write_text("# ahead\n", encoding="utf-8")
    _git(root, "add", "qualification.py")
    _git(root, "commit", "-m", "ahead")
    with pytest.raises(PermissionError, match="upstream"):
        qualification.capture_published_source(root, surface_paths=surface)


@pytest.mark.parametrize(
    "mutation",
    [
        "pushurl",
        "instead_of",
        "push_instead_of",
        "include",
        "include_if",
        "multiple_url",
        "multiple_branch_remote",
    ],
)
def test_source_capture_rejects_git_config_escape_hatches_and_multiple_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    root, surface, _ = _prepare_clean_source_repository(tmp_path)
    if mutation == "pushurl":
        _git(root, "config", "--add", "remote.origin.pushurl", "/private/tmp/escape.git")
    elif mutation == "instead_of":
        _git(root, "config", "--add", "url.ssh://escape.invalid/.insteadOf", "git@github.com:")
    elif mutation == "push_instead_of":
        _git(
            root,
            "config",
            "--add",
            "url.ssh://escape.invalid/.pushInsteadOf",
            "git@github.com:",
        )
    elif mutation in {"include", "include_if"}:
        included = (tmp_path / "included.config").resolve()
        included.write_text("[core]\n\tcompression = 0\n", encoding="utf-8")
        key = "include.path" if mutation == "include" else "includeIf.gitdir:/private/tmp/.path"
        _git(root, "config", "--add", key, os.fspath(included))
    elif mutation == "multiple_url":
        _git(
            root,
            "config",
            "--add",
            "remote.origin.url",
            qualification._APPROVED_REMOTE_URL,
        )
    else:
        _git(
            root,
            "config",
            "--add",
            f"branch.{qualification._APPROVED_BRANCH}.remote",
            qualification._APPROVED_REMOTE_NAME,
        )

    def forbidden_transport(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("remote transport ran after invalid local config")

    monkeypatch.setattr(qualification, "_remote_transport", forbidden_transport)
    with pytest.raises(PermissionError, match="forbidden|multiple|approved singleton"):
        qualification.capture_published_source(root, surface_paths=surface)


def test_git_config_snapshot_binds_every_value_and_matches_runner_parity(
    tmp_path: Path,
) -> None:
    root, _, _ = _prepare_clean_source_repository(tmp_path)
    _git(root, "config", "--add", "custom.MultiValue", "zeta")
    _git(root, "config", "--add", "CUSTOM.multivalue", "alpha")

    guard = qualification._approved_git_config_snapshot(root)
    pairs = [pair for pair in guard["local_pairs"] if pair["key"] == "custom.multivalue"]
    assert pairs == [
        {"key": "custom.multivalue", "value": "alpha"},
        {"key": "custom.multivalue", "value": "zeta"},
    ]
    assert guard["local_pairs"] == sorted(
        guard["local_pairs"],
        key=lambda pair: (pair["key"], pair["value"]),
    )
    assert guard["effective_pairs"] == sorted(
        [
            *guard["local_pairs"],
            {"key": "core.fsmonitor", "value": "false"},
            {"key": "core.hookspath", "value": "/dev/null"},
            {"key": "core.untrackedcache", "value": "false"},
        ],
        key=lambda pair: (pair["key"], pair["value"]),
    )

    runner = _load_runner()
    runner.REPOSITORY_ROOT = root
    assert runner._approved_git_config_snapshot() == guard


@pytest.mark.parametrize(
    "worktree_contents",
    [
        '[remote "origin"]\n\tpushurl = /private/tmp/escape.git\n',
        '[url "ssh://escape.invalid/"]\n\tinsteadOf = git@github.com:\n',
        "[include]\n\tpath = /private/tmp/escape.config\n",
        '[remote "origin"]\n\turl = ssh://escape.invalid/world.model.git\n',
        (
            f'[branch "{qualification._APPROVED_BRANCH}"]\n'
            f"\tremote = {qualification._APPROVED_REMOTE_NAME}\n"
        ),
    ],
)
def test_source_capture_rejects_canonical_worktree_config_scope_before_remote_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    worktree_contents: str,
) -> None:
    root, surface, _ = _prepare_clean_source_repository(tmp_path)
    _git(root, "config", "Extensions.WorkTreeConfig", "true")
    (root / ".git" / "config.worktree").write_text(worktree_contents, encoding="utf-8")

    monkeypatch.setattr(
        qualification,
        "_remote_transport",
        lambda *_args, **_kwargs: pytest.fail("remote probe ran with worktree config enabled"),
    )
    with pytest.raises(PermissionError, match="per-worktree Git config.*absent"):
        qualification.capture_published_source(root, surface_paths=surface)


def test_source_capture_rejects_casefolded_worktree_extension_without_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, surface, _ = _prepare_clean_source_repository(tmp_path)
    _git(root, "config", "Extensions.WorkTreeConfig", "false")
    monkeypatch.setattr(
        qualification,
        "_remote_transport",
        lambda *_args, **_kwargs: pytest.fail("remote probe ran with worktree extension present"),
    )
    with pytest.raises(PermissionError, match="forbidden directive extensions.worktreeconfig"):
        qualification.capture_published_source(root, surface_paths=surface)


def test_source_capture_rejects_same_key_value_race_around_remote_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, surface, commit = _prepare_clean_source_repository(tmp_path)

    def racing_transport(
        command: list[str],
        **_kwargs: Any,
    ) -> subprocess.CompletedProcess[bytes]:
        _git(
            root,
            "config",
            "remote.origin.fetch",
            "+refs/heads/raced:refs/remotes/origin/raced",
        )
        return subprocess.CompletedProcess(
            command,
            0,
            f"{commit}\t{qualification._APPROVED_BRANCH_MERGE_REF}\n".encode("ascii"),
            b"",
        )

    monkeypatch.setattr(qualification, "_remote_transport", racing_transport)
    with pytest.raises(PermissionError, match="changed during exact Git capture"):
        qualification.capture_published_source(root, surface_paths=surface)


@pytest.mark.parametrize(
    "mutation",
    [
        "uppercase_key",
        "unsorted_pairs",
        "duplicate_approved",
        "wrong_approved_value",
        "pushurl",
        "url_rewrite",
        "include_if",
        "worktree_extension",
        "worktree_present",
    ],
)
def test_source_validator_rejects_rehashed_noncanonical_or_forbidden_config_guard(
    mutation: str,
) -> None:
    source = json.loads(json.dumps(_fake_source()))
    guard = source["remote_publication"]["config_guard"]
    approved_key = f"remote.{qualification._APPROVED_REMOTE_NAME}.url".casefold()
    if mutation == "uppercase_key":
        guard["local_pairs"][0]["key"] = guard["local_pairs"][0]["key"].upper()
    elif mutation == "unsorted_pairs":
        guard["local_pairs"] = list(reversed(guard["local_pairs"]))
    elif mutation == "duplicate_approved":
        duplicate = {"key": approved_key, "value": qualification._APPROVED_REMOTE_URL}
        guard["local_pairs"].append(duplicate)
        guard["effective_pairs"].append(dict(duplicate))
        guard["local_pairs"].sort(key=lambda pair: (pair["key"], pair["value"]))
        guard["effective_pairs"].sort(key=lambda pair: (pair["key"], pair["value"]))
    elif mutation == "wrong_approved_value":
        for scope in ("approved", "local_pairs", "effective_pairs"):
            for pair in guard[scope]:
                if pair["key"] == approved_key:
                    pair["value"] = "ssh://escape.invalid/world.model.git"
    elif mutation in {"pushurl", "url_rewrite", "include_if", "worktree_extension"}:
        forbidden = {
            "pushurl": {"key": "remote.origin.pushurl", "value": "/private/tmp/escape"},
            "url_rewrite": {
                "key": "url.ssh://escape.invalid/.pushinsteadof",
                "value": "git@github.com:",
            },
            "include_if": {
                "key": "includeif.gitdir:/private/tmp/.path",
                "value": "/private/tmp/escape.config",
            },
            "worktree_extension": {
                "key": "extensions.worktreeconfig",
                "value": "false",
            },
        }[mutation]
        guard["local_pairs"].append(forbidden)
        guard["effective_pairs"].append(dict(forbidden))
        guard["local_pairs"].sort(key=lambda pair: (pair["key"], pair["value"]))
        guard["effective_pairs"].sort(key=lambda pair: (pair["key"], pair["value"]))
    else:
        guard["config_paths"]["worktree"]["state"] = "present"
    source["remote_publication"]["config_guard_sha256"] = qualification.canonical_sha256(guard)
    with pytest.raises((TypeError, ValueError), match="config|Git"):
        qualification._validated_source_provenance(source)


@pytest.mark.parametrize(
    ("stdout", "stderr", "returncode"),
    [
        (b"", b"", 1),
        (b"wrong\n", b"", 0),
        (
            f"{'a' * 40}\t{qualification._APPROVED_BRANCH_MERGE_REF}\n"
            f"{'a' * 40}\trefs/heads/other\n".encode("ascii"),
            b"",
            0,
        ),
        (
            f"{'A' * 40}\t{qualification._APPROVED_BRANCH_MERGE_REF}\n".encode("ascii"),
            b"",
            0,
        ),
        (
            f"{'a' * 40}\t{qualification._APPROVED_BRANCH_MERGE_REF}\0\n".encode("ascii"),
            b"",
            0,
        ),
        (b"x" * (qualification._REMOTE_PROBE_MAX_OUTPUT_BYTES + 1), b"", 0),
        (
            f"{'a' * 40}\t{qualification._APPROVED_BRANCH_MERGE_REF}^{{}}\n".encode("ascii"),
            b"",
            0,
        ),
        (
            f"{'a' * 40}\t{qualification._APPROVED_BRANCH_MERGE_REF}\n".encode("ascii"),
            b"x" * (qualification._REMOTE_PROBE_MAX_OUTPUT_BYTES + 1),
            0,
        ),
    ],
)
def test_remote_publication_probe_rejects_malformed_or_failed_advertisement(
    monkeypatch: pytest.MonkeyPatch,
    stdout: bytes,
    stderr: bytes,
    returncode: int,
) -> None:
    def mocked_transport(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(args[0], returncode, stdout, stderr)

    monkeypatch.setattr(qualification, "_remote_transport", mocked_transport)
    with pytest.raises(PermissionError, match="probe|advertised"):
        qualification._probe_remote_publication("a" * 40)


def test_remote_publication_probe_rejects_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def timed_out(command: list[str], **kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(qualification, "_remote_transport", timed_out)
    with pytest.raises(PermissionError, match="timed out"):
        qualification._probe_remote_publication("a" * 40)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_remote",
        "wrong_advertised_commit",
        "wrong_advertisement_sha",
        "wrong_literal_url",
        "missing_host_policy",
        "wrong_host_fingerprint",
        "float_transport_timeout",
        "wrong_config_guard_sha",
        "missing_config_guard",
    ],
)
def test_source_validator_rejects_missing_or_tampered_remote_publication(
    mutation: str,
) -> None:
    source = json.loads(json.dumps(_fake_source()))
    if mutation == "missing_remote":
        source.pop("remote_publication")
    else:
        remote = source["remote_publication"]
        if mutation == "wrong_advertised_commit":
            remote["advertised_commit"] = "b" * 40
        elif mutation == "wrong_advertisement_sha":
            remote["advertisement_sha256"] = "0" * 64
        elif mutation == "wrong_literal_url":
            remote["literal_url"] = "ssh://escape.invalid/world.model.git"
        elif mutation == "missing_host_policy":
            del remote["transport_profile"]["environment"]["GIT_SSH_COMMAND"]
        elif mutation == "wrong_host_fingerprint":
            remote["transport_profile"]["pinned_host_key_fingerprint"] = "SHA256:wrong"
        elif mutation == "float_transport_timeout":
            remote["transport_profile"]["timeout_seconds"] = 20.0
        elif mutation == "wrong_config_guard_sha":
            remote["config_guard_sha256"] = "0" * 64
        else:
            remote.pop("config_guard")
    with pytest.raises((TypeError, ValueError), match="source provenance|remote"):
        qualification._validated_source_provenance(source)


def test_pinned_github_ed25519_key_matches_documented_fingerprint() -> None:
    host, algorithm, encoded = qualification._PINNED_GITHUB_HOST_KEY.strip().split()
    assert (host, algorithm) == ("github.com", "ssh-ed25519")
    key_bytes = base64.b64decode(encoded, validate=True)
    fingerprint = base64.b64encode(hashlib.sha256(key_bytes).digest()).decode("ascii").rstrip("=")
    assert f"SHA256:{fingerprint}" == qualification._PINNED_GITHUB_HOST_KEY_FINGERPRINT


def test_remote_transport_profile_has_no_config_proxy_or_host_key_fallback() -> None:
    profile = qualification._remote_transport_profile()
    environment = profile["environment"]
    ssh = environment["GIT_SSH_COMMAND"]
    assert profile["credential_profile"] == "owned_stable_ssh_agent_socket_v1"
    assert profile["temporary_directory"] == "/private/tmp"
    assert environment["GIT_ALLOW_PROTOCOL"] == "ssh"
    assert environment["GIT_PROTOCOL_FROM_USER"] == "0"
    assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
    assert environment["GIT_CONFIG_GLOBAL"] == os.devnull
    assert environment["GIT_CONFIG_SYSTEM"] == os.devnull
    assert "SSH_AUTH_SOCK" not in environment
    for exact in (
        "-F /dev/null",
        "-oBatchMode=yes",
        "-oProxyCommand=none",
        "-oProxyJump=none",
        "-oStrictHostKeyChecking=yes",
        "-oIdentityFile=/dev/null",
        "-oIdentitiesOnly=no",
        "-oIdentityAgent=SSH_AUTH_SOCK",
        "-oAddKeysToAgent=no",
        "-oPKCS11Provider=none",
        "-oSecurityKeyProvider=none",
        "-oGSSAPIAuthentication=no",
        "-oHostbasedAuthentication=no",
        "-oHostKeyAlgorithms=ssh-ed25519",
        "-oHostKeyAlias=github.com",
        "-oUserKnownHostsFile={known_hosts_file}",
        "-oGlobalKnownHostsFile=/dev/null",
    ):
        assert exact in ssh
    assert profile["pinned_host_key_fingerprint"] == (
        qualification._PINNED_GITHUB_HOST_KEY_FINGERPRINT
    )


def test_ssh_agent_socket_rejects_absent_regular_and_symlink_authority(
    tmp_path: Path,
) -> None:
    with pytest.raises(PermissionError, match="absent or noncanonical"):
        qualification._validated_ssh_agent_socket(_environ={})
    regular = (tmp_path / "regular").resolve()
    regular.write_bytes(b"not a socket")
    with pytest.raises(PermissionError, match="owned live socket"):
        qualification._validated_ssh_agent_socket(_environ={"SSH_AUTH_SOCK": os.fspath(regular)})
    alias = (tmp_path / "agent-link.sock").resolve()
    alias.symlink_to(regular)
    with pytest.raises(PermissionError, match="owned live socket"):
        qualification._validated_ssh_agent_socket(_environ={"SSH_AUTH_SOCK": os.fspath(alias)})

    class SocketMetadata:
        st_mode = qualification.stat.S_IFSOCK | 0o600
        st_uid = qualification.os.geteuid()
        st_gid = 20
        st_nlink = 1
        st_dev = 7
        st_ino = 11

    assert qualification._validated_ssh_agent_socket(
        _environ={"SSH_AUTH_SOCK": "/private/tmp/fake-agent.sock"},
        _lstat=lambda _path: SocketMetadata(),
    ) == (
        "/private/tmp/fake-agent.sock",
        (7, 11, SocketMetadata.st_mode, qualification.os.geteuid(), 20, 1),
    )


def test_remote_transport_uses_pinned_host_file_and_stable_owned_agent() -> None:
    agent_path = "/private/tmp/fake-agent.sock"
    agent_identity = (7, 11, qualification.stat.S_IFSOCK | 0o600, os.geteuid(), 20, 1)
    profile = qualification._remote_transport_profile()
    observed: dict[str, Any] = {}

    def agent_authority() -> tuple[str, tuple[int, int, int, int, int, int]]:
        return agent_path, agent_identity

    def mocked_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        observed.update({"command": command, **kwargs})
        ssh = kwargs["env"]["GIT_SSH_COMMAND"]
        known_hosts_path = ssh.split("-oUserKnownHostsFile=", 1)[1].split(" ", 1)[0]
        known_hosts = Path(known_hosts_path)
        assert known_hosts.read_bytes() == qualification._PINNED_GITHUB_HOST_KEY.encode("ascii")
        assert known_hosts.stat().st_mode & 0o777 == 0o600
        return subprocess.CompletedProcess(command, 0, b"ok", b"")

    completed = qualification._remote_transport(
        profile["argv"],
        cwd=profile["cwd"],
        env=profile["environment"],
        timeout=profile["timeout_seconds"],
        known_hosts=qualification._PINNED_GITHUB_HOST_KEY.encode("ascii"),
        _run=mocked_run,
        _agent=agent_authority,
    )
    assert completed.returncode == 0
    assert observed["command"] == profile["argv"]
    assert observed["cwd"] == "/"
    assert observed["timeout"] == 20
    assert observed["env"]["SSH_AUTH_SOCK"] == agent_path
    assert set(observed["env"]) == {*profile["environment"], "SSH_AUTH_SOCK"}
    assert "{known_hosts_file}" not in observed["env"]["GIT_SSH_COMMAND"]


def test_remote_transport_rejects_any_environment_extension() -> None:
    profile = qualification._remote_transport_profile()
    extended = dict(profile["environment"])
    extended["HTTPS_PROXY"] = "sentinel-secret-value"

    with pytest.raises(PermissionError, match="environment differs from the fixed profile"):
        qualification._remote_transport(
            profile["argv"],
            cwd=profile["cwd"],
            env=extended,
            timeout=profile["timeout_seconds"],
            known_hosts=qualification._PINNED_GITHUB_HOST_KEY.encode("ascii"),
            _run=lambda *_args, **_kwargs: pytest.fail("extended environment reached Git"),
            _agent=lambda: pytest.fail("extended environment reached agent authority"),
        )


def test_remote_transport_captured_agent_ignores_post_load_global_rebinding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = qualification._remote_transport.__kwdefaults__["_agent"]
    assert captured is qualification._validated_ssh_agent_socket

    class SocketMetadata:
        st_mode = qualification.stat.S_IFSOCK | 0o600
        st_uid = qualification.os.geteuid()
        st_gid = 20
        st_nlink = 1
        st_dev = 7
        st_ino = 11

    monkeypatch.setitem(captured.__kwdefaults__, "_environ", {"SSH_AUTH_SOCK": "/x.sock"})
    monkeypatch.setitem(captured.__kwdefaults__, "_lstat", lambda _path: SocketMetadata())

    def forbidden_global(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("post-load global agent rebinding changed captured transport authority")

    monkeypatch.setattr(qualification, "_validated_ssh_agent_socket", forbidden_global)
    profile = qualification._remote_transport_profile()
    completed = qualification._remote_transport(
        profile["argv"],
        cwd=profile["cwd"],
        env=profile["environment"],
        timeout=profile["timeout_seconds"],
        known_hosts=qualification._PINNED_GITHUB_HOST_KEY.encode("ascii"),
        _run=lambda command, **_kwargs: subprocess.CompletedProcess(command, 0, b"", b""),
    )
    assert completed.returncode == 0


def test_remote_transport_rejects_agent_socket_replacement_during_probe() -> None:
    agent_path = "/private/tmp/fake-agent.sock"
    identities = iter(
        [
            (7, 11, qualification.stat.S_IFSOCK | 0o600, os.geteuid(), 20, 1),
            (7, 12, qualification.stat.S_IFSOCK | 0o600, os.geteuid(), 20, 1),
        ]
    )
    profile = qualification._remote_transport_profile()

    def agent_authority() -> tuple[str, tuple[int, int, int, int, int, int]]:
        return agent_path, next(identities)

    def replacing_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(command, 0, b"", b"")

    with pytest.raises(PermissionError, match="changed during"):
        qualification._remote_transport(
            profile["argv"],
            cwd=profile["cwd"],
            env=profile["environment"],
            timeout=profile["timeout_seconds"],
            known_hosts=qualification._PINNED_GITHUB_HOST_KEY.encode("ascii"),
            _run=replacing_run,
            _agent=agent_authority,
        )


def _load_runner() -> Any:
    path = qualification.REPOSITORY_ROOT / "scripts/run_rgbd_known_action_qualification.py"
    spec = importlib.util.spec_from_file_location("known_action_runner_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_local_git_and_child_exec_environments_exclude_hostile_ambient_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _, _ = _prepare_clean_source_repository(tmp_path)
    hostile_names = {
        "ALL_PROXY",
        "CUSTOM_SENTINEL",
        "DYLD_INSERT_LIBRARIES",
        "GIT_ASKPASS",
        "HOME",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "LD_PRELOAD",
        "NO_PROXY",
        "PYTHONHOME",
        "PYTHONPATH",
        "SSH_ASKPASS",
        "SSH_AUTH_SOCK",
        "TEMP",
        "TMP",
        "TMPDIR",
        "XDG_CONFIG_HOME",
        "XDG_RUNTIME_DIR",
    }
    sentinel = "sentinel-secret-value"
    for name in hostile_names:
        monkeypatch.setenv(name, sentinel)

    expected_git_keys = {
        "GIT_DIR",
        "GIT_COMMON_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_NO_REPLACE_OBJECTS",
        "GIT_CONFIG_NOSYSTEM",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_SYSTEM",
        "GIT_ATTR_NOSYSTEM",
        "GIT_LITERAL_PATHSPECS",
        "GIT_OPTIONAL_LOCKS",
        "GIT_TERMINAL_PROMPT",
        "LANG",
        "LC_ALL",
        "PATH",
    }
    local_environment = qualification._git_environment(root)
    assert set(local_environment) == expected_git_keys
    assert sentinel not in local_environment.values()
    assert hostile_names.isdisjoint(local_environment)

    runner = _load_runner()
    runner.REPOSITORY_ROOT = root
    runner_git_environment = runner._git_environment()
    assert runner_git_environment == local_environment

    safe_agent = "/private/tmp/validated-known-action-agent.sock"
    monkeypatch.setattr(
        runner,
        "_validated_ssh_agent_socket",
        lambda: (safe_agent, (1, 2, runner.stat.S_IFSOCK | 0o600, os.geteuid(), 20, 1)),
    )
    observed: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        observed.update({"command": command, **kwargs})
        return subprocess.CompletedProcess(command, 0, b"", b"")

    nonce = "a" * 64
    returncode = runner._outer_main(
        ["--phase", "development"],
        _preflight=lambda _argv: None,
        _arguments=lambda _argv: None,
        _capture=lambda _argv: (
            {"nonce": nonce},
            {"authorization": "fake"},
            b"# exact fake runner blob\n",
        ),
        _environ=dict(os.environ),
        _run=fake_run,
        _repository_root=root,
        _runner_path=root / "scripts/run_rgbd_known_action_qualification.py",
        _executable="/exact/orpheus/python",
    )
    assert returncode == 0
    assert observed["command"][0] == "/exact/orpheus/python"
    assert observed["env"] == {
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
        "SSH_AUTH_SOCK": safe_agent,
        runner._RECEIPT_FD_ENV: observed["env"][runner._RECEIPT_FD_ENV],
        runner._RECEIPT_SHA_ENV: observed["env"][runner._RECEIPT_SHA_ENV],
        runner._RECEIPT_NONCE_ENV: nonce,
    }
    assert sentinel not in observed["env"].values()
    assert (hostile_names - {"SSH_AUTH_SOCK"}).isdisjoint(observed["env"])


def test_runner_requires_exactly_three_hashes_and_has_no_eager_project_import() -> None:
    runner_path = qualification.REPOSITORY_ROOT / "scripts/run_rgbd_known_action_qualification.py"
    source = runner_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    eager_project_imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            eager_project_imports.extend(
                alias.name for alias in node.names if alias.name.startswith("world_model")
            )
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("world_model"):
            eager_project_imports.append(node.module)
    assert eager_project_imports == []
    assert "_pipe: Any = os.pipe" in source
    assert "_pipe()" in source
    assert '"-I"' in source
    assert "_ExactCommitLoader" in source
    assert "rgbd_known_action_scene" not in source

    runner = _load_runner()
    digest = "a" * 64
    with pytest.raises(SystemExit):
        runner.arguments(["--phase", "qualification"])
    with pytest.raises(SystemExit):
        runner.arguments(["--phase", "development", "--reviewed-checkpoint-sha256", digest])
    parsed = runner.arguments(
        [
            "--phase",
            "qualification",
            "--reviewed-checkpoint-sha256",
            digest,
            "--reviewed-report-sha256",
            digest,
            "--reviewed-development-ledger-sha256",
            digest,
        ]
    )
    assert parsed.phase == "qualification"


def test_formal_entrypoints_fail_before_canonical_run_directory_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical_accessed = False

    def forbidden() -> Path:
        nonlocal canonical_accessed
        canonical_accessed = True
        raise AssertionError("canonical run directory was opened")

    monkeypatch.setattr(qualification, "_canonical_run_directory", forbidden)
    with pytest.raises(PermissionError, match="outer-runner authority"):
        qualification.run_development()
    with pytest.raises(PermissionError, match="outer-runner authority"):
        qualification.run_qualification()
    assert canonical_accessed is False


def test_formal_evaluator_constructor_rejects_before_config_or_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accesses: list[str] = []

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        accesses.append("accessed")
        raise AssertionError("formal constructor crossed its ledger authority boundary")

    monkeypatch.setattr(qualification, "assert_known_action_config", forbidden)
    monkeypatch.setattr(qualification, "_formal_scene_module", forbidden)
    monkeypatch.setattr(qualification, "_accepted_orbital_owner", forbidden)
    with pytest.raises(PermissionError, match="constructor type or state differs"):
        qualification._FormalKnownActionEvaluator(object(), authority=object())
    forged = qualification._EvaluatorConstructionAuthority(
        stage="development",
        ledger_identity=1,
        directory_pin=object(),
        ledger_path=Path("/private/tmp/forged-ledger"),
        ledger_file_identity=(1, 2, 3, 4, 5, 6, 1),
        ledger_generation=0,
        ledger_record_sha256="a" * 64,
        config_identity=1,
        config_sha256="b" * 64,
        owner_thread=0,
        nonce=object(),
    )
    with pytest.raises(PermissionError, match="forged or replayed"):
        qualification._FormalKnownActionEvaluator(object(), authority=forged)
    assert accesses == []


def test_formal_scene_import_is_lazy_and_entrypoints_require_runtime_guards() -> None:
    path = qualification.REPOSITORY_ROOT / "world_model/training/rgbd_known_action_qualification.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert all("rgbd_known_action_scene" not in name for name in imports)
    assert "from world_model.training.rgbd_known_action_scene" not in source
    assert "import world_model.training.rgbd_known_action_scene" not in source
    top_level = {
        node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    for name in ("run_development", "run_qualification"):
        calls = {
            child.func.id
            for child in ast.walk(top_level[name])
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
        }
        assert "_require_scene_freeze" in calls
        assert "_activate_runtime_dependencies" in calls
        assert "_FormalKnownActionEvaluator" in calls
    evaluator_class = top_level["_FormalKnownActionEvaluator"]
    active_methods = {
        child.name: child
        for child in evaluator_class.body
        if isinstance(child, ast.FunctionDef)
        and child.name in {"evaluate_public_batch", "finalize_public_split", "score_private_split"}
    }
    assert set(active_methods) == {
        "evaluate_public_batch",
        "finalize_public_split",
        "score_private_split",
    }
    evaluate_names = {
        child.id
        for child in ast.walk(active_methods["evaluate_public_batch"])
        if isinstance(child, ast.Name)
    }
    finalize_names = {
        child.id
        for child in ast.walk(active_methods["finalize_public_split"])
        if isinstance(child, ast.Name)
    }
    private_names = {
        child.id
        for child in ast.walk(active_methods["score_private_split"])
        if isinstance(child, ast.Name)
    }
    assert "_issue_public_materialization_envelope" in evaluate_names
    assert "_finalize_materializer_public_split" in finalize_names
    assert "_bind_materializer_private_request" in private_names
    assert "_formal_private_scoring_receipt" in private_names


def test_formal_public_authority_precedes_science_and_binds_direct_evaluation() -> None:
    path = qualification.REPOSITORY_ROOT / "world_model/training/rgbd_known_action_qualification.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    evaluator = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "_FormalKnownActionEvaluator"
    )
    methods = {node.name: node for node in evaluator.body if isinstance(node, ast.FunctionDef)}
    public_source = ast.get_source_segment(source, methods["evaluate_public_batch"]) or ""
    science_source = ast.get_source_segment(source, methods["_evaluate_formal_public_batch"]) or ""
    private_source = ast.get_source_segment(source, methods["score_private_split"]) or ""
    assert public_source.index("_begin_formal_public_science_materialization(") < (
        public_source.index("self._evaluate_formal_public_batch(")
    )
    assert public_source.index("self._evaluate_formal_public_batch(") < public_source.index(
        "_issue_public_materialization_envelope("
    )
    public_calls = [
        node for node in ast.walk(methods["evaluate_public_batch"]) if isinstance(node, ast.Call)
    ]
    envelope_call = next(
        node
        for node in public_calls
        if isinstance(node.func, ast.Name)
        and node.func.id == "_issue_public_materialization_envelope"
    )
    assert {keyword.arg for keyword in envelope_call.keywords} >= {
        "preauthorized_token_receipts",
        "evaluation",
    }
    assert science_source.index("_construct_authorized_formal_batch(") < science_source.index(
        "scene.scene_metadata("
    )
    construction_call = next(
        node
        for node in ast.walk(methods["_evaluate_formal_public_batch"])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_construct_authorized_formal_batch"
    )
    authorities_keyword = next(
        keyword for keyword in construction_call.keywords if keyword.arg == "authorities"
    )
    assert isinstance(authorities_keyword.value, ast.Name)
    assert authorities_keyword.value.id == "authorities"
    assert "manual_prefix_trajectory" not in science_source
    assert "manual_candidate_trajectory" not in science_source
    assert "candidate_costs" not in science_source
    assert "self._truth_registry[" not in science_source
    private_binding = private_source.index("_bind_materializer_private_request(")
    assert private_binding < private_source.index("manual_prefix_trajectory(")
    assert private_binding < private_source.index("manual_candidate_trajectory(")
    assert private_binding < private_source.index("candidate_costs(")
    assert private_binding < private_source.index("self._truth_registry[truth_key]")


def test_formal_entrypoints_freeze_one_development_then_protected_early_stop() -> None:
    assert qualification.PROTECTED_SPLITS == (
        "selector",
        "confirmation",
        "final_test",
    )
    path = qualification.REPOSITORY_ROOT / "world_model/training/rgbd_known_action_qualification.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    development_calls = [
        node
        for node in ast.walk(functions["run_development"])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_evaluate_split"
    ]
    assert len(development_calls) == 1
    development_split = next(
        keyword for keyword in development_calls[0].keywords if keyword.arg == "split"
    )
    assert isinstance(development_split.value, ast.Constant)
    assert development_split.value.value == "development"

    protected_loops = [
        node
        for node in ast.walk(functions["run_qualification"])
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Name)
        and node.iter.id == "PROTECTED_SPLITS"
    ]
    assert len(protected_loops) == 1
    loop = protected_loops[0]
    assert isinstance(loop.target, ast.Name) and loop.target.id == "split"
    split_calls = [
        node
        for node in ast.walk(loop)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_evaluate_split"
    ]
    assert len(split_calls) == 1
    split_keyword = next(keyword for keyword in split_calls[0].keywords if keyword.arg == "split")
    assert isinstance(split_keyword.value, ast.Name)
    assert split_keyword.value.id == "split"
    assert any(
        isinstance(node, ast.Break)
        for branch in loop.body
        if isinstance(branch, ast.If)
        for node in ast.walk(branch)
    )


def test_materializer_constructor_chain_is_one_shot_lazy_and_hook_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accessed: list[str] = []

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        accessed.append("science hook")
        raise AssertionError("lazy materializer construction touched a science hook")

    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker="constructor-success",
    ) as (ledger, _pin, config):
        with monkeypatch.context() as scoped:
            scoped.setattr(qualification, "assert_known_action_config", forbidden)
            scoped.setattr(qualification, "_formal_scene_module", forbidden)
            scoped.setattr(qualification, "_accepted_orbital_owner", forbidden)
            scoped.setattr(qualification, "load_config", forbidden)
            scoped.setattr(qualification.importlib, "import_module", forbidden)
            authority, evaluator, port = _construct_synthetic_materializer(ledger, config)
            evaluator_registration = qualification._EVALUATOR_CONSTRUCTION_AUTHORITY_REGISTRY[
                id(authority)
            ]
            port_registration = qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY[id(port)]
            materializer_registration = qualification._MATERIALIZER_CONSTRUCTION_AUTHORITY_REGISTRY[
                id(port_registration.authority)
            ]
            assert evaluator_registration[3] == "consumed"
            assert evaluator_registration[4] == (
                qualification._evaluator_construction_authority_binding(authority)
            )
            assert materializer_registration.state == "consumed"
            assert port_registration.state == "active"
            assert port_registration.evaluator is evaluator
            assert port_registration.binding == qualification._trusted_materializer_port_binding(
                port
            )
            assert not hasattr(evaluator, "_scene")
            assert not hasattr(evaluator, "_owner")
            assert not hasattr(evaluator, "_accepted_config")
            for call in (
                lambda: evaluator.evaluate_public_batch(object()),
                lambda: evaluator.finalize_public_split("development", ()),
                lambda: evaluator.score_private_split(object()),
            ):
                with pytest.raises(PermissionError):
                    call()
            with pytest.raises(PermissionError, match="forged or replayed"):
                qualification._FormalKnownActionEvaluator(config, authority=authority)
            with pytest.raises(PermissionError, match="one fresh"):
                ledger.mint_formal_evaluator_authority(config)
            assert accessed == []
        qualification._close_trusted_materializer_port(port)
        assert port_registration.state == "closed"


def test_closed_materializer_port_cannot_gain_a_second_canonical_port(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker="constructor-closed-port-replay",
    ) as (ledger, _pin, config):
        _authority, evaluator, port = _construct_synthetic_materializer(ledger, config)
        original_registration = qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY[id(port)]
        construction_registration = qualification._MATERIALIZER_CONSTRUCTION_AUTHORITY_REGISTRY[
            id(original_registration.authority)
        ]
        qualification._close_trusted_materializer_port(port)
        replay = qualification._TrustedMaterializerPort(
            stage=port.stage,
            construction_authority_identity=port.construction_authority_identity,
            construction_authority_nonce=port.construction_authority_nonce,
            ledger_identity=port.ledger_identity,
            directory_pin=port.directory_pin,
            ledger_path=port.ledger_path,
            ledger_file_identity=port.ledger_file_identity,
            ledger_generation=port.ledger_generation,
            ledger_record_sha256=port.ledger_record_sha256,
            config_identity=port.config_identity,
            config_sha256=port.config_sha256,
            owner_thread=port.owner_thread,
            nonce=object(),
            _factory_nonce=qualification._MATERIALIZER_FACTORY_NONCE,
        )
        replay_registration = qualification._MaterializerPortRegistration(
            port=replay,
            authority=original_registration.authority,
            ledger=ledger,
            config=config,
            evaluator=evaluator,
            binding=qualification._trusted_materializer_port_binding(replay),
            state="active",
        )
        qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY[id(replay)] = replay_registration
        with pytest.raises(PermissionError, match="reverse bijection"):
            qualification._validated_materializer_global_registry_cut(
                ledger=ledger,
            )
        assert construction_registration.state == "failed"
        assert original_registration.state == "failed"
        assert replay_registration.state == "failed"
        assert qualification._MATERIALIZER_SPLIT_VAULT_REGISTRY == {}
        assert qualification._MATERIALIZER_SPLIT_VAULT_SLOT_REGISTRY == {}
        qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY.pop(id(replay), None)


@pytest.mark.parametrize(
    "latch_case",
    ("port_evaluator", "registry_none", "registry_missing"),
)
def test_formal_evaluator_reinitialization_cannot_orphan_first_ledger_vault(
    latch_case: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker="constructor-reinit-first",
        clear_after=False,
    ) as (first_ledger, _first_pin, first_config):
        _first_authority, evaluator, first_port = _construct_synthetic_materializer(
            first_ledger,
            first_config,
        )
        manifest = qualification._ManifestCapability(
            split="development",
            ledger=first_ledger,
        )
        vault = qualification._open_materializer_split_vault(
            first_port,
            split="development",
        )
        _request, envelope, _evaluation = _materializer_issue_public_batch(
            manifest=manifest,
            port=first_port,
            vault=vault,
            start=0,
        )
        first_attributes = (
            evaluator._ledger,
            evaluator._config,
            evaluator._materializer_port,
            evaluator._records,
            evaluator._manifest_rows,
            evaluator._truth_registry,
            evaluator._closed,
        )
        first_port_identities = tuple(qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY)
        port_registration = qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY[id(first_port)]
        if latch_case == "port_evaluator":
            original_latch = port_registration.evaluator
            port_registration.evaluator = object()

            def restore_latch() -> None:
                port_registration.evaluator = original_latch

        elif latch_case == "registry_none":
            original_latch = qualification._FORMAL_EVALUATOR_INITIALIZATION_REGISTRY[id(evaluator)]
            qualification._FORMAL_EVALUATOR_INITIALIZATION_REGISTRY[id(evaluator)] = None

            def restore_latch() -> None:
                qualification._FORMAL_EVALUATOR_INITIALIZATION_REGISTRY[id(evaluator)] = (
                    original_latch
                )

        else:
            original_latch = qualification._FORMAL_EVALUATOR_INITIALIZATION_REGISTRY[id(evaluator)]
            qualification._FORMAL_EVALUATOR_INITIALIZATION_REGISTRY.pop(id(evaluator))

            def restore_latch() -> None:
                qualification._FORMAL_EVALUATOR_INITIALIZATION_REGISTRY[id(evaluator)] = (
                    original_latch
                )

        with _formal_materializer_ledger(
            tmp_path,
            monkeypatch,
            marker="constructor-reinit-second",
            clear_after=False,
        ) as (second_ledger, _second_pin, second_config):
            second_authority = second_ledger.mint_formal_evaluator_authority(second_config)
            with pytest.raises(PermissionError, match="already initialized"):
                evaluator.__init__(second_config, authority=second_authority)
            assert (
                qualification._EVALUATOR_CONSTRUCTION_AUTHORITY_REGISTRY[id(second_authority)][3]
                == "failed"
            )
            assert tuple(qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY) == (
                first_port_identities
            )
            assert (
                evaluator._ledger,
                evaluator._config,
                evaluator._materializer_port,
                evaluator._records,
                evaluator._manifest_rows,
                evaluator._truth_registry,
                evaluator._closed,
            ) == first_attributes
            envelope_registration = qualification._PUBLIC_MATERIALIZATION_ENVELOPE_REGISTRY[
                id(envelope)
            ]
            assert envelope_registration.runtime_tensors is not None
            assert envelope_registration.blinding_nonces is not None
        evaluator.abort_split("development")
        _assert_materializer_vault_scrubbed(vault)
        manifest.abort()
        _assert_materializer_cleanup_refusal_then_restore(restore_latch)
    qualification._clear_ephemeral_registries_for_tests()


@pytest.mark.parametrize(
    "layer,field",
    (
        ("evaluator", "state"),
        ("evaluator", "binding"),
        ("materializer", "state"),
        ("materializer", "binding"),
    ),
)
def test_materializer_constructor_registry_tamper_is_hook_free_and_terminal(
    layer: str,
    field: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class EqualitySpy:
        def __eq__(self, _other: Any) -> bool:
            calls.append("eq")
            raise AssertionError("constructor registry equality hook ran")

        def __hash__(self) -> int:
            calls.append("hash")
            raise AssertionError("constructor registry hash hook ran")

    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker=f"constructor-registry-{layer}-{field}",
    ) as (ledger, _pin, config):
        evaluator_authority = ledger.mint_formal_evaluator_authority(config)
        if layer == "evaluator":
            registration = list(
                qualification._EVALUATOR_CONSTRUCTION_AUTHORITY_REGISTRY[id(evaluator_authority)]
            )
            if field == "state":
                registration[3] = EqualitySpy()
            else:
                registration[4] = (EqualitySpy(), *registration[4][1:])
            qualification._EVALUATOR_CONSTRUCTION_AUTHORITY_REGISTRY[id(evaluator_authority)] = (
                tuple(registration)
            )
            with pytest.raises(PermissionError):
                qualification._FormalKnownActionEvaluator(
                    config,
                    authority=evaluator_authority,
                )
            assert (
                qualification._EVALUATOR_CONSTRUCTION_AUTHORITY_REGISTRY[id(evaluator_authority)][3]
                == "failed"
            )
        else:
            evaluator = object.__new__(qualification._FormalKnownActionEvaluator)
            qualification._FORMAL_EVALUATOR_CONSTRUCTION_IN_PROGRESS[id(evaluator)] = (
                evaluator,
                evaluator_authority,
                None,
            )
            _same_ledger, materializer_authority = (
                qualification._consume_formal_evaluator_authority(
                    evaluator_authority,
                    config=config,
                    evaluator=evaluator,
                )
            )
            registration = qualification._MATERIALIZER_CONSTRUCTION_AUTHORITY_REGISTRY[
                id(materializer_authority)
            ]
            original_binding = registration.binding
            if field == "state":
                registration.state = EqualitySpy()  # type: ignore[assignment]
            else:
                registration.binding = (
                    EqualitySpy(),
                    *registration.binding[1:],
                )
            try:
                with pytest.raises(PermissionError):
                    qualification._construct_trusted_materializer_port(
                        materializer_authority,
                        config=config,
                        evaluator=evaluator,
                    )
                assert registration.state == "failed"
            finally:
                qualification._FORMAL_EVALUATOR_CONSTRUCTION_IN_PROGRESS.pop(
                    id(evaluator),
                    None,
                )
        assert calls == []
        assert qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY == {}
        assert qualification._FORMAL_EVALUATOR_CONSTRUCTION_IN_PROGRESS == {}
        assert not qualification._MATERIALIZER_CONSTRUCTION_IN_PROGRESS
        if layer == "materializer" and field == "binding":
            assert not qualification._exact_materializer_terminal_tombstone_cut()
            assert calls == []
            registration.binding = original_binding
            assert qualification._exact_materializer_terminal_tombstone_cut()


def test_formal_evaluator_constructor_rolls_back_port_if_postconstruction_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker="constructor-post-port-failure",
    ) as (ledger, _pin, config):
        authority = ledger.mint_formal_evaluator_authority(config)
        original_construct = qualification._construct_trusted_materializer_port
        captured: dict[str, Any] = {}

        def construct_then_fail(*args: Any, **kwargs: Any) -> Any:
            port = original_construct(*args, **kwargs)
            captured["port"] = port
            raise RuntimeError("injected post-port construction failure")

        monkeypatch.setattr(
            qualification,
            "_construct_trusted_materializer_port",
            construct_then_fail,
        )
        with pytest.raises(RuntimeError, match="post-port"):
            qualification._FormalKnownActionEvaluator(config, authority=authority)
        port = captured["port"]
        port_registration = qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY[id(port)]
        materializer_registration = qualification._MATERIALIZER_CONSTRUCTION_AUTHORITY_REGISTRY[
            id(port_registration.authority)
        ]
        assert port_registration.state == "failed"
        assert materializer_registration.state == "failed"
        assert (
            qualification._EVALUATOR_CONSTRUCTION_AUTHORITY_REGISTRY[id(authority)][3] == "failed"
        )
        assert qualification._FORMAL_EVALUATOR_CONSTRUCTION_IN_PROGRESS == {}
        assert not qualification._MATERIALIZER_CONSTRUCTION_IN_PROGRESS
        assert not any(
            registration.state == "active"
            for registration in qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY.values()
        )


def test_formal_evaluator_constructor_recovers_authority_if_consume_handoff_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker="constructor-consume-handoff-failure",
    ) as (ledger, _pin, config):
        authority = ledger.mint_formal_evaluator_authority(config)
        original_consume = qualification._consume_formal_evaluator_authority
        captured: dict[str, Any] = {}

        def consume_then_fail(*args: Any, **kwargs: Any) -> Any:
            same_ledger, materializer_authority = original_consume(*args, **kwargs)
            captured["ledger"] = same_ledger
            captured["authority"] = materializer_authority
            raise RuntimeError("injected consume handoff failure")

        monkeypatch.setattr(
            qualification,
            "_consume_formal_evaluator_authority",
            consume_then_fail,
        )
        with pytest.raises(RuntimeError, match="consume handoff"):
            qualification._FormalKnownActionEvaluator(config, authority=authority)
        materializer_authority = captured["authority"]
        assert captured["ledger"] is ledger
        assert (
            qualification._MATERIALIZER_CONSTRUCTION_AUTHORITY_REGISTRY[
                id(materializer_authority)
            ].state
            == "failed"
        )
        assert (
            qualification._EVALUATOR_CONSTRUCTION_AUTHORITY_REGISTRY[id(authority)][3] == "failed"
        )
        assert qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY == {}
        assert qualification._FORMAL_EVALUATOR_CONSTRUCTION_IN_PROGRESS == {}
        assert not qualification._MATERIALIZER_CONSTRUCTION_IN_PROGRESS


@pytest.mark.parametrize("case", ("short_tuple", "getitem_spy"))
def test_evaluator_authority_mint_rejects_malformed_registry_without_hooks(
    case: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class GetItemSpy:
        def __getitem__(self, _index: int) -> Any:
            calls.append("getitem")
            raise AssertionError("evaluator registry getitem hook ran")

    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker=f"evaluator-mint-malformed-{case}",
    ) as (ledger, _pin, config):
        qualification._EVALUATOR_CONSTRUCTION_AUTHORITY_REGISTRY[-1] = (  # type: ignore[assignment]
            () if case == "short_tuple" else GetItemSpy()
        )
        try:
            with pytest.raises(PermissionError, match="registry is malformed"):
                ledger.mint_formal_evaluator_authority(config)
            assert calls == []
            assert not any(
                type(registration) is tuple and len(registration) == 5 and registration[1] is ledger
                for identity, registration in (
                    qualification._EVALUATOR_CONSTRUCTION_AUTHORITY_REGISTRY.items()
                )
                if identity != -1
            )
        finally:
            qualification._EVALUATOR_CONSTRUCTION_AUTHORITY_REGISTRY.pop(-1, None)


@pytest.mark.parametrize(
    "field",
    (
        "stage",
        "ledger_identity",
        "directory_pin",
        "ledger_path",
        "ledger_file_identity",
        "ledger_generation",
        "ledger_record_sha256",
        "config_identity",
        "config_sha256",
        "owner_thread",
        "nonce",
    ),
)
def test_evaluator_construction_authority_tamper_is_fail_closed(
    field: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replacements: dict[str, Any] = {
        "stage": "qualification",
        "ledger_identity": -1,
        "directory_pin": object(),
        "ledger_path": Path("/private/tmp/rebound-materializer-ledger"),
        "ledger_file_identity": (-1, -1, -1, -1, -1, -1, -1),
        "ledger_generation": 1,
        "ledger_record_sha256": "f" * 64,
        "config_identity": -1,
        "config_sha256": "e" * 64,
        "owner_thread": -1,
        "nonce": object(),
    }
    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker=f"evaluator-tamper-{field}",
    ) as (ledger, _pin, config):
        authority = ledger.mint_formal_evaluator_authority(config)
        object.__setattr__(authority, field, replacements[field])
        with pytest.raises(PermissionError, match="rebound"):
            qualification._FormalKnownActionEvaluator(config, authority=authority)
        registration = qualification._EVALUATOR_CONSTRUCTION_AUTHORITY_REGISTRY[id(authority)]
        assert registration[3] == "failed"
        assert qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY == {}


@pytest.mark.parametrize(
    "field,replacement",
    (
        ("ledger_identity", -1),
        ("directory_pin", None),
        ("ledger_generation", 1),
        ("ledger_record_sha256", "d" * 64),
        ("config_sha256", "c" * 64),
        ("nonce", None),
        ("_factory_nonce", None),
    ),
)
def test_materializer_construction_authority_tamper_is_fail_closed(
    field: str,
    replacement: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker=f"materializer-tamper-{field}",
    ) as (ledger, _pin, config):
        evaluator_authority = ledger.mint_formal_evaluator_authority(config)
        evaluator = object.__new__(qualification._FormalKnownActionEvaluator)
        qualification._FORMAL_EVALUATOR_CONSTRUCTION_IN_PROGRESS[id(evaluator)] = (
            evaluator,
            evaluator_authority,
            None,
        )
        try:
            _same_ledger, authority = qualification._consume_formal_evaluator_authority(
                evaluator_authority,
                config=config,
                evaluator=evaluator,
            )
            original = getattr(authority, field)
            object.__setattr__(authority, field, replacement)
            with pytest.raises(PermissionError, match="rebound"):
                qualification._construct_trusted_materializer_port(
                    authority,
                    config=config,
                    evaluator=evaluator,
                )
            assert (
                qualification._MATERIALIZER_CONSTRUCTION_AUTHORITY_REGISTRY[id(authority)].state
                == "failed"
            )
            assert qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY == {}
        finally:
            qualification._FORMAL_EVALUATOR_CONSTRUCTION_IN_PROGRESS.pop(
                id(evaluator),
                None,
            )
        _assert_materializer_cleanup_refusal_then_restore(
            lambda: object.__setattr__(authority, field, original)
        )


def test_evaluator_construction_cross_thread_burns_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker="constructor-cross-thread",
    ) as (ledger, _pin, config):
        authority = ledger.mint_formal_evaluator_authority(config)
        errors: list[BaseException] = []

        def construct() -> None:
            try:
                qualification._FormalKnownActionEvaluator(config, authority=authority)
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=construct)
        thread.start()
        thread.join(timeout=5.0)
        assert not thread.is_alive()
        assert len(errors) == 1 and isinstance(errors[0], PermissionError)
        assert (
            qualification._EVALUATOR_CONSTRUCTION_AUTHORITY_REGISTRY[id(authority)][3] == "failed"
        )
        assert qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY == {}


def test_materializer_port_binding_tamper_fails_before_request_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker="port-tamper",
    ) as (ledger, _pin, config):
        _authority, evaluator, port = _construct_synthetic_materializer(ledger, config)
        original_nonce = port.nonce
        object.__setattr__(port, "nonce", object())
        with pytest.raises(PermissionError, match="binding differs"):
            evaluator.evaluate_public_batch(object())
        assert qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY[id(port)].state == "failed"
        _assert_materializer_cleanup_refusal_then_restore(
            lambda: object.__setattr__(port, "nonce", original_nonce)
        )


def test_materializer_tensor_clone_validation_and_disjoint_storage() -> None:
    sensors = _materializer_sensor_tensors()
    tasks = _materializer_safe_task_tensors()
    runtime = _materializer_public_runtime_tensors()
    qualification._validated_public_sensor_tensors(sensors)
    qualification._validated_safe_task_batch_tensors(tasks)
    qualification._validated_public_runtime_tensors(runtime)
    inventory = qualification._materializer_tensor_storage_inventory(
        {"runtime": runtime, "sensors": sensors, "tasks": tasks},
        label="test materializer tensors",
    )
    assert len(inventory) == (
        len(qualification.PUBLIC_SENSOR_TENSOR_KEYS)
        + len(qualification.SAFE_TASK_TENSOR_KEYS)
        + len(qualification.PUBLIC_RUNTIME_TENSOR_KEYS)
    )
    source = qualification.torch.arange(12, dtype=qualification.torch.float32).reshape(3, 4)
    source.requires_grad_(True)
    cloned = qualification._clone_materializer_tensor(source)
    assert qualification.torch.equal(source.detach(), cloned)
    assert cloned.device.type == "cpu"
    assert cloned.is_contiguous()
    assert cloned.requires_grad is False and cloned.grad_fn is None
    assert cloned.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
    with qualification.torch.no_grad():
        source.add_(1.0)
    assert not qualification.torch.equal(source.detach(), cloned)


@pytest.mark.parametrize(
    "case",
    ("rgb_range", "depth_nonfinite", "timestamp_order", "camera_row", "intrinsics"),
)
def test_public_sensor_tensor_validation_rejects_malformed_batches(case: str) -> None:
    tensors = _materializer_sensor_tensors()
    if case == "rgb_range":
        tensors["rgb"][0, 0, 0, 0, 0] = 1.1
    elif case == "depth_nonfinite":
        tensors["depth"][0, 0, 0, 0, 0] = float("inf")
    elif case == "timestamp_order":
        tensors["timestamps"][0, 1] = tensors["timestamps"][0, 0]
    elif case == "camera_row":
        tensors["world_from_camera"][0, 0, 3, 3] = 0.0
    else:
        tensors["intrinsics"][0, 0, 0, 0] = -1.0
    with pytest.raises(ValueError):
        qualification._validated_public_sensor_tensors(tensors)


@pytest.mark.parametrize(
    "surface,forbidden",
    (
        ("safe", "handle_role"),
        ("safe", "resolved_persistent_id"),
        ("safe", "goal_horizons"),
        ("safe", "canonical_q"),
        ("safe", "selected_display_index"),
        ("runtime", "canonical_q"),
        ("runtime", "truth_labels"),
        ("runtime", "goal_positions_world"),
    ),
)
def test_materializer_tensor_surfaces_have_exact_public_allowlists(
    surface: str,
    forbidden: str,
) -> None:
    if surface == "safe":
        tensors = _materializer_safe_task_tensors()
        tensors[forbidden] = qualification.torch.zeros(1)
        validator = qualification._validated_safe_task_batch_tensors
    else:
        tensors = _materializer_public_runtime_tensors()
        tensors[forbidden] = qualification.torch.zeros(1)
        validator = qualification._validated_public_runtime_tensors
    with pytest.raises(ValueError, match="allowlist"):
        validator(tensors)
    assert "resolved_persistent_id" not in qualification.SAFE_TASK_TENSOR_KEYS
    assert "persistent_id_by_slot" not in qualification.SAFE_TASK_TENSOR_KEYS
    assert "selected_display_index" in qualification.PUBLIC_RUNTIME_TENSOR_KEYS


@pytest.mark.parametrize("surface", ("sensor", "task", "runtime"))
def test_materializer_tensor_mapping_keys_are_type_gated_before_hooks(surface: str) -> None:
    calls: list[str] = []

    class KeySpy:
        armed = False

        def __hash__(self) -> int:
            if self.armed:
                calls.append("hash")
                raise AssertionError("tensor mapping key hash hook ran")
            return 0x515151

        def __eq__(self, _other: Any) -> bool:
            if self.armed:
                calls.append("eq")
                raise AssertionError("tensor mapping key equality hook ran")
            return self is _other

    if surface == "sensor":
        values = _materializer_sensor_tensors()
        validator = qualification._validated_public_sensor_tensors
    elif surface == "task":
        values = _materializer_safe_task_tensors()
        validator = qualification._validated_safe_task_batch_tensors
    else:
        values = _materializer_public_runtime_tensors()
        validator = qualification._validated_public_runtime_tensors
    key = KeySpy()
    values[key] = qualification.torch.zeros(1)  # type: ignore[index]
    key.armed = True
    with pytest.raises(ValueError, match="allowlist"):
        validator(values)
    assert calls == []


def test_materializer_task_horizon_and_runtime_alias_are_rejected() -> None:
    tasks = _materializer_safe_task_tensors()
    tasks["candidate_timestamps"][:, -1] = 2.01
    with pytest.raises(ValueError, match="within 2.0 seconds"):
        qualification._validated_safe_task_batch_tensors(tasks)
    runtime = _materializer_public_runtime_tensors()
    runtime["candidate_velocities"] = runtime["candidate_positions"]
    with pytest.raises(PermissionError, match="alias storage"):
        qualification._validated_public_runtime_tensors(runtime)
    assert qualification.MATERIALIZER_FIXED_GOAL_HORIZON_SECONDS == 2.0


def test_blinded_materialization_commitment_uses_exact_little_endian_frames() -> None:
    surface = _materializer_public_surface()
    nonce = bytes(range(32))
    body = qualification._canonical_json(surface)
    expected = hashlib.sha256(
        qualification.MATERIALIZER_BLINDED_COMMITMENT_DOMAIN
        + len(body).to_bytes(8, "little")
        + body
        + len(nonce).to_bytes(8, "little")
        + nonce
    ).hexdigest()
    actual = qualification._blinded_materialization_commitment(surface, nonce)
    assert actual == expected
    assert qualification._blinded_materialization_commitment(surface, nonce) == actual
    changed = _materializer_public_surface(ordinal=1)
    assert qualification._blinded_materialization_commitment(changed, nonce) != actual
    assert qualification._blinded_materialization_commitment(surface, b"x" * 32) != actual
    with pytest.raises(ValueError, match="exactly 32"):
        qualification._blinded_materialization_commitment(surface, b"x" * 31)
    malformed = dict(surface)
    malformed["sensor_sha256"] = "A" * 64
    with pytest.raises(ValueError, match="lowercase"):
        qualification._blinded_materialization_commitment(malformed, nonce)
    malformed = dict(surface)
    malformed["private_body"] = {"truth": True}
    with pytest.raises(ValueError, match="schema differs"):
        qualification._blinded_materialization_commitment(malformed, nonce)


def test_materializer_one_public_batch_binds_and_abort_scrubs_every_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker="one-batch-abort",
    ) as (ledger, _pin, config):
        _authority, evaluator, port = _construct_synthetic_materializer(ledger, config)
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        vault = qualification._open_materializer_split_vault(port, split="development")
        _request, envelope, hashes, receipt = _materializer_public_batch(
            manifest=manifest,
            port=port,
            vault=vault,
            start=0,
        )
        vault_registration = qualification._MATERIALIZER_SPLIT_VAULT_REGISTRY[id(vault)]
        envelope_registration = qualification._PUBLIC_MATERIALIZATION_ENVELOPE_REGISTRY[
            id(envelope)
        ]
        sensor_registration = qualification._PUBLIC_SENSOR_BATCH_REGISTRY[
            id(vault_registration.sensor_batches[0])
        ]
        task_registration = qualification._SAFE_TASK_BATCH_REGISTRY[
            id(vault_registration.task_batches[0])
        ]
        assert vault_registration.state == "collecting_public"
        assert vault_registration.next_ordinal == 4
        assert envelope_registration.state == "evaluated"
        assert sensor_registration.tensors is not None
        assert task_registration.tensors is not None
        assert envelope_registration.runtime_tensors is not None
        assert envelope_registration.public_bodies is not None
        assert envelope_registration.blinding_nonces is not None
        assert hashes["public_evaluation_sha256"] == receipt["public_result_sha256"]
        assert qualification._CONSUMED_ORDINAL_REGISTRY == {}
        evaluator.abort_split("development")
        manifest.abort()
        assert vault_registration.state == "revoked"
        assert qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY[id(port)].state == "failed"
        assert envelope_registration.state == "retired"
        assert envelope_registration.request is None
        assert envelope_registration.sensor_batch is None
        assert envelope_registration.task_batch is None
        assert envelope_registration.runtime_tensors is None
        assert envelope_registration.public_bodies is None
        assert envelope_registration.blinding_nonces is None
        assert sensor_registration.state == "revoked" and sensor_registration.tensors is None
        assert task_registration.state == "revoked" and task_registration.tensors is None
        assert all(
            qualification._SEALED_MATERIALIZATION_ROW_REGISTRY[id(row)].state == "revoked"
            for row in vault_registration.rows
        )
        assert qualification._BATCH_REGISTRY == {}
        assert qualification._TOKEN_REGISTRY == {}
        assert qualification._MATERIALIZER_PUBLIC_BATCH_EVALUATION_REGISTRY == {}


def test_materializer_frozen_sensor_action_bounds_and_tensor_byte_vectors() -> None:
    sensors = _materializer_sensor_tensors()
    tasks = _materializer_safe_task_tensors(start=16)
    qualification._validated_public_sensor_tensors(sensors)
    qualification._validated_safe_task_batch_tensors(tasks)
    qualification._validate_exact_materializer_safe_rows(
        tasks,
        split="development",
        ordinals=(16, 17, 18, 19),
    )
    assert qualification.MATERIALIZER_SENSOR_DEPTH_METERS == (0.0, 8.0)
    assert qualification.MATERIALIZER_CAMERA_INTRINSICS == (
        (71.87317657470703, 0.0, 31.5),
        (0.0, 71.87317657470703, 31.5),
        (0.0, 0.0, 1.0),
    )
    assert qualification.torch.equal(
        sensors["timestamps"][0],
        qualification.torch.arange(16, dtype=qualification.torch.float32) / 20.0,
    )
    assert qualification.torch.equal(
        tasks["candidate_timestamps"],
        qualification.torch.full((4, 8), 1.1, dtype=qualification.torch.float32),
    )
    expected_first_impulse = qualification.torch.tensor(
        (-0.00575, -0.005, -0.0065),
        dtype=qualification.torch.float32,
    )
    assert qualification.torch.equal(
        tasks["candidate_impulses_world"][0, 0],
        expected_first_impulse,
    )
    assert qualification.torch.equal(
        tasks["candidate_impulses_world"][0, 7],
        -expected_first_impulse,
    )

    class Recorder:
        def __init__(self) -> None:
            self.parts: list[bytes] = []

        def update(self, value: bytes) -> None:
            self.parts.append(value)

    float_recorder = Recorder()
    float_vector = qualification.torch.tensor(
        [1.0, -2.0],
        dtype=qualification.torch.float32,
    )
    qualification._tensor_digest(float_recorder, "float32-vector", float_vector)
    assert float_recorder.parts[-1] == bytes.fromhex("0000803f000000c0")
    assert hashlib.sha256(b"".join(float_recorder.parts)).hexdigest() == (
        "a823c5b0a3e340be42c889fd6159c51164489a82bdd56fccfa778987d52e166e"
    )
    assert float_recorder.parts[-1] != bytes.fromhex("3f800000c0000000")

    int_recorder = Recorder()
    int_vector = qualification.torch.tensor([1, -2], dtype=qualification.torch.int64)
    qualification._tensor_digest(int_recorder, "int64-vector", int_vector)
    assert int_recorder.parts[-1] == bytes.fromhex("0100000000000000feffffffffffffff")
    assert hashlib.sha256(b"".join(int_recorder.parts)).hexdigest() == (
        "4ca4ce7d0f3b2e899040730f3f44de5d04355d4a0fb556d5a50886197db3492f"
    )


@pytest.mark.parametrize(
    "case",
    ("depth_high", "timestamp", "camera_radius", "camera_step", "shear"),
)
def test_materializer_frozen_sensor_bounds_reject_nearby_malformed_values(
    case: str,
) -> None:
    sensors = _materializer_sensor_tensors()
    if case == "depth_high":
        sensors["depth"][0, 0, 0, 0, 0] = 8.0001
    elif case == "timestamp":
        sensors["timestamps"][0, 1] += 1.0e-4
    elif case == "camera_radius":
        sensors["world_from_camera"][0, 0, 0, 3] += 1.0e-3
    elif case == "camera_step":
        sensors["world_from_camera"][0, 1] = sensors["world_from_camera"][0, 0]
    else:
        sensors["intrinsics"][0, 0, 1, 0] = 1.0e-6
    with pytest.raises(ValueError):
        qualification._validated_public_sensor_tensors(sensors)


def test_materializer_config_binding_is_hook_free_and_container_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class CopySpy:
        def __deepcopy__(self, _memo: Any) -> Any:
            calls.append("deepcopy")
            raise AssertionError("unsupported config value hook ran")

    with pytest.raises(TypeError, match="unsupported type"):
        qualification._materializer_config_sha256({"spy": CopySpy()})
    assert calls == []
    recursive: dict[str, Any] = {}
    recursive["self"] = recursive
    with pytest.raises(ValueError, match="recursive"):
        qualification._materializer_config_sha256(recursive)

    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker="config-container-rebind",
    ) as (ledger, _pin, _config):
        config: dict[str, Any] = {"shape": (1,), "device": "cpu"}
        authority = ledger.mint_formal_evaluator_authority(config)
        config["shape"] = [1]
        with pytest.raises(PermissionError, match="rebound"):
            qualification._FormalKnownActionEvaluator(config, authority=authority)
        assert (
            qualification._EVALUATOR_CONSTRUCTION_AUTHORITY_REGISTRY[id(authority)][3] == "failed"
        )
        assert qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY == {}


@pytest.mark.parametrize("case", ("pathlike", "equality"))
def test_materializer_authority_type_gates_precede_path_and_equality_hooks(
    case: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class PathSpy:
        def __fspath__(self) -> str:
            calls.append("fspath")
            return "/private/tmp/forged"

    class EqualitySpy:
        def __eq__(self, _other: Any) -> bool:
            calls.append("eq")
            return False

    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker=f"constructor-no-hook-{case}",
    ) as (ledger, _pin, config):
        authority = ledger.mint_formal_evaluator_authority(config)
        if case == "pathlike":
            object.__setattr__(authority, "ledger_path", PathSpy())
        else:
            object.__setattr__(authority, "ledger_generation", EqualitySpy())
        with pytest.raises(PermissionError, match="rebound"):
            qualification._FormalKnownActionEvaluator(config, authority=authority)
        assert calls == []
        assert (
            qualification._EVALUATOR_CONSTRUCTION_AUTHORITY_REGISTRY[id(authority)][3] == "failed"
        )


def test_materializer_unregistered_config_class_and_constructor_type_are_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker="exact-config-class",
    ) as (ledger, _pin, _config):
        with pytest.raises(TypeError, match="exact nonempty dictionary"):
            qualification._materializer_config_sha256(object.__new__(qualification.OrpheusConfig))
        config = _config
        authority = ledger.mint_formal_evaluator_authority(config)
        calls: list[str] = []

        class HostileEvaluator(qualification._FormalKnownActionEvaluator):
            def __setattr__(self, _name: str, _value: Any) -> None:
                calls.append("setattr")
                raise AssertionError("hostile evaluator hook ran")

        with pytest.raises(PermissionError, match="constructor type"):
            HostileEvaluator(config, authority=authority)
        assert calls == []
        assert (
            qualification._EVALUATOR_CONSTRUCTION_AUTHORITY_REGISTRY[id(authority)][3] == "issued"
        )
        with pytest.raises(PermissionError, match="ledger authority"):
            qualification._consume_formal_evaluator_authority(
                authority,
                config=config,
                evaluator=object(),
            )
        evaluator = qualification._FormalKnownActionEvaluator(config, authority=authority)
        assert evaluator._config is config
        qualification._close_trusted_materializer_port(evaluator._materializer_port)


@pytest.mark.parametrize(
    "case",
    (
        "sensor_tensor",
        "public_body",
        "blinding_nonce",
        "envelope_nonce",
        "row_nonce",
        "sensor_batch_nonce",
        "task_batch_nonce",
        "port_nonce",
        "vault_nonce",
        "envelope_vault_identity",
    ),
)
def test_materializer_opening_mutation_is_fail_closed_and_nonretryable(
    case: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker=f"opening-mutation-{case}",
    ) as (ledger, _pin, config):
        _authority, _evaluator, port = _construct_synthetic_materializer(ledger, config)
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        vault = qualification._open_materializer_split_vault(port, split="development")
        _request, envelope, evaluation = _materializer_issue_public_batch(
            manifest=manifest,
            port=port,
            vault=vault,
            start=0,
        )
        vault_registration = qualification._MATERIALIZER_SPLIT_VAULT_REGISTRY[id(vault)]
        envelope_registration = qualification._PUBLIC_MATERIALIZATION_ENVELOPE_REGISTRY[
            id(envelope)
        ]
        sensor_batch = envelope_registration.sensor_batch
        task_batch = envelope_registration.task_batch
        assert sensor_batch is not None and task_batch is not None
        sensor_registration = qualification._PUBLIC_SENSOR_BATCH_REGISTRY[id(sensor_batch)]
        row = envelope_registration.rows[0]
        restore: tuple[object, str, Any] | None = None
        if case == "sensor_tensor":
            assert sensor_registration.tensors is not None
            with qualification.torch.no_grad():
                sensor_registration.tensors["rgb"][0, 0, 0, 0, 0] += 0.01
        elif case == "public_body":
            assert envelope_registration.public_bodies is not None
            envelope_registration.public_bodies[0]["sensor_sha256"] = "f" * 64
        elif case == "blinding_nonce":
            original = envelope_registration.blinding_nonces
            assert original is not None
            envelope_registration.blinding_nonces = (b"z" * 32, *original[1:])
        elif case == "envelope_nonce":
            restore = (envelope, "nonce", envelope.nonce)
            object.__setattr__(envelope, "nonce", object())
        elif case == "row_nonce":
            restore = (row, "nonce", row.nonce)
            object.__setattr__(row, "nonce", object())
        elif case == "sensor_batch_nonce":
            restore = (sensor_batch, "nonce", sensor_batch.nonce)
            object.__setattr__(sensor_batch, "nonce", object())
        elif case == "task_batch_nonce":
            restore = (task_batch, "nonce", task_batch.nonce)
            object.__setattr__(task_batch, "nonce", object())
        elif case == "port_nonce":
            restore = (port, "nonce", port.nonce)
            object.__setattr__(port, "nonce", object())
        elif case == "vault_nonce":
            restore = (vault, "nonce", vault.nonce)
            object.__setattr__(vault, "nonce", object())
        else:
            restore = (envelope, "vault_identity", envelope.vault_identity)
            object.__setattr__(envelope, "vault_identity", -1)
        with pytest.raises((PermissionError, ValueError)):
            qualification._register_public_materializer_evaluation(
                port,
                vault,
                envelope,
                evaluation,
            )
        assert vault_registration.state == "failed"
        assert qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY[id(port)].state == "failed"
        _assert_materializer_vault_scrubbed(vault)
        with pytest.raises(PermissionError):
            qualification._register_public_materializer_evaluation(
                port,
                vault,
                envelope,
                evaluation,
            )
        manifest.abort()
        if restore is None:
            assert qualification._exact_materializer_terminal_tombstone_cut()
        else:
            target, field, value = restore
            _assert_materializer_cleanup_refusal_then_restore(
                lambda: object.__setattr__(target, field, value)
            )


def test_materializer_accepts_one_preauthorized_evaluated_public_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker="preauthorized-evaluated-batch",
    ) as (ledger, _pin, config):
        _authority, _evaluator, port = _construct_synthetic_materializer(ledger, config)
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        vault = qualification._open_materializer_split_vault(port, split="development")
        ordinals = tuple(range(qualification.BATCH_SIZE))
        batch = manifest.begin_batch(ordinals)
        request = qualification.PublicBatchEvaluationRequest(
            split="development",
            ordinals=ordinals,
            tokens=batch.tokens,
            _manifest=manifest,
            _batch=batch,
        )
        authorities = qualification._begin_formal_public_science_materialization(
            port,
            vault,
            request,
        )
        evaluation = _fake_public_evaluation_without_consumption(request)
        token_receipts = [
            qualification.copy.deepcopy(authority.token_receipt) for authority in authorities
        ]
        for authority in authorities:
            qualification._consume_materialization_variant(
                authority,
                split="development",
                ordinal=authority.ordinal,
                variant="base",
            )
            qualification._consume_materialization_variant(
                authority,
                split="development",
                ordinal=authority.ordinal,
                variant="palette",
            )
        envelope = qualification._issue_public_materialization_envelope(
            port,
            vault,
            request,
            sensor_tensors=_materializer_sensor_tensors(),
            safe_task_tensors=_materializer_safe_task_tensors(start=0),
            public_runtime_tensors=_materializer_public_runtime_tensors(),
            preauthorized_token_receipts=token_receipts,
            evaluation=evaluation,
        )
        registration = qualification._PUBLIC_MATERIALIZATION_ENVELOPE_REGISTRY[id(envelope)]
        assert registration.state == "evaluated"
        assert registration.request is None
        receipt = manifest.complete_batch(batch, evaluation)
        assert receipt["ordinals"] == list(ordinals)
        assert ledger.record["splits"]["development"]["public_next_ordinal"] == (
            qualification.BATCH_SIZE
        )
        qualification._abort_materializer_split(port, split="development")
        manifest.abort()
        _assert_materializer_vault_scrubbed(vault)


def test_materializer_binding_failure_precedes_evaluation_mapping_hooks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class MappingSpy(Mapping[str, Any]):
        def __getitem__(self, _key: str) -> Any:
            calls.append("getitem")
            raise AssertionError("mapping hook ran")

        def __iter__(self) -> Iterator[str]:
            calls.append("iter")
            raise AssertionError("mapping hook ran")

        def __len__(self) -> int:
            calls.append("len")
            raise AssertionError("mapping hook ran")

    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker="binding-before-mapping",
    ) as (ledger, _pin, config):
        _authority, _evaluator, port = _construct_synthetic_materializer(ledger, config)
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        vault = qualification._open_materializer_split_vault(port, split="development")
        _request, envelope, _evaluation = _materializer_issue_public_batch(
            manifest=manifest,
            port=port,
            vault=vault,
            start=0,
        )
        row = qualification._PUBLIC_MATERIALIZATION_ENVELOPE_REGISTRY[id(envelope)].rows[0]
        original_nonce = row.nonce
        object.__setattr__(row, "nonce", object())
        with pytest.raises(PermissionError):
            qualification._register_public_materializer_evaluation(
                port,
                vault,
                envelope,
                MappingSpy(),
            )
        assert calls == []
        _assert_materializer_vault_scrubbed(vault)
        manifest.abort()
        _assert_materializer_cleanup_refusal_then_restore(
            lambda: object.__setattr__(row, "nonce", original_nonce),
            hook_calls=calls,
        )


def test_materializer_public_registration_fail_burns_consumed_registry_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker="public-registration-consumed-registry-corruption",
    ) as (ledger, _pin, config):
        _authority, _evaluator, port = _construct_synthetic_materializer(ledger, config)
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        vault = qualification._open_materializer_split_vault(port, split="development")
        request, envelope, evaluation = _materializer_issue_public_batch(
            manifest=manifest,
            port=port,
            vault=vault,
            start=0,
        )
        qualification._CONSUMED_ORDINAL_REGISTRY[-1] = ()  # type: ignore[assignment]
        with pytest.raises(PermissionError, match="consumed ordinal registry"):
            qualification._register_public_materializer_evaluation(
                port,
                vault,
                envelope,
                evaluation,
            )
        assert -1 not in qualification._CONSUMED_ORDINAL_REGISTRY
        assert id(request._batch) not in (
            qualification._MATERIALIZER_PUBLIC_BATCH_EVALUATION_REGISTRY
        )
        _assert_materializer_vault_scrubbed(vault)
        assert qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY[id(port)].state == "failed"
        manifest.abort()
        assert qualification._BATCH_REGISTRY == {}
        assert qualification._TOKEN_REGISTRY == {}


@pytest.mark.parametrize("case", ("malformed", "changed"))
def test_materializer_registered_evaluation_must_equal_durable_batch(
    case: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker="registered-evaluation-cut",
    ) as (ledger, _pin, config):
        _authority, _evaluator, port = _construct_synthetic_materializer(ledger, config)
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        vault = qualification._open_materializer_split_vault(port, split="development")
        request, _envelope, evaluation, _hashes = _materializer_issue_and_register_public_batch(
            manifest=manifest,
            port=port,
            vault=vault,
            start=0,
        )
        candidate: Any
        if case == "malformed":
            candidate = object()
        else:
            candidate = qualification.copy.deepcopy(evaluation)
            candidate["public_resources"]["planning_seconds"] = 0.125
        before_bytes = ledger.path.read_bytes()
        before_identity = ledger._last_identity
        with pytest.raises((PermissionError, TypeError), match="dictionary|differ"):
            manifest.complete_batch(request._batch, candidate)
        assert ledger.path.read_bytes() == before_bytes
        assert ledger._last_identity == before_identity
        assert ledger._record["splits"]["development"]["public_active_batch"] == [0, 1, 2, 3]
        _assert_materializer_vault_scrubbed(vault)
        with pytest.raises(PermissionError):
            manifest.complete_batch(request._batch, evaluation)
        manifest.abort()


@pytest.mark.parametrize(
    "case",
    ("malformed", "hash_spy", "vault_hash_spy", "foreign_envelope"),
)
def test_manifest_rejects_crossbound_materializer_evaluation_before_commit(
    case: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class EqualitySpy:
        def __eq__(self, _other: Any) -> bool:
            calls.append("eq")
            raise AssertionError("materializer evaluation equality hook ran")

        def __hash__(self) -> int:
            calls.append("hash")
            raise AssertionError("materializer evaluation hash hook ran")

    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker=f"manifest-materializer-crossbind-{case}",
    ) as (ledger, _pin, config):
        _authority, _evaluator, port = _construct_synthetic_materializer(ledger, config)
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        vault = qualification._open_materializer_split_vault(port, split="development")
        foreign_envelope = None
        start = 0
        if case == "foreign_envelope":
            _first_request, foreign_envelope, _first_hashes, _first_receipt = (
                _materializer_public_batch(
                    manifest=manifest,
                    port=port,
                    vault=vault,
                    start=0,
                )
            )
            start = qualification.BATCH_SIZE
        request, envelope, evaluation, _hashes = _materializer_issue_and_register_public_batch(
            manifest=manifest,
            port=port,
            vault=vault,
            start=start,
        )
        identity = id(request._batch)
        original = qualification._MATERIALIZER_PUBLIC_BATCH_EVALUATION_REGISTRY[identity]
        original_vault_identity = envelope.vault_identity
        if case == "malformed":
            replacement: Any = ()
        elif case == "hash_spy":
            replacement = (*original[:3], EqualitySpy())
        elif case == "vault_hash_spy":
            object.__setattr__(envelope, "vault_identity", EqualitySpy())
            replacement = original
        else:
            assert foreign_envelope is not None
            replacement = (original[0], original[1], foreign_envelope, original[3])
        qualification._MATERIALIZER_PUBLIC_BATCH_EVALUATION_REGISTRY[identity] = replacement
        before = qualification.copy.deepcopy(ledger.record)
        expected_error = (
            "materializer vault registry"
            if case in {"vault_hash_spy", "foreign_envelope"}
            else "materializer evaluation"
        )
        with pytest.raises(PermissionError, match=expected_error):
            manifest.complete_batch(request._batch, evaluation)
        assert ledger.record == before
        assert identity not in qualification._MATERIALIZER_PUBLIC_BATCH_EVALUATION_REGISTRY
        assert calls == []
        _assert_materializer_vault_scrubbed(vault)
        assert qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY[id(port)].state == "failed"
        manifest.abort()
        if case == "vault_hash_spy":
            _assert_materializer_cleanup_refusal_then_restore(
                lambda: object.__setattr__(
                    envelope,
                    "vault_identity",
                    original_vault_identity,
                ),
                hook_calls=calls,
            )
        else:
            assert qualification._exact_materializer_terminal_tombstone_cut()


def test_materializer_public_evaluation_token_receipts_crossbind_envelope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker="evaluation-token-crossbind",
    ) as (ledger, _pin, config):
        _authority, _evaluator, port = _construct_synthetic_materializer(ledger, config)
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        vault = qualification._open_materializer_split_vault(port, split="development")
        _request, envelope, evaluation = _materializer_issue_public_batch(
            manifest=manifest,
            port=port,
            vault=vault,
            start=0,
        )
        forged = qualification.copy.deepcopy(evaluation)
        evidence = forged["public_evidence"]
        row = evidence["row_receipts"][0]
        token = dict(row["token_receipt"])
        token.pop("receipt_sha256")
        token["capability_nonce"] = "f" * 64
        forged_token = {
            **token,
            "receipt_sha256": qualification.canonical_sha256(token),
        }
        row_body = dict(row)
        row_body.pop("receipt_sha256")
        row_body["token_receipt"] = forged_token
        evidence["row_receipts"][0] = {
            **row_body,
            "receipt_sha256": qualification.canonical_sha256(row_body),
        }
        evidence["ordered_row_receipts_sha256"] = qualification.canonical_sha256(
            [item["receipt_sha256"] for item in evidence["row_receipts"]]
        )
        forged["public_evidence_sha256"] = qualification.canonical_sha256(evidence)
        forged["public_metrics"]["public_numeric_evidence_sha256"] = forged[
            "public_evidence_sha256"
        ]
        with pytest.raises(PermissionError, match="token receipts differ"):
            qualification._register_public_materializer_evaluation(
                port,
                vault,
                envelope,
                forged,
            )
        _assert_materializer_vault_scrubbed(vault)
        manifest.abort()


@pytest.mark.parametrize(
    "case",
    ("sensor", "public_body", "nonce", "row_binding", "evaluation_hash"),
)
def test_formal_batch_reopens_retained_materialization_immediately_before_commit(
    case: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker=f"precommit-reopen-{case}",
    ) as (ledger, _pin, config):
        _authority, _evaluator, port = _construct_synthetic_materializer(ledger, config)
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        vault = qualification._open_materializer_split_vault(port, split="development")
        request, envelope, evaluation, _hashes = _materializer_issue_and_register_public_batch(
            manifest=manifest,
            port=port,
            vault=vault,
            start=0,
        )
        envelope_registration = qualification._PUBLIC_MATERIALIZATION_ENVELOPE_REGISTRY[
            id(envelope)
        ]
        sensor_batch = envelope_registration.sensor_batch
        assert sensor_batch is not None
        sensor_registration = qualification._PUBLIC_SENSOR_BATCH_REGISTRY[id(sensor_batch)]
        restore: tuple[object, str, Any] | None = None
        if case == "sensor":
            assert sensor_registration.tensors is not None
            sensor_registration.tensors["rgb"][0, 0, 0, 0, 0] += 0.01
        elif case == "public_body":
            assert envelope_registration.public_bodies is not None
            envelope_registration.public_bodies[0]["sensor_sha256"] = "f" * 64
        elif case == "nonce":
            assert envelope_registration.blinding_nonces is not None
            envelope_registration.blinding_nonces = (
                b"q" * 32,
                *envelope_registration.blinding_nonces[1:],
            )
        elif case == "row_binding":
            row = envelope_registration.rows[0]
            restore = (row, "nonce", row.nonce)
            object.__setattr__(row, "nonce", object())
        else:
            assert envelope_registration.evaluation_hashes is not None
            envelope_registration.evaluation_hashes["public_metrics_sha256"] = "f" * 64
            envelope_registration.evaluation_hashes_binding_sha256 = qualification.canonical_sha256(
                envelope_registration.evaluation_hashes
            )
        before = qualification.copy.deepcopy(ledger.record)
        with pytest.raises((PermissionError, ValueError)):
            manifest.complete_batch(request._batch, evaluation)
        assert ledger.record == before
        assert id(request._batch) not in (
            qualification._MATERIALIZER_PUBLIC_BATCH_EVALUATION_REGISTRY
        )
        _assert_materializer_vault_scrubbed(vault)
        assert qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY[id(port)].state == "failed"
        manifest.abort()
        if restore is None:
            assert qualification._exact_materializer_terminal_tombstone_cut()
        else:
            target, field, value = restore
            _assert_materializer_cleanup_refusal_then_restore(
                lambda: object.__setattr__(target, field, value)
            )


@pytest.mark.parametrize("rebind", (False, True))
def test_materializer_finalize_reopens_every_registered_evaluation_hash(
    rebind: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker="finalize-evaluation-opening",
    ) as (ledger, _pin, config):
        _authority, _evaluator, port = _construct_synthetic_materializer(ledger, config)
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        vault = qualification._open_materializer_split_vault(port, split="development")
        for start in range(
            0,
            qualification.SCENES_PER_SPLIT,
            qualification.BATCH_SIZE,
        ):
            _materializer_public_batch(
                manifest=manifest,
                port=port,
                vault=vault,
                start=start,
            )
        first_envelope = qualification._MATERIALIZER_SPLIT_VAULT_REGISTRY[id(vault)].envelopes[0]
        first_registration = qualification._PUBLIC_MATERIALIZATION_ENVELOPE_REGISTRY[
            id(first_envelope)
        ]
        assert first_registration.evaluation_hashes is not None
        first_registration.evaluation_hashes["public_resources_sha256"] = "f" * 64
        if rebind:
            first_registration.evaluation_hashes_binding_sha256 = qualification.canonical_sha256(
                first_registration.evaluation_hashes
            )
        with pytest.raises(PermissionError, match="evaluation hash|vault registry"):
            qualification._finalize_materializer_public_split(port, vault)
        _assert_materializer_vault_scrubbed(vault)
        manifest.abort()


def test_formal_public_seal_rejects_skipped_materializer_finalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker="seal-without-materializer-finalize",
    ) as (ledger, _pin, config):
        _authority, _evaluator, port = _construct_synthetic_materializer(ledger, config)
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        vault = qualification._open_materializer_split_vault(port, split="development")
        evaluations, receipts = _materializer_fill_public_split(
            manifest=manifest,
            port=port,
            vault=vault,
        )
        finalization = _FakeEvaluator().finalize_public_split(
            "development",
            evaluations,
        )
        before = qualification.copy.deepcopy(ledger.record)
        with pytest.raises(PermissionError, match="not finalized"):
            manifest.seal_public_split(
                finalization,
                batch_results=evaluations,
                receipts=receipts,
            )
        assert ledger.record == before
        _assert_materializer_vault_scrubbed(vault)
        manifest.abort()


@pytest.mark.parametrize(
    "case",
    ("sensor", "row_order", "child_state", "digest", "evaluation_hash"),
)
def test_formal_public_seal_reopens_exact_finalized_materializer_inventory(
    case: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker=f"seal-finalized-reopen-{case}",
    ) as (ledger, _pin, config):
        _authority, _evaluator, port = _construct_synthetic_materializer(ledger, config)
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        vault = qualification._open_materializer_split_vault(port, split="development")
        evaluations, receipts = _materializer_fill_public_split(
            manifest=manifest,
            port=port,
            vault=vault,
        )
        qualification._finalize_materializer_public_split(port, vault)
        registration = qualification._MATERIALIZER_SPLIT_VAULT_REGISTRY[id(vault)]
        original_rows = tuple(registration.rows)
        first_envelope = registration.envelopes[0]
        envelope_registration = qualification._PUBLIC_MATERIALIZATION_ENVELOPE_REGISTRY[
            id(first_envelope)
        ]
        if case == "sensor":
            sensor_batch = registration.sensor_batches[0]
            tensors = qualification._PUBLIC_SENSOR_BATCH_REGISTRY[id(sensor_batch)].tensors
            assert tensors is not None
            tensors["rgb"][0, 0, 0, 0, 0] += 0.01
        elif case == "row_order":
            registration.rows[0], registration.rows[1] = (
                registration.rows[1],
                registration.rows[0],
            )
        elif case == "child_state":
            envelope_registration.state = "evaluated"
        elif case == "digest":
            registration.commitment_inventory_sha256 = "f" * 64
        else:
            assert envelope_registration.evaluation_hashes is not None
            envelope_registration.evaluation_hashes["public_resources_sha256"] = "f" * 64
            envelope_registration.evaluation_hashes_binding_sha256 = qualification.canonical_sha256(
                envelope_registration.evaluation_hashes
            )
        finalization = _FakeEvaluator().finalize_public_split(
            "development",
            evaluations,
        )
        before = qualification.copy.deepcopy(ledger.record)
        with pytest.raises((PermissionError, ValueError)):
            manifest.seal_public_split(
                finalization,
                batch_results=evaluations,
                receipts=receipts,
            )
        assert ledger.record == before
        _assert_materializer_vault_scrubbed(vault)
        manifest.abort()
        if case == "row_order":
            _assert_materializer_cleanup_refusal_then_restore(
                lambda: setattr(registration, "rows", list(original_rows))
            )
        else:
            assert qualification._exact_materializer_terminal_tombstone_cut()


@pytest.mark.parametrize(
    "case",
    (
        "malformed_input",
        "post_state",
        "post_rows",
        "post_binding",
        "post_binding_unused_port",
        "post_port_owner",
        "ledger_generation",
        "post_port_alias",
        "post_slot_alias",
    ),
)
def test_formal_public_seal_latch_is_fail_closed_across_durable_transition(
    case: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker=f"public-seal-latch-{case}",
    ) as (ledger, _pin, config):
        _authority, _evaluator, port = _construct_synthetic_materializer(ledger, config)
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        vault = qualification._open_materializer_split_vault(port, split="development")
        evaluations, receipts = _materializer_fill_public_split(
            manifest=manifest,
            port=port,
            vault=vault,
        )
        qualification._finalize_materializer_public_split(port, vault)
        root = qualification._MATERIALIZER_SPLIT_VAULT_REGISTRY[id(vault)]
        original_root_rows = tuple(root.rows)
        original_root_binding = root.binding
        port_registration = qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY[id(port)]
        finalization = _FakeEvaluator().finalize_public_split(
            "development",
            evaluations,
        )
        before_bytes = ledger.path.read_bytes()
        before_identity = ledger._last_identity
        before_generation = ledger.record["generation"]
        if case != "malformed_input":
            original_complete = ledger.complete_public_split

            def complete_then_mutate(
                split: str,
                summary: Mapping[str, Any],
                seal: Mapping[str, Any],
            ) -> None:
                original_complete(split, summary, seal)
                if case == "post_state":
                    root.state = "public_finalized"
                elif case == "post_rows":
                    root.rows.pop()
                elif case == "post_binding":
                    root.binding = (*root.binding[:-1], object())
                elif case == "post_binding_unused_port":
                    root.binding = (id(port) + 1, *root.binding[1:])
                elif case == "post_port_owner":
                    port_registration.ledger = object()  # type: ignore[assignment]
                elif case == "ledger_generation":
                    ledger._record["generation"] += 1
                elif case == "post_port_alias":
                    qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY[-201] = (
                        qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY[id(port)]
                    )
                else:
                    qualification._MATERIALIZER_SPLIT_VAULT_SLOT_REGISTRY[
                        (id(ledger) + 1, "development")
                    ] = id(vault)

            monkeypatch.setattr(ledger, "complete_public_split", complete_then_mutate)
        candidate: Any = object() if case == "malformed_input" else finalization
        with pytest.raises((PermissionError, TypeError)):
            manifest.seal_public_split(
                candidate,
                batch_results=evaluations,
                receipts=receipts,
            )
        if case == "post_port_owner":
            port_registration.ledger = ledger
            qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY[id(port)] = port_registration
        if case == "malformed_input":
            assert ledger.path.read_bytes() == before_bytes
            assert ledger._last_identity == before_identity
            assert ledger.record["generation"] == before_generation
        else:
            disk_bytes = ledger.path.read_bytes()
            parsed_disk = qualification._parse_ledger_bytes(
                disk_bytes,
                label="postcallback materializer public seal ledger",
            )
            parsed_disk.pop("record_sha256")
            assert parsed_disk["generation"] == before_generation + 1
            if case == "ledger_generation":
                assert ledger.record["generation"] == before_generation + 2
                with pytest.raises(PermissionError):
                    ledger._verify_disk()
                ledger._record = parsed_disk
            else:
                assert ledger.record["generation"] == before_generation + 1
            assert disk_bytes == ledger._last_bytes
            ledger._verify_disk()
            state = parsed_disk["splits"]["development"]
            qualification._validated_public_split_seal(
                state["public_seal"],
                split="development",
                summary=state["public_summary"],
            )
        _assert_materializer_vault_scrubbed(vault)
        assert port_registration.state == "failed"
        with pytest.raises(PermissionError):
            manifest.seal_public_split(
                finalization,
                batch_results=evaluations,
                receipts=receipts,
            )
        manifest.abort()
        if case == "post_rows":
            _assert_materializer_cleanup_refusal_then_restore(
                lambda: setattr(root, "rows", list(original_root_rows))
            )
        elif case in {"post_binding", "post_binding_unused_port"}:
            _assert_materializer_cleanup_refusal_then_restore(
                lambda: setattr(root, "binding", original_root_binding)
            )
        else:
            assert qualification._exact_materializer_terminal_tombstone_cut()


def test_materializer_exact_sixteen_batch_private_lifecycle_is_one_shot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker="full-private-lifecycle",
    ) as (ledger, _pin, config):
        _authority, _evaluator, port = _construct_synthetic_materializer(ledger, config)
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        vault = qualification._open_materializer_split_vault(port, split="development")
        evaluations, receipts, seal = _materializer_fill_and_seal_public_split(
            manifest=manifest,
            port=port,
            vault=vault,
        )
        vault_registration = qualification._MATERIALIZER_SPLIT_VAULT_REGISTRY[id(vault)]
        assert len(evaluations) == len(receipts) == 16
        assert len(vault_registration.envelopes) == 16
        assert len(vault_registration.rows) == 64
        assert [row.ordinal for row in vault_registration.rows] == list(range(64))
        assert vault_registration.state == "public_sealed"
        assert ledger._record["splits"]["development"]["status"] == "public_complete"
        replay_finalization = _FakeEvaluator().finalize_public_split(
            "development",
            evaluations,
        )
        sealed_bytes = ledger.path.read_bytes()
        sealed_identity = ledger._last_identity
        with pytest.raises(PermissionError, match="already crossed"):
            manifest.seal_public_split(
                replay_finalization,
                batch_results=evaluations,
                receipts=receipts,
            )
        assert ledger.path.read_bytes() == sealed_bytes
        assert ledger._last_identity == sealed_identity
        assert vault_registration.state == "public_sealed"
        request = manifest.begin_private_scoring(public_seal=seal)
        assert ledger._record["splits"]["development"]["status"] == "truth_evaluating"
        truth_bytes = ledger.path.read_bytes()
        truth_identity = ledger._last_identity
        with pytest.raises(PermissionError, match="already crossed"):
            manifest.seal_public_split(
                replay_finalization,
                batch_results=evaluations,
                receipts=receipts,
            )
        assert ledger.path.read_bytes() == truth_bytes
        assert ledger._last_identity == truth_identity
        assert vault_registration.state == "public_sealed"
        binding = qualification._bind_materializer_private_request(
            port,
            vault,
            request,
        )
        assert qualification.canonical_sha256(binding) == request.request_sha256
        assert vault_registration.state == "private_consuming"
        assert qualification._TRUTH_AUTHORITY_REGISTRY[id(request._authority)][3] == ("consumed")
        with pytest.raises(PermissionError):
            qualification._bind_materializer_private_request(port, vault, request)
        with pytest.raises(PermissionError, match="replayed"):
            request.consume()
        commitments = qualification._formal_materializer_private_rows(
            port,
            vault,
            request,
        )
        formal_receipt = qualification._formal_private_scoring_receipt(
            split="development",
            public_seal_sha256=request.public_seal_sha256,
            truth_request=binding,
            private_row_commitments=commitments,
            metrics=_passing_metrics(),
            inherited_orbital_evidence=qualification._inherited_orbital_evidence(
                _passing_inherited_metrics()
            ),
        )
        formal_result = qualification._split_result_from_private_receipt(
            split="development",
            state=ledger.record["splits"]["development"],
            private_receipt=formal_receipt,
        )
        manifest.close(request, formal_receipt, formal_result)
        assert ledger.record["splits"]["development"]["status"] == "passed"
        assert vault_registration.state == "consumed"
        _assert_materializer_vault_scrubbed(vault)
        assert qualification._TRUTH_AUTHORITY_REGISTRY[id(request._authority)][3] == ("retired")
        with pytest.raises(PermissionError):
            manifest.close(request, formal_receipt, formal_result)
        with pytest.raises(PermissionError):
            qualification._complete_materializer_private_consumption(
                port,
                vault,
                request,
            )
        qualification._close_trusted_materializer_port(port)
        assert qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY[id(port)].state == "closed"


def test_formal_manifest_and_ledger_completion_type_gate_before_argument_hooks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class MappingSpy(Mapping[str, Any]):
        def __getitem__(self, key: str) -> Any:
            calls.append(f"getitem:{key}")
            raise AssertionError("formal completion traversed a hostile mapping")

        def __iter__(self) -> Iterator[str]:
            calls.append("iter")
            raise AssertionError("formal completion iterated a hostile mapping")

        def __len__(self) -> int:
            calls.append("len")
            raise AssertionError("formal completion measured a hostile mapping")

    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker="formal-close-type-gates",
    ) as (ledger, _pin, _config):
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        before = qualification.copy.deepcopy(ledger.record)
        with pytest.raises(PermissionError, match="arguments are forged"):
            manifest.close(object(), MappingSpy(), MappingSpy())
        with pytest.raises(PermissionError, match="bound private scoring receipt"):
            ledger.complete_split(
                "development",
                MappingSpy(),
                authority=object(),
                receipt=MappingSpy(),
            )
        assert calls == []
        assert ledger.record == before
        manifest.abort()


def test_fake_private_scoring_receipt_remains_explicitly_nonformal() -> None:
    receipt = qualification._private_scoring_receipt(
        split="development",
        public_seal_sha256="a" * 64,
        truth_request={
            "schema": "rgbd_known_action_split_truth_request_v1",
            "split": "development",
            "ledger_generation": 1,
            "ledger_record_sha256": "b" * 64,
            "ledger_file_identity": [1, 2, 3, 4, 5],
            "public_seal_sha256": "a" * 64,
        },
        metrics=_passing_metrics(),
        inherited_orbital_evidence=qualification._inherited_orbital_evidence(
            _passing_inherited_metrics()
        ),
    )
    assert receipt["schema"] == "rgbd_known_action_fake_test_private_scoring_receipt_v1"
    assert receipt["scoring_mode"] == "fake_test_no_scene_or_truth"
    assert (
        qualification._validated_private_scoring_receipt(
            receipt,
            split="development",
            public_seal_sha256="a" * 64,
        )
        == receipt
    )


def test_materializer_early_finalize_and_second_live_vault_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker="early-finalize",
    ) as (ledger, _pin, config):
        _authority, _evaluator, port = _construct_synthetic_materializer(ledger, config)
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        vault = qualification._open_materializer_split_vault(port, split="development")
        with pytest.raises(PermissionError, match="one newly opened"):
            qualification._open_materializer_split_vault(port, split="development")
        _materializer_public_batch(
            manifest=manifest,
            port=port,
            vault=vault,
            start=0,
        )
        with pytest.raises(PermissionError, match="incomplete"):
            qualification._finalize_materializer_public_split(port, vault)
        _assert_materializer_vault_scrubbed(vault)
        manifest.abort()


def test_evaluate_split_public_failure_aborts_evaluator_before_manifest(
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    def capture(_call: int, request: qualification.PublicBatchEvaluationRequest) -> None:
        captured["request"] = request

    def inspect_abort(split: str) -> None:
        request = captured["request"]
        manifest = request._manifest
        registration = qualification._BATCH_REGISTRY[id(request._batch)]
        assert split == "development"
        assert manifest._closed is False
        assert manifest._active is request._batch
        assert registration.status == "issued"
        assert all(
            qualification._TOKEN_REGISTRY[id(token)][2] == "consumed" for token in request.tokens
        )

    evaluator = _FakeEvaluator(
        raise_on_call=1,
        callback=capture,
        abort_callback=inspect_abort,
    )
    with _pinned_run(tmp_path) as pin:
        ledger, _ = _new_development_ledger(pin)
        with pytest.raises(RuntimeError, match="injected evaluator failure"):
            qualification._evaluate_split(
                split="development",
                ledger=ledger,
                evaluator=evaluator,
                boundary_guard=lambda _label: ledger._verify_disk(),
            )
        request = captured["request"]
        assert request._manifest._closed is True
        assert request._manifest._active is None
        assert qualification._BATCH_REGISTRY == {}
        assert qualification._TOKEN_REGISTRY == {}
        assert evaluator.events == [
            ("public_batch", "development"),
            ("abort_split", "development"),
        ]
        _terminate_test_ledger(pin, ledger, message="public evaluator abort ordering")


def test_evaluate_split_private_failure_revokes_truth_after_evaluator_abort(
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    def capture_private(request: qualification.PrivateSplitScoringRequest) -> None:
        captured["request"] = request

    def inspect_abort(split: str) -> None:
        request = captured["request"]
        registration = qualification._TRUTH_AUTHORITY_REGISTRY[id(request._authority)]
        assert split == "development"
        assert request._manifest._closed is False
        assert registration[3] == "consumed"

    evaluator = _FakeEvaluator(
        private_callback=capture_private,
        abort_callback=inspect_abort,
        raise_on_private=True,
    )
    with _pinned_run(tmp_path) as pin:
        ledger, _ = _new_development_ledger(pin)
        with pytest.raises(RuntimeError, match="private scoring failure"):
            qualification._evaluate_split(
                split="development",
                ledger=ledger,
                evaluator=evaluator,
                boundary_guard=lambda _label: ledger._verify_disk(),
            )
        request = captured["request"]
        assert request._manifest._closed is True
        assert qualification._TRUTH_AUTHORITY_REGISTRY[id(request._authority)][3] == ("revoked")
        assert qualification._BATCH_REGISTRY == {}
        assert qualification._TOKEN_REGISTRY == {}
        assert evaluator.events[-3:] == [
            ("public_finalize", "development"),
            ("private_score", "development"),
            ("abort_split", "development"),
        ]
        _terminate_test_ledger(pin, ledger, message="private evaluator abort ordering")


def test_evaluate_split_manifest_cleanup_runs_when_evaluator_abort_raises(
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    def capture(_call: int, request: qualification.PublicBatchEvaluationRequest) -> None:
        captured["manifest"] = request._manifest

    def fail_abort(_split: str) -> None:
        raise RuntimeError("injected abort callback failure")

    evaluator = _FakeEvaluator(
        raise_on_call=1,
        callback=capture,
        abort_callback=fail_abort,
    )
    with _pinned_run(tmp_path) as pin:
        ledger, _ = _new_development_ledger(pin)
        with pytest.raises(RuntimeError, match="abort callback failure"):
            qualification._evaluate_split(
                split="development",
                ledger=ledger,
                evaluator=evaluator,
                boundary_guard=lambda _label: ledger._verify_disk(),
            )
        manifest = captured["manifest"]
        assert manifest._closed is True and manifest._active is None
        assert qualification._BATCH_REGISTRY == {}
        assert qualification._TOKEN_REGISTRY == {}
        _terminate_test_ledger(pin, ledger, message="abort callback cleanup")


def test_formal_evaluator_abort_purges_local_split_in_finally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = object.__new__(qualification._FormalKnownActionEvaluator)
    object.__setattr__(evaluator, "_materializer_port", object())
    object.__setattr__(evaluator, "_closed", set())
    object.__setattr__(evaluator, "_records", {"development": [1], "selector": [2]})
    object.__setattr__(
        evaluator,
        "_manifest_rows",
        {"development": [3], "selector": [4]},
    )
    object.__setattr__(
        evaluator,
        "_truth_registry",
        {("development", 0): object(), ("selector", 0): object()},
    )
    object.__setattr__(
        evaluator,
        "_vaults",
        {"development": object(), "selector": object()},
    )

    def fail_abort(*_args: Any, **_kwargs: Any) -> None:
        raise PermissionError("injected materializer abort failure")

    monkeypatch.setattr(qualification, "_abort_materializer_split", fail_abort)
    with pytest.raises(PermissionError, match="injected materializer"):
        evaluator.abort_split("development")
    assert evaluator._closed == {"development"}
    assert evaluator._records == {"selector": [2]}
    assert evaluator._manifest_rows == {"selector": [4]}
    assert set(evaluator._vaults) == {"selector"}
    assert set(evaluator._truth_registry) == {("selector", 0)}


@pytest.mark.parametrize(
    "case",
    (
        "port_nonce",
        "port_registration_port",
        "port_registration_port_and_ledger",
        "malformed_port_registration",
        "vault_port",
        "envelope_owner",
        "sensor_owner",
    ),
)
def test_materializer_abort_scrubs_live_payload_after_binding_corruption(
    case: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker=f"abort-binding-corruption-{case}",
    ) as (ledger, _pin, config):
        _authority, evaluator, port = _construct_synthetic_materializer(ledger, config)
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        vault = qualification._open_materializer_split_vault(port, split="development")
        _request, envelope, _evaluation = _materializer_issue_public_batch(
            manifest=manifest,
            port=port,
            vault=vault,
            start=0,
        )
        envelope_registration = qualification._PUBLIC_MATERIALIZATION_ENVELOPE_REGISTRY[
            id(envelope)
        ]
        sensor_batch = envelope_registration.sensor_batch
        assert sensor_batch is not None
        port_registration = qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY[id(port)]
        vault_registration = qualification._MATERIALIZER_SPLIT_VAULT_REGISTRY[id(vault)]
        sensor_registration = qualification._PUBLIC_SENSOR_BATCH_REGISTRY[id(sensor_batch)]
        original_port_nonce = port.nonce
        original_port_registration_port = port_registration.port
        original_port_registration_ledger = port_registration.ledger
        original_vault_port = vault_registration.port
        original_envelope_vault = envelope_registration.vault
        original_sensor_vault = sensor_registration.vault
        closed_with_cleanup = False
        if case == "port_nonce":
            object.__setattr__(port, "nonce", object())
        elif case == "port_registration_port":
            port_registration.port = object()
            with pytest.raises(PermissionError, match="binding differs"):
                qualification._close_trusted_materializer_port(port)
        elif case == "port_registration_port_and_ledger":
            port_registration.port = object()
            port_registration.ledger = object()  # type: ignore[assignment]
        elif case == "malformed_port_registration":
            qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY[id(port)] = object()  # type: ignore[assignment]
        elif case == "vault_port":
            vault_registration.port = object()
            with pytest.raises(PermissionError, match="registry was corrupted"):
                qualification._close_trusted_materializer_port(port)
            closed_with_cleanup = True
        elif case == "envelope_owner":
            envelope_registration.vault = object()
        else:
            sensor_registration.vault = object()
        if not closed_with_cleanup:
            with pytest.raises(PermissionError):
                evaluator.abort_split("development")
        _assert_materializer_vault_scrubbed(vault)
        with pytest.raises(PermissionError):
            evaluator.evaluate_public_batch(object())
        manifest.abort()
        assert qualification._BATCH_REGISTRY == {}
        assert qualification._TOKEN_REGISTRY == {}
        if case == "port_nonce":

            def restore() -> None:
                object.__setattr__(port, "nonce", original_port_nonce)

        elif case == "port_registration_port":

            def restore() -> None:
                qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY[
                    id(port)
                ].port = original_port_registration_port

        elif case == "port_registration_port_and_ledger":

            def restore() -> None:
                registration = qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY[id(port)]
                registration.port = original_port_registration_port
                registration.ledger = original_port_registration_ledger

        elif case == "malformed_port_registration":

            def restore() -> None:
                port_registration.state = "failed"
                qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY[id(port)] = port_registration

        elif case == "vault_port":

            def restore() -> None:
                qualification._MATERIALIZER_SPLIT_VAULT_REGISTRY[
                    id(vault)
                ].port = original_vault_port

        elif case == "envelope_owner":

            def restore() -> None:
                qualification._PUBLIC_MATERIALIZATION_ENVELOPE_REGISTRY[
                    id(envelope)
                ].vault = original_envelope_vault

        else:

            def restore() -> None:
                qualification._PUBLIC_SENSOR_BATCH_REGISTRY[
                    id(sensor_batch)
                ].vault = original_sensor_vault

        _assert_materializer_cleanup_refusal_then_restore(restore)


@pytest.mark.parametrize("case", ("binding", "ledger", "authority_field"))
def test_trusted_materializer_port_registry_tamper_scrubs_without_hooks(
    case: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class EqualitySpy:
        def __eq__(self, _other: Any) -> bool:
            calls.append("eq")
            raise AssertionError("trusted port equality hook ran")

        def __ne__(self, _other: Any) -> bool:
            calls.append("ne")
            raise AssertionError("trusted port inequality hook ran")

        def __getattribute__(self, name: str) -> Any:
            if name == "_calls":
                return calls
            if name.startswith("__"):
                return object.__getattribute__(self, name)
            calls.append(f"get:{name}")
            raise AssertionError("trusted port attribute hook ran")

    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker=f"trusted-port-registry-{case}",
    ) as (ledger, _pin, config):
        _authority, evaluator, port = _construct_synthetic_materializer(ledger, config)
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        vault = qualification._open_materializer_split_vault(port, split="development")
        _request, _envelope, _evaluation = _materializer_issue_public_batch(
            manifest=manifest,
            port=port,
            vault=vault,
            start=0,
        )
        port_registration = qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY[id(port)]
        if case == "binding":
            original = port_registration.binding
            port_registration.binding = (EqualitySpy(), *port_registration.binding[1:])

            def restore() -> None:
                port_registration.binding = original

        elif case == "ledger":
            original = port_registration.ledger
            port_registration.ledger = EqualitySpy()  # type: ignore[assignment]

            def restore() -> None:
                port_registration.ledger = original

        else:
            original = port_registration.authority.ledger_generation
            object.__setattr__(
                port_registration.authority,
                "ledger_generation",
                EqualitySpy(),
            )

            def restore() -> None:
                object.__setattr__(
                    port_registration.authority,
                    "ledger_generation",
                    original,
                )

        with pytest.raises(PermissionError):
            evaluator.evaluate_public_batch(object())
        assert calls == []
        assert port_registration.state == "failed"
        _assert_materializer_vault_scrubbed(vault)
        manifest.abort()
        _assert_materializer_cleanup_refusal_then_restore(
            restore,
            hook_calls=calls,
        )


def test_materializer_abort_sweeps_registries_when_owned_inventory_is_corrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker="abort-corrupt-owned-inventory",
    ) as (ledger, _pin, config):
        _authority, evaluator, port = _construct_synthetic_materializer(ledger, config)
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        vault = qualification._open_materializer_split_vault(port, split="development")
        _request, envelope, _evaluation = _materializer_issue_public_batch(
            manifest=manifest,
            port=port,
            vault=vault,
            start=0,
        )
        vault_registration = qualification._MATERIALIZER_SPLIT_VAULT_REGISTRY[id(vault)]
        envelope_registration = qualification._PUBLIC_MATERIALIZATION_ENVELOPE_REGISTRY[
            id(envelope)
        ]
        sensor_batch = envelope_registration.sensor_batch
        task_batch = envelope_registration.task_batch
        rows = envelope_registration.rows
        assert sensor_batch is not None and task_batch is not None
        vault_registration.envelopes = None  # type: ignore[assignment]
        with pytest.raises((PermissionError, TypeError)):
            evaluator.abort_split("development")
        assert vault_registration.state == "failed"
        assert envelope_registration.runtime_tensors is None
        assert envelope_registration.public_bodies is None
        assert envelope_registration.blinding_nonces is None
        assert envelope_registration.evaluation_hashes is None
        assert qualification._PUBLIC_SENSOR_BATCH_REGISTRY[id(sensor_batch)].tensors is None
        assert qualification._SAFE_TASK_BATCH_REGISTRY[id(task_batch)].tensors is None
        assert all(
            qualification._SEALED_MATERIALIZATION_ROW_REGISTRY[id(row)].state == "revoked"
            for row in rows
        )
        manifest.abort()


@pytest.mark.parametrize(
    "registry_name",
    ("evaluation", "envelope", "sensor", "task", "row", "vault"),
)
def test_materializer_abort_scrubs_before_reporting_malformed_child_registry(
    registry_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker=f"abort-malformed-{registry_name}-registry",
    ) as (ledger, _pin, config):
        _authority, evaluator, port = _construct_synthetic_materializer(ledger, config)
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        vault = qualification._open_materializer_split_vault(port, split="development")
        _request, _envelope, _evaluation = _materializer_issue_public_batch(
            manifest=manifest,
            port=port,
            vault=vault,
            start=0,
        )
        registries = {
            "evaluation": qualification._MATERIALIZER_PUBLIC_BATCH_EVALUATION_REGISTRY,
            "envelope": qualification._PUBLIC_MATERIALIZATION_ENVELOPE_REGISTRY,
            "sensor": qualification._PUBLIC_SENSOR_BATCH_REGISTRY,
            "task": qualification._SAFE_TASK_BATCH_REGISTRY,
            "row": qualification._SEALED_MATERIALIZATION_ROW_REGISTRY,
            "vault": qualification._MATERIALIZER_SPLIT_VAULT_REGISTRY,
        }
        registry = registries[registry_name]
        registry[-1] = () if registry_name == "evaluation" else object()  # type: ignore[assignment]
        with pytest.raises(PermissionError, match="registry was corrupted"):
            evaluator.abort_split("development")
        assert -1 not in registry
        _assert_materializer_vault_scrubbed(vault)
        assert qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY[id(port)].state == "failed"
        manifest.abort()
        assert qualification._BATCH_REGISTRY == {}
        assert qualification._TOKEN_REGISTRY == {}


@pytest.mark.parametrize("registry_name", ("evaluation", "vault"))
def test_materializer_revoke_scrubs_before_reporting_malformed_registry(
    registry_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker=f"revoke-malformed-{registry_name}-registry",
    ) as (ledger, _pin, config):
        _authority, _evaluator, port = _construct_synthetic_materializer(ledger, config)
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        vault = qualification._open_materializer_split_vault(port, split="development")
        _request, _envelope, _evaluation = _materializer_issue_public_batch(
            manifest=manifest,
            port=port,
            vault=vault,
            start=0,
        )
        registry = (
            qualification._MATERIALIZER_PUBLIC_BATCH_EVALUATION_REGISTRY
            if registry_name == "evaluation"
            else qualification._MATERIALIZER_SPLIT_VAULT_REGISTRY
        )
        registry[-1] = () if registry_name == "evaluation" else object()  # type: ignore[assignment]
        with pytest.raises(PermissionError, match="registry was corrupted"):
            qualification._revoke_trusted_materializer_port(port)
        assert -1 not in registry
        _assert_materializer_vault_scrubbed(vault)
        assert qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY[id(port)].state == "failed"
        manifest.abort()


@pytest.mark.parametrize(
    ("operation", "registry_name"),
    (
        ("abort", "envelope"),
        ("revoke", "evaluation"),
        ("abort", "vault"),
        ("register", "consumed"),
        ("abort", "slot"),
        ("register", "port"),
        ("register", "batch"),
        ("register", "token"),
    ),
)
def test_materializer_registry_key_quarantine_never_invokes_hooks(
    operation: str,
    registry_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class RegistryKeySpy:
        def __init__(self, collision_hash: int) -> None:
            self.armed = False
            self.collision_hash = collision_hash

        def __hash__(self) -> int:
            if self.armed:
                calls.append("hash")
                raise AssertionError("malformed registry key hash hook ran")
            return self.collision_hash

        def __eq__(self, _other: Any) -> bool:
            if self.armed:
                calls.append("eq")
                raise AssertionError("malformed registry key equality hook ran")
            return self is _other

    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker=f"registry-key-no-hook-{operation}-{registry_name}",
    ) as (ledger, _pin, config):
        _authority, evaluator, port = _construct_synthetic_materializer(ledger, config)
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        vault = qualification._open_materializer_split_vault(port, split="development")
        request, envelope, evaluation = _materializer_issue_public_batch(
            manifest=manifest,
            port=port,
            vault=vault,
            start=0,
        )
        if registry_name == "evaluation":
            qualification._register_public_materializer_evaluation(
                port,
                vault,
                envelope,
                evaluation,
            )
        registries = {
            "envelope": qualification._PUBLIC_MATERIALIZATION_ENVELOPE_REGISTRY,
            "evaluation": (qualification._MATERIALIZER_PUBLIC_BATCH_EVALUATION_REGISTRY),
            "vault": qualification._MATERIALIZER_SPLIT_VAULT_REGISTRY,
            "consumed": qualification._CONSUMED_ORDINAL_REGISTRY,
            "slot": qualification._MATERIALIZER_SPLIT_VAULT_SLOT_REGISTRY,
            "port": qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY,
            "batch": qualification._BATCH_REGISTRY,
            "token": qualification._TOKEN_REGISTRY,
        }
        registry = registries[registry_name]
        exact_key = next(iter(registry))
        key = RegistryKeySpy(hash(exact_key))
        exact_value: Any = (
            qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY[id(port)]
            if registry_name == "port"
            else qualification._BATCH_REGISTRY[id(request._batch)]
            if registry_name == "batch"
            else qualification._TOKEN_REGISTRY[id(request.tokens[0])]
            if registry_name == "token"
            else object()
        )
        registry[key] = exact_value  # type: ignore[index, assignment]
        key.armed = True
        calls.clear()
        with pytest.raises(PermissionError, match="registry.*corrupted"):
            if operation == "abort":
                evaluator.abort_split("development")
            elif operation == "revoke":
                qualification._revoke_trusted_materializer_port(port)
            else:
                qualification._register_public_materializer_evaluation(
                    port,
                    vault,
                    envelope,
                    evaluation,
                )
        assert calls == []
        assert all(
            type(identity) is int
            or (
                registry_name == "slot"
                and type(identity) is tuple
                and len(identity) == 2
                and type(identity[0]) is int
                and type(identity[1]) is str
            )
            for identity in registry
        )
        _assert_materializer_vault_scrubbed(vault)
        assert qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY[id(port)].state == "failed"
        manifest.abort()


@pytest.mark.parametrize("operation", ("register", "abort", "revoke"))
def test_materializer_slot_value_is_type_gated_before_equality_hooks(
    operation: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class SlotValueSpy:
        def __eq__(self, _other: Any) -> bool:
            calls.append("eq")
            raise AssertionError("materializer slot value equality hook ran")

        def __hash__(self) -> int:
            calls.append("hash")
            raise AssertionError("materializer slot value hash hook ran")

    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker=f"slot-value-no-hook-{operation}",
    ) as (ledger, _pin, config):
        _authority, evaluator, port = _construct_synthetic_materializer(ledger, config)
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        vault = qualification._open_materializer_split_vault(port, split="development")
        _request, envelope, evaluation = _materializer_issue_public_batch(
            manifest=manifest,
            port=port,
            vault=vault,
            start=0,
        )
        slot = (id(ledger), "development")
        qualification._MATERIALIZER_SPLIT_VAULT_SLOT_REGISTRY[slot] = SlotValueSpy()  # type: ignore[assignment]
        with pytest.raises(PermissionError):
            if operation == "register":
                qualification._register_public_materializer_evaluation(
                    port,
                    vault,
                    envelope,
                    evaluation,
                )
            elif operation == "abort":
                evaluator.abort_split("development")
            else:
                qualification._revoke_trusted_materializer_port(port)
        assert calls == []
        assert slot not in qualification._MATERIALIZER_SPLIT_VAULT_SLOT_REGISTRY
        _assert_materializer_vault_scrubbed(vault)
        assert qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY[id(port)].state == "failed"
        manifest.abort()


@pytest.mark.parametrize(
    "case",
    (
        "envelope_wrong_key",
        "sensor_wrong_key",
        "row_wrong_key",
        "well_typed_crossbind",
        "slot_missing",
        "slot_wrong_ledger",
        "slot_wrong_vault",
        "root_duplicate_envelope",
    ),
)
def test_materializer_child_keys_root_inventories_and_slots_are_exact_bijections(
    case: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker=f"registry-bijection-{case}",
    ) as (ledger, _pin, config):
        _authority, _evaluator, port = _construct_synthetic_materializer(ledger, config)
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        vault = qualification._open_materializer_split_vault(port, split="development")
        _request, envelope, evaluation = _materializer_issue_public_batch(
            manifest=manifest,
            port=port,
            vault=vault,
            start=0,
        )
        root = qualification._MATERIALIZER_SPLIT_VAULT_REGISTRY[id(vault)]
        envelope_registration = qualification._PUBLIC_MATERIALIZATION_ENVELOPE_REGISTRY[
            id(envelope)
        ]
        sensor_batch = envelope_registration.sensor_batch
        assert sensor_batch is not None
        sensor_registration = qualification._PUBLIC_SENSOR_BATCH_REGISTRY[id(sensor_batch)]
        original_sensor_vault = sensor_registration.vault
        original_envelopes = tuple(root.envelopes)
        row = envelope_registration.rows[0]
        slot = (id(ledger), "development")
        if case == "envelope_wrong_key":
            qualification._PUBLIC_MATERIALIZATION_ENVELOPE_REGISTRY.pop(id(envelope))
            qualification._PUBLIC_MATERIALIZATION_ENVELOPE_REGISTRY[id(envelope) + 1] = (
                envelope_registration
            )
        elif case == "sensor_wrong_key":
            sensor_registration = qualification._PUBLIC_SENSOR_BATCH_REGISTRY.pop(id(sensor_batch))
            qualification._PUBLIC_SENSOR_BATCH_REGISTRY[id(sensor_batch) + 1] = sensor_registration
        elif case == "row_wrong_key":
            row_registration = qualification._SEALED_MATERIALIZATION_ROW_REGISTRY.pop(id(row))
            qualification._SEALED_MATERIALIZATION_ROW_REGISTRY[id(row) + 1] = row_registration
        elif case == "well_typed_crossbind":
            foreign_vault = qualification._MaterializerSplitVault(
                port_identity=id(port),
                ledger_identity=id(ledger),
                split="selector",
                owner_thread=threading.get_ident(),
                nonce=object(),
                _factory_nonce=qualification._MATERIALIZER_FACTORY_NONCE,
            )
            qualification._PUBLIC_SENSOR_BATCH_REGISTRY[id(sensor_batch)].vault = foreign_vault
        elif case == "slot_missing":
            qualification._MATERIALIZER_SPLIT_VAULT_SLOT_REGISTRY.pop(slot)
        elif case == "slot_wrong_ledger":
            qualification._MATERIALIZER_SPLIT_VAULT_SLOT_REGISTRY.pop(slot)
            qualification._MATERIALIZER_SPLIT_VAULT_SLOT_REGISTRY[
                (id(ledger) + 1, "development")
            ] = id(vault)
        elif case == "slot_wrong_vault":
            qualification._MATERIALIZER_SPLIT_VAULT_SLOT_REGISTRY[slot] = id(vault) + 1
        else:
            root.envelopes.append(envelope)
        before = qualification.copy.deepcopy(ledger.record)
        with pytest.raises(PermissionError, match="registry|binding"):
            qualification._register_public_materializer_evaluation(
                port,
                vault,
                envelope,
                evaluation,
            )
        assert ledger.record == before
        _assert_materializer_vault_scrubbed(vault)
        assert qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY[id(port)].state == "failed"
        manifest.abort()
        if case == "well_typed_crossbind":
            _assert_materializer_cleanup_refusal_then_restore(
                lambda: setattr(sensor_registration, "vault", original_sensor_vault)
            )
        elif case == "root_duplicate_envelope":
            _assert_materializer_cleanup_refusal_then_restore(
                lambda: setattr(root, "envelopes", list(original_envelopes))
            )
        else:
            assert qualification._exact_materializer_terminal_tombstone_cut()


def test_materializer_active_batch_requires_one_exact_vault_and_slot_inverse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker="active-batch-missing-root-slot",
    ) as (ledger, _pin, config):
        _authority, evaluator, port = _construct_synthetic_materializer(ledger, config)
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        vault = qualification._open_materializer_split_vault(port, split="development")
        batch = manifest.begin_batch((0, 1, 2, 3))
        before_bytes = ledger.path.read_bytes()
        before_identity = ledger._last_identity
        qualification._MATERIALIZER_SPLIT_VAULT_REGISTRY.pop(id(vault))
        qualification._MATERIALIZER_SPLIT_VAULT_SLOT_REGISTRY.pop((id(ledger), "development"))
        with pytest.raises(PermissionError, match="vault/slot"):
            qualification._validated_materializer_global_registry_cut(
                ledger=ledger,
                port=port,
            )
        assert ledger.path.read_bytes() == before_bytes
        assert ledger._last_identity == before_identity
        assert qualification._BATCH_REGISTRY.get(id(batch)) is None
        assert all(id(token) not in qualification._TOKEN_REGISTRY for token in batch.tokens)
        assert qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY[id(port)].state == "failed"
        with suppress(PermissionError):
            evaluator.abort_split("development")
        with suppress(PermissionError):
            manifest.abort()


def test_materializer_authorities_require_exact_consumed_batch_and_token_inverse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker="authority-consumption-inverse",
    ) as (ledger, _pin, config):
        _authority, evaluator, port = _construct_synthetic_materializer(ledger, config)
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        vault = qualification._open_materializer_split_vault(port, split="development")
        request, _envelope, _evaluation = _materializer_issue_public_batch(
            manifest=manifest,
            port=port,
            vault=vault,
            start=0,
        )
        batch_registration = qualification._BATCH_REGISTRY[id(request._batch)]
        assert batch_registration.consumed == set(request.ordinals)
        assert len(qualification._CONSUMED_ORDINAL_REGISTRY) == qualification.BATCH_SIZE
        batch_registration.consumed.clear()
        for token in request.tokens:
            qualification._TOKEN_REGISTRY[id(token)] = (
                token,
                batch_registration,
                "issued",
            )
        before_bytes = ledger.path.read_bytes()
        before_identity = ledger._last_identity
        with pytest.raises(PermissionError, match="authority|consum"):
            qualification._validated_materializer_global_registry_cut(
                ledger=ledger,
                port=port,
            )
        assert ledger.path.read_bytes() == before_bytes
        assert ledger._last_identity == before_identity
        assert qualification._CONSUMED_ORDINAL_REGISTRY == {}
        _assert_materializer_vault_scrubbed(vault)
        assert qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY[id(port)].state == "failed"
        with suppress(PermissionError):
            evaluator.abort_split("development")
        with suppress(PermissionError):
            manifest.abort()


@pytest.mark.parametrize(
    "case",
    (
        "missing",
        "aliased_batch",
        "alias_without_evaluation",
        "replacement_alias",
        "committed_orphan",
    ),
)
def test_materializer_pending_evaluation_registry_is_one_exact_bijection(
    case: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker=f"pending-evaluation-bijection-{case}",
    ) as (ledger, _pin, config):
        _authority, _evaluator, port = _construct_synthetic_materializer(ledger, config)
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        vault = qualification._open_materializer_split_vault(port, split="development")
        request, envelope, evaluation, _hashes = _materializer_issue_and_register_public_batch(
            manifest=manifest,
            port=port,
            vault=vault,
            start=0,
        )
        original = qualification._MATERIALIZER_PUBLIC_BATCH_EVALUATION_REGISTRY[id(request._batch)]
        if case == "missing":
            qualification._MATERIALIZER_PUBLIC_BATCH_EVALUATION_REGISTRY.pop(id(request._batch))
        elif case in {
            "aliased_batch",
            "alias_without_evaluation",
            "replacement_alias",
        }:
            alias_tokens = tuple(
                qualification._OrdinalCapability(
                    split="development",
                    ordinal=ordinal,
                    nonce=object(),
                )
                for ordinal in request.ordinals
            )
            alias_batch = qualification._BatchCapability(
                split="development",
                ordinals=request.ordinals,
                tokens=alias_tokens,
                capability_nonces=request._batch.capability_nonces,
                capability_nonce_preflight_sha256=(
                    request._batch.capability_nonce_preflight_sha256
                ),
                nonce=object(),
            )
            alias_registration = qualification._BatchRegistration(
                batch=alias_batch,
                manifest=manifest,
                consumed=set(request.ordinals),
                status="issued",
            )
            qualification._BATCH_REGISTRY[
                id(request._batch) if case == "replacement_alias" else id(alias_batch)
            ] = alias_registration
            for token in alias_tokens:
                qualification._TOKEN_REGISTRY[id(token)] = (
                    token,
                    alias_registration,
                    "consumed",
                )
            if case == "aliased_batch":
                qualification._MATERIALIZER_PUBLIC_BATCH_EVALUATION_REGISTRY[id(alias_batch)] = (
                    alias_batch,
                    manifest,
                    envelope,
                    original[3],
                )
        else:
            manifest.complete_batch(request._batch, evaluation)
            next_request, _next_envelope, next_evaluation, _next_hashes = (
                _materializer_issue_and_register_public_batch(
                    manifest=manifest,
                    port=port,
                    vault=vault,
                    start=qualification.BATCH_SIZE,
                )
            )
            qualification._MATERIALIZER_PUBLIC_BATCH_EVALUATION_REGISTRY[id(request._batch)] = (
                original
            )
            request = next_request
            evaluation = next_evaluation
        before_bytes = ledger.path.read_bytes()
        before_identity = ledger._last_identity
        with pytest.raises(PermissionError):
            manifest.complete_batch(request._batch, evaluation)
        assert ledger.path.read_bytes() == before_bytes
        assert ledger._last_identity == before_identity
        assert qualification._MATERIALIZER_PUBLIC_BATCH_EVALUATION_REGISTRY == {}
        _assert_materializer_vault_scrubbed(vault)
        assert qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY[id(port)].state == "failed"
        with suppress(PermissionError):
            manifest.abort()


def test_materializer_direct_fail_uses_canonical_owner_not_root_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker="direct-fail-canonical-owner",
    ) as (ledger, _pin, config):
        _authority, _evaluator, port = _construct_synthetic_materializer(ledger, config)
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        vault = qualification._open_materializer_split_vault(port, split="development")
        request, _envelope, _evaluation = _materializer_issue_public_batch(
            manifest=manifest,
            port=port,
            vault=vault,
            start=0,
        )
        root = qualification._MATERIALIZER_SPLIT_VAULT_REGISTRY[id(vault)]
        port_registration = qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY[id(port)]
        batch_registration = qualification._BATCH_REGISTRY[id(request._batch)]
        root.binding = (id(port) + 1, *root.binding[1:])
        with pytest.raises(PermissionError, match="root/port binding"):
            qualification._fail_materializer_vault(root)
        assert port_registration.state == "failed"
        assert root.state == "failed"
        assert batch_registration.status == "revoked"
        assert manifest._active is None
        assert qualification._BATCH_REGISTRY == {}
        assert qualification._TOKEN_REGISTRY == {}
        assert qualification._CONSUMED_ORDINAL_REGISTRY == {}
        assert qualification._MATERIALIZER_PUBLIC_BATCH_EVALUATION_REGISTRY == {}
        _assert_materializer_vault_scrubbed(vault)
        manifest.abort()
        assert qualification._exact_materializer_terminal_tombstone_cut(), {
            "construction": [
                item.state
                for item in qualification._MATERIALIZER_CONSTRUCTION_AUTHORITY_REGISTRY.values()
            ],
            "ports": [
                item.state for item in qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY.values()
            ],
            "roots": [
                (
                    item.state,
                    qualification._exact_materializer_vault_root_registration(
                        id(item.vault),
                        item,
                    ),
                    qualification._materializer_vault_requires_cleanup(item),
                )
                for item in qualification._MATERIALIZER_SPLIT_VAULT_REGISTRY.values()
            ],
            "initializations": len(qualification._FORMAL_EVALUATOR_INITIALIZATION_REGISTRY),
        }


@pytest.mark.parametrize(
    "case",
    (
        "root_state",
        "root_rows",
        "root_binding",
        "root_binding_unused_port",
        "port_registration_owner",
        "ledger_generation",
        "batch_registry",
        "batch_token_replacement",
        "token_registry",
        "public_body_key",
        "evaluation_hash_key",
        "sensor_alias",
        "task_alias",
        "row_alias",
        "evaluation_alias",
        "batch_alias",
        "token_alias",
        "slot_alias",
    ),
)
def test_materializer_commit_latch_survives_durable_and_registry_mutation_windows(
    case: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class ArmedHashSpy:
        armed = False

        def __hash__(self) -> int:
            if self.armed:
                calls.append("hash")
                raise AssertionError("postcommit hostile key hash hook ran")
            return 8675309

        def __eq__(self, _other: Any) -> bool:
            if self.armed:
                calls.append("eq")
                raise AssertionError("postcommit hostile key equality hook ran")
            return False

    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker=f"commit-latch-window-{case}",
    ) as (ledger, _pin, config):
        _authority, _evaluator, port = _construct_synthetic_materializer(ledger, config)
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        vault = qualification._open_materializer_split_vault(port, split="development")
        request, envelope, evaluation, _hashes = _materializer_issue_and_register_public_batch(
            manifest=manifest,
            port=port,
            vault=vault,
            start=0,
        )
        root = qualification._MATERIALIZER_SPLIT_VAULT_REGISTRY[id(vault)]
        original_root_rows = tuple(root.rows)
        original_root_binding = root.binding
        batch_registration = qualification._BATCH_REGISTRY[id(request._batch)]
        port_registration = qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY[id(port)]
        replacement_registrations: list[qualification._BatchRegistration] = []
        envelope_registration = qualification._PUBLIC_MATERIALIZATION_ENVELOPE_REGISTRY[
            id(envelope)
        ]
        original_commit = ledger.commit_batch

        def commit_then_mutate(split: str, receipt: Mapping[str, Any]) -> None:
            original_commit(split, receipt)
            if case == "root_state":
                root.state = "collecting_public"
            elif case == "root_rows":
                root.rows.pop()
            elif case == "root_binding":
                root.binding = (*root.binding[:-1], object())
            elif case == "root_binding_unused_port":
                root.binding = (id(port) + 1, *root.binding[1:])
            elif case == "port_registration_owner":
                port_registration.ledger = object()  # type: ignore[assignment]
            elif case == "ledger_generation":
                ledger._record["generation"] += 1
            elif case == "batch_registry":
                qualification._BATCH_REGISTRY.pop(id(request._batch))
            elif case == "batch_token_replacement":
                replacement = qualification._BatchRegistration(
                    batch=request._batch,
                    manifest=manifest,
                    consumed=set(request.ordinals),
                    status="issued",
                )
                replacement_registrations.append(replacement)
                qualification._BATCH_REGISTRY[id(request._batch)] = replacement
                for token in request.tokens:
                    qualification._TOKEN_REGISTRY.pop(id(token), None)
            elif case == "token_registry":
                qualification._TOKEN_REGISTRY.pop(id(request.tokens[0]))
            elif case == "public_body_key":
                assert envelope_registration.public_bodies is not None
                key = ArmedHashSpy()
                envelope_registration.public_bodies[0][key] = "forged"  # type: ignore[index]
                key.armed = True
            elif case == "evaluation_hash_key":
                assert envelope_registration.evaluation_hashes is not None
                key = ArmedHashSpy()
                envelope_registration.evaluation_hashes[key] = "forged"  # type: ignore[index]
                key.armed = True
            if case == "sensor_alias":
                sensor = envelope_registration.sensor_batch
                assert sensor is not None
                qualification._PUBLIC_SENSOR_BATCH_REGISTRY[-101] = (
                    qualification._PUBLIC_SENSOR_BATCH_REGISTRY[id(sensor)]
                )
            elif case == "task_alias":
                task = envelope_registration.task_batch
                assert task is not None
                qualification._SAFE_TASK_BATCH_REGISTRY[-102] = (
                    qualification._SAFE_TASK_BATCH_REGISTRY[id(task)]
                )
            elif case == "row_alias":
                row = envelope_registration.rows[0]
                qualification._SEALED_MATERIALIZATION_ROW_REGISTRY[-103] = (
                    qualification._SEALED_MATERIALIZATION_ROW_REGISTRY[id(row)]
                )
            elif case == "evaluation_alias":
                qualification._MATERIALIZER_PUBLIC_BATCH_EVALUATION_REGISTRY[-104] = (
                    qualification._MATERIALIZER_PUBLIC_BATCH_EVALUATION_REGISTRY[id(request._batch)]
                )
            elif case == "batch_alias":
                qualification._BATCH_REGISTRY[-105] = qualification._BATCH_REGISTRY[
                    id(request._batch)
                ]
            elif case == "token_alias":
                qualification._TOKEN_REGISTRY[-106] = qualification._TOKEN_REGISTRY[
                    id(request.tokens[0])
                ]
            elif case == "slot_alias":
                qualification._MATERIALIZER_SPLIT_VAULT_SLOT_REGISTRY[
                    (id(ledger) + 1, "development")
                ] = id(vault)

        monkeypatch.setattr(ledger, "commit_batch", commit_then_mutate)
        before_generation = ledger.record["generation"]
        with pytest.raises((PermissionError, TypeError)):
            manifest.complete_batch(request._batch, evaluation)
        if case == "port_registration_owner":
            port_registration.ledger = ledger
            qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY[id(port)] = port_registration
        disk_bytes = ledger.path.read_bytes()
        parsed_disk = qualification._parse_ledger_bytes(
            disk_bytes,
            label="postcallback materializer commit ledger",
        )
        parsed_disk.pop("record_sha256")
        assert parsed_disk["generation"] == before_generation + 1
        if case == "ledger_generation":
            assert ledger.record["generation"] == before_generation + 2
            with pytest.raises(PermissionError):
                ledger._verify_disk()
            ledger._record = parsed_disk
        else:
            assert ledger.record["generation"] == before_generation + 1
        assert disk_bytes == ledger._last_bytes
        ledger._verify_disk()
        durable_receipt = parsed_disk["splits"]["development"]["public_batch_receipts"][0]
        qualification._validate_batch_receipt(
            durable_receipt,
            split="development",
            batch_index=0,
        )
        assert calls == []
        assert qualification._BATCH_REGISTRY == {}
        assert qualification._TOKEN_REGISTRY == {}
        assert qualification._CONSUMED_ORDINAL_REGISTRY == {}
        assert qualification._MATERIALIZER_PUBLIC_BATCH_EVALUATION_REGISTRY == {}
        assert manifest._active is None
        assert batch_registration.status == "revoked"
        assert all(item.status == "revoked" for item in replacement_registrations)
        _assert_materializer_vault_scrubbed(vault)
        assert port_registration.state == "failed"
        with suppress(PermissionError):
            manifest.abort()
        if case == "root_rows":
            _assert_materializer_cleanup_refusal_then_restore(
                lambda: setattr(root, "rows", list(original_root_rows))
            )
        elif case in {"root_binding", "root_binding_unused_port"}:
            _assert_materializer_cleanup_refusal_then_restore(
                lambda: setattr(root, "binding", original_root_binding)
            )
        else:
            assert qualification._exact_materializer_terminal_tombstone_cut()


@pytest.mark.parametrize("case", ("missing", "reordered", "coherent_rollback"))
def test_materializer_root_rows_are_exact_before_next_durable_reservation(
    case: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker=f"root-row-before-reserve-{case}",
    ) as (ledger, _pin, config):
        _authority, _evaluator, port = _construct_synthetic_materializer(ledger, config)
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        vault = qualification._open_materializer_split_vault(port, split="development")
        request, envelope, _hashes, _receipt = _materializer_public_batch(
            manifest=manifest,
            port=port,
            vault=vault,
            start=0,
        )
        root = qualification._MATERIALIZER_SPLIT_VAULT_REGISTRY[id(vault)]
        original_rows = tuple(root.rows)
        original_next_ordinal = root.next_ordinal
        envelope_registration = qualification._PUBLIC_MATERIALIZATION_ENVELOPE_REGISTRY[
            id(envelope)
        ]
        if case == "missing":
            root.rows.pop()
        elif case == "reordered":
            root.rows[0], root.rows[1] = root.rows[1], root.rows[0]
        else:
            root.next_ordinal = 0
            root.rows.clear()
            envelope_registration.request = request
            envelope_registration.evaluation_hashes = None
            envelope_registration.evaluation_hashes_binding_sha256 = None
            envelope_registration.state = "issued"
            sensor = envelope_registration.sensor_batch
            task = envelope_registration.task_batch
            assert sensor is not None and task is not None
            qualification._PUBLIC_SENSOR_BATCH_REGISTRY[id(sensor)].state = "issued"
            qualification._SAFE_TASK_BATCH_REGISTRY[id(task)].state = "issued"
            for row in envelope_registration.rows:
                qualification._SEALED_MATERIALIZATION_ROW_REGISTRY[id(row)].state = "issued"
        before_bytes = ledger.path.read_bytes()
        before_identity = ledger._last_identity
        with pytest.raises(PermissionError):
            manifest.begin_batch(tuple(range(4, 8)))
        assert ledger.path.read_bytes() == before_bytes
        assert ledger._last_identity == before_identity
        _assert_materializer_vault_scrubbed(vault)
        assert qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY[id(port)].state == "failed"
        manifest.abort()
        if case == "coherent_rollback":

            def restore_root_inventory() -> None:
                root.rows = list(original_rows)
                root.next_ordinal = original_next_ordinal

            _assert_materializer_cleanup_refusal_then_restore(restore_root_inventory)
        else:
            _assert_materializer_cleanup_refusal_then_restore(
                lambda: setattr(root, "rows", list(original_rows))
            )


def test_materializer_root_record_type_gate_precedes_mapping_hooks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class RecordSpy:
        def get(self, _key: str, _default: Any = None) -> Any:
            calls.append("get")
            raise AssertionError("materializer root traversed a hostile ledger record")

    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker="root-record-type-gate",
    ) as (ledger, _pin, config):
        _authority, evaluator, port = _construct_synthetic_materializer(ledger, config)
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        vault = qualification._open_materializer_split_vault(port, split="development")
        root = qualification._MATERIALIZER_SPLIT_VAULT_REGISTRY[id(vault)]
        original_record = ledger._record
        ledger._record = RecordSpy()  # type: ignore[assignment]
        try:
            assert not qualification._exact_materializer_vault_root_registration(
                id(vault),
                root,
            )
            assert calls == []
        finally:
            ledger._record = original_record
        evaluator.abort_split("development")
        manifest.abort()


@pytest.mark.parametrize("operation", ("abort", "revoke"))
def test_materializer_replaced_live_vault_root_is_reconstructed_and_scrubbed(
    operation: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker=f"replaced-live-vault-root-{operation}",
    ) as (ledger, _pin, config):
        _authority, evaluator, port = _construct_synthetic_materializer(ledger, config)
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        vault = qualification._open_materializer_split_vault(port, split="development")
        _request, _envelope, _evaluation = _materializer_issue_public_batch(
            manifest=manifest,
            port=port,
            vault=vault,
            start=0,
        )
        qualification._MATERIALIZER_SPLIT_VAULT_REGISTRY[id(vault)] = object()  # type: ignore[assignment]
        with pytest.raises(PermissionError, match="vault registry was corrupted"):
            if operation == "abort":
                evaluator.abort_split("development")
            else:
                qualification._revoke_trusted_materializer_port(port)
        reconstructed = qualification._MATERIALIZER_SPLIT_VAULT_REGISTRY[id(vault)]
        assert type(reconstructed) is qualification._MaterializerVaultRegistration
        _assert_materializer_vault_scrubbed(vault)
        assert qualification._MATERIALIZER_PUBLIC_BATCH_EVALUATION_REGISTRY == {}
        assert qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY[id(port)].state == "failed"
        with pytest.raises(PermissionError):
            evaluator.evaluate_public_batch(object())
        manifest.abort()


@pytest.mark.parametrize("operation", ("register", "complete", "reopen", "close"))
@pytest.mark.parametrize("replacement", ("object", "attribute_spy"))
def test_materializer_normal_access_root_replacement_is_hook_free_and_fail_burns(
    operation: str,
    replacement: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class AttributeSpy:
        def __getattribute__(self, name: str) -> Any:
            if name != "__class__":
                calls.append(name)
                raise AssertionError("malformed vault registration attribute hook ran")
            return object.__getattribute__(self, name)

    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker=f"normal-root-replacement-{operation}-{replacement}",
    ) as (ledger, _pin, config):
        _authority, _evaluator, port = _construct_synthetic_materializer(ledger, config)
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        vault = qualification._open_materializer_split_vault(port, split="development")
        request, envelope, evaluation = _materializer_issue_public_batch(
            manifest=manifest,
            port=port,
            vault=vault,
            start=0,
        )
        if operation == "complete":
            qualification._register_public_materializer_evaluation(
                port,
                vault,
                envelope,
                evaluation,
            )
        replacement_value = object() if replacement == "object" else AttributeSpy()
        qualification._MATERIALIZER_SPLIT_VAULT_REGISTRY[id(vault)] = replacement_value  # type: ignore[assignment]
        calls.clear()
        with pytest.raises(PermissionError, match="registry was corrupted"):
            if operation == "register":
                qualification._register_public_materializer_evaluation(
                    port,
                    vault,
                    envelope,
                    evaluation,
                )
            elif operation == "complete":
                manifest.complete_batch(request._batch, evaluation)
            elif operation == "reopen":
                qualification._open_materializer_split_vault(
                    port,
                    split="development",
                )
            else:
                qualification._close_trusted_materializer_port(port)
        assert calls == []
        reconstructed = qualification._MATERIALIZER_SPLIT_VAULT_REGISTRY[id(vault)]
        assert type(reconstructed) is qualification._MaterializerVaultRegistration
        _assert_materializer_vault_scrubbed(vault)
        assert qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY[id(port)].state == "failed"
        assert ledger.record["splits"]["development"]["public_batch_receipts"] == []
        with pytest.raises(PermissionError):
            qualification._register_public_materializer_evaluation(
                port,
                vault,
                envelope,
                evaluation,
            )
        manifest.abort()


@pytest.mark.parametrize("operation", ("register", "complete"))
@pytest.mark.parametrize("field", ("ledger", "port"))
def test_materializer_root_crossbinding_is_scrubbed_before_normal_access(
    operation: str,
    field: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker=f"normal-root-crossbinding-{operation}-{field}",
    ) as (ledger, _pin, config):
        _authority, _evaluator, port = _construct_synthetic_materializer(ledger, config)
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        vault = qualification._open_materializer_split_vault(port, split="development")
        request, envelope, evaluation = _materializer_issue_public_batch(
            manifest=manifest,
            port=port,
            vault=vault,
            start=0,
        )
        if operation == "complete":
            qualification._register_public_materializer_evaluation(
                port,
                vault,
                envelope,
                evaluation,
            )
        registration = qualification._MATERIALIZER_SPLIT_VAULT_REGISTRY[id(vault)]
        setattr(registration, field, object())
        with pytest.raises(PermissionError, match="registry was corrupted"):
            if operation == "register":
                qualification._register_public_materializer_evaluation(
                    port,
                    vault,
                    envelope,
                    evaluation,
                )
            else:
                manifest.complete_batch(request._batch, evaluation)
        reconstructed = qualification._MATERIALIZER_SPLIT_VAULT_REGISTRY[id(vault)]
        assert type(reconstructed) is qualification._MaterializerVaultRegistration
        _assert_materializer_vault_scrubbed(vault)
        assert qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY[id(port)].state == "failed"
        assert ledger.record["splits"]["development"]["public_batch_receipts"] == []
        with pytest.raises(PermissionError):
            qualification._register_public_materializer_evaluation(
                port,
                vault,
                envelope,
                evaluation,
            )
        manifest.abort()


@pytest.mark.parametrize("operation", ("register", "complete"))
@pytest.mark.parametrize("field", ("port_identity", "ledger_identity"))
def test_materializer_vault_identity_is_type_gated_before_equality_hooks(
    operation: str,
    field: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class EqualitySpy:
        def __eq__(self, _other: Any) -> bool:
            calls.append("eq")
            raise AssertionError("materializer vault identity equality hook ran")

        def __ne__(self, _other: Any) -> bool:
            calls.append("ne")
            raise AssertionError("materializer vault identity inequality hook ran")

    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker=f"vault-identity-no-hook-{operation}-{field}",
    ) as (ledger, _pin, config):
        _authority, _evaluator, port = _construct_synthetic_materializer(ledger, config)
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        vault = qualification._open_materializer_split_vault(port, split="development")
        request, envelope, evaluation = _materializer_issue_public_batch(
            manifest=manifest,
            port=port,
            vault=vault,
            start=0,
        )
        if operation == "complete":
            qualification._register_public_materializer_evaluation(
                port,
                vault,
                envelope,
                evaluation,
            )
        original = getattr(vault, field)
        object.__setattr__(vault, field, EqualitySpy())
        with pytest.raises(PermissionError, match="registry was corrupted"):
            if operation == "register":
                qualification._register_public_materializer_evaluation(
                    port,
                    vault,
                    envelope,
                    evaluation,
                )
            else:
                manifest.complete_batch(request._batch, evaluation)
        assert calls == []
        _assert_materializer_vault_scrubbed(vault)
        assert qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY[id(port)].state == "failed"
        assert ledger.record["splits"]["development"]["public_batch_receipts"] == []
        with pytest.raises(PermissionError):
            qualification._register_public_materializer_evaluation(
                port,
                vault,
                envelope,
                evaluation,
            )
        manifest.abort()
        _assert_materializer_cleanup_refusal_then_restore(
            lambda: object.__setattr__(vault, field, original),
            hook_calls=calls,
        )


def test_materializer_port_cannot_close_over_unknown_live_vault_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker="close-unknown-vault-state",
    ) as (ledger, _pin, config):
        _authority, evaluator, port = _construct_synthetic_materializer(ledger, config)
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        vault = qualification._open_materializer_split_vault(port, split="development")
        _request, _envelope, _evaluation = _materializer_issue_public_batch(
            manifest=manifest,
            port=port,
            vault=vault,
            start=0,
        )
        registration = qualification._MATERIALIZER_SPLIT_VAULT_REGISTRY[id(vault)]
        registration.state = "unknown_corrupt_state"
        with pytest.raises(PermissionError, match="registry was corrupted"):
            qualification._close_trusted_materializer_port(port)
        assert qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY[id(port)].state == "failed"
        _assert_materializer_vault_scrubbed(vault)
        manifest.abort()


@pytest.mark.parametrize("forged_state", ("consumed", "failed", "revoked"))
def test_materializer_forged_terminal_state_cannot_hide_retained_payload(
    forged_state: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker=f"forged-terminal-state-{forged_state}",
    ) as (ledger, _pin, config):
        _authority, evaluator, port = _construct_synthetic_materializer(ledger, config)
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        vault = qualification._open_materializer_split_vault(port, split="development")
        _request, _envelope, _evaluation = _materializer_issue_public_batch(
            manifest=manifest,
            port=port,
            vault=vault,
            start=0,
        )
        registration = qualification._MATERIALIZER_SPLIT_VAULT_REGISTRY[id(vault)]
        assert registration.terminal_marker is None
        registration.state = forged_state
        with pytest.raises(PermissionError, match="registry was corrupted"):
            qualification._close_trusted_materializer_port(port)
        assert qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY[id(port)].state == "failed"
        _assert_materializer_vault_scrubbed(vault)
        manifest.abort()


@pytest.mark.parametrize("case", ("port_state", "vault_state", "vault_binding_split"))
def test_materializer_abort_type_gates_registry_state_before_hooks(
    case: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class StateSpy:
        def __eq__(self, _other: Any) -> bool:
            calls.append("eq")
            raise AssertionError("registry state equality hook ran")

        def __hash__(self) -> int:
            calls.append("hash")
            raise AssertionError("registry state hash hook ran")

    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker=f"abort-state-no-hook-{case}",
    ) as (ledger, _pin, config):
        _authority, evaluator, port = _construct_synthetic_materializer(ledger, config)
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        vault = qualification._open_materializer_split_vault(port, split="development")
        _request, _envelope, _evaluation = _materializer_issue_public_batch(
            manifest=manifest,
            port=port,
            vault=vault,
            start=0,
        )
        port_registration = qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY[id(port)]
        vault_registration = qualification._MATERIALIZER_SPLIT_VAULT_REGISTRY[id(vault)]
        if case == "port_state":
            port_registration.state = StateSpy()  # type: ignore[assignment]
        elif case == "vault_state":
            vault_registration.state = StateSpy()  # type: ignore[assignment]
        else:
            binding = list(vault_registration.binding)
            binding[2] = StateSpy()
            vault_registration.binding = tuple(binding)
        with pytest.raises(PermissionError):
            evaluator.abort_split("development")
        assert calls == []
        assert port_registration.state == "failed"
        _assert_materializer_vault_scrubbed(vault)
        manifest.abort()


def test_materializer_scrub_preserves_prior_consumed_children_on_same_port(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker="scrub-preserves-retired",
    ) as (ledger, _pin, config):
        _authority, _evaluator, port = _construct_synthetic_materializer(ledger, config)
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        first_vault = qualification._open_materializer_split_vault(
            port,
            split="development",
        )
        _request, envelope, _evaluation = _materializer_issue_public_batch(
            manifest=manifest,
            port=port,
            vault=first_vault,
            start=0,
        )
        first_registration = qualification._MATERIALIZER_SPLIT_VAULT_REGISTRY[id(first_vault)]
        first_envelope_registration = qualification._PUBLIC_MATERIALIZATION_ENVELOPE_REGISTRY[
            id(envelope)
        ]
        sensor_batch = first_envelope_registration.sensor_batch
        task_batch = first_envelope_registration.task_batch
        rows = first_envelope_registration.rows
        assert sensor_batch is not None and task_batch is not None
        qualification._scrub_materializer_vault(first_registration, state="consumed")
        second_vault = qualification._MaterializerSplitVault(
            port_identity=id(port),
            ledger_identity=id(ledger),
            split="selector",
            owner_thread=threading.get_ident(),
            nonce=object(),
            _factory_nonce=qualification._MATERIALIZER_FACTORY_NONCE,
        )
        second_registration = qualification._MaterializerVaultRegistration(
            vault=second_vault,
            port=port,
            ledger=ledger,
            envelopes=[],
            sensor_batches=[],
            task_batches=[],
            rows=[],
            next_ordinal=0,
            commitment_inventory_sha256=None,
            private_request=None,
            private_request_binding=None,
            binding=qualification._materializer_vault_binding(second_vault),
            terminal_marker=None,
            state="collecting_public",
        )
        qualification._MATERIALIZER_SPLIT_VAULT_REGISTRY[id(second_vault)] = second_registration
        qualification._scrub_materializer_vault(second_registration, state="revoked")
        assert qualification._PUBLIC_SENSOR_BATCH_REGISTRY[id(sensor_batch)].state == ("retired")
        assert qualification._SAFE_TASK_BATCH_REGISTRY[id(task_batch)].state == "retired"
        assert all(
            qualification._SEALED_MATERIALIZATION_ROW_REGISTRY[id(row)].state == "retired"
            for row in rows
        )
        with pytest.raises(
            PermissionError,
            match="manifest capability registry ownership changed during abort",
        ):
            manifest.abort()


@pytest.mark.parametrize("case", ("request_nonce", "truth_manifest"))
def test_materializer_private_completion_revalidates_bound_request_and_truth_tuple(
    case: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker=f"private-completion-rebind-{case}",
    ) as (ledger, _pin, config):
        _authority, _evaluator, port = _construct_synthetic_materializer(ledger, config)
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        vault = qualification._open_materializer_split_vault(port, split="development")
        _evaluations, _receipts, seal = _materializer_fill_and_seal_public_split(
            manifest=manifest,
            port=port,
            vault=vault,
        )
        request = manifest.begin_private_scoring(public_seal=seal)
        qualification._bind_materializer_private_request(port, vault, request)
        authority = request._authority
        truth_registration = qualification._TRUTH_AUTHORITY_REGISTRY[id(authority)]
        if case == "request_nonce":
            original = authority.nonce
            object.__setattr__(authority, "nonce", object())
        else:
            original = truth_registration
            qualification._TRUTH_AUTHORITY_REGISTRY[id(authority)] = (
                authority,
                object(),
                request,
                "consumed",
                None,
            )
        with pytest.raises(PermissionError):
            qualification._complete_materializer_private_consumption(
                port,
                vault,
                request,
            )
        _assert_materializer_vault_scrubbed(vault)
        if case == "request_nonce":
            object.__setattr__(authority, "nonce", original)
        else:
            qualification._TRUTH_AUTHORITY_REGISTRY[id(authority)] = original
        manifest.abort()
        assert qualification._TRUTH_AUTHORITY_REGISTRY[id(authority)][3] == "revoked"


@pytest.mark.parametrize(
    "case",
    (
        "forged_return",
        "root_latch",
        "ledger_cut",
        "port_owner",
        "port_alias",
        "sensor_alias",
        "slot_alias",
    ),
)
def test_materializer_private_bind_reopens_after_the_real_consume_window(
    case: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker=f"private-consume-window-{case}",
    ) as (ledger, _pin, config):
        _authority, _evaluator, port = _construct_synthetic_materializer(ledger, config)
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        vault = qualification._open_materializer_split_vault(port, split="development")
        _evaluations, _receipts, seal = _materializer_fill_and_seal_public_split(
            manifest=manifest,
            port=port,
            vault=vault,
        )
        request = manifest.begin_private_scoring(public_seal=seal)
        root = qualification._MATERIALIZER_SPLIT_VAULT_REGISTRY[id(vault)]
        original_root_rows = tuple(root.rows)
        original_root_binding = root.binding
        port_registration = qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY[id(port)]
        original_consume = qualification.PrivateSplitScoringRequest.consume
        original_generation = ledger._record["generation"]
        ledger_bytes = ledger.path.read_bytes()
        ledger_identity = qualification._file_identity(ledger.path.stat())
        consumed_witness: list[str] = []

        def consume_then_mutate(
            candidate: qualification.PrivateSplitScoringRequest,
        ) -> dict[str, Any]:
            binding = original_consume(candidate)
            truth_registration = qualification._TRUTH_AUTHORITY_REGISTRY[id(candidate._authority)]
            assert truth_registration[3] == "consumed"
            consumed_witness.append(truth_registration[3])
            if case == "forged_return":
                forged = qualification.copy.deepcopy(binding)
                forged["ledger_generation"] += 1
                return forged
            if case == "root_latch":
                root.state = "public_sealed"
                root.rows.pop()
                root.binding = (*root.binding[:-1], object())
            elif case == "ledger_cut":
                ledger._record["generation"] += 1
            elif case == "port_owner":
                port_registration.ledger = object()  # type: ignore[assignment]
            elif case == "port_alias":
                alias_port_registration = qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY[
                    id(port)
                ]
                qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY[-301] = alias_port_registration
            elif case == "sensor_alias":
                sensor = root.sensor_batches[0]
                sensor_registration = qualification._PUBLIC_SENSOR_BATCH_REGISTRY[id(sensor)]
                qualification._PUBLIC_SENSOR_BATCH_REGISTRY[-302] = sensor_registration
            else:
                qualification._MATERIALIZER_SPLIT_VAULT_SLOT_REGISTRY[
                    (id(ledger) + 1, "development")
                ] = id(vault)
            return binding

        monkeypatch.setattr(
            qualification.PrivateSplitScoringRequest,
            "consume",
            consume_then_mutate,
        )
        with pytest.raises((PermissionError, ValueError)):
            qualification._bind_materializer_private_request(port, vault, request)
        assert consumed_witness == ["consumed"]
        if case == "ledger_cut":
            ledger._record["generation"] = original_generation
        if case == "port_owner":
            port_registration.ledger = ledger
            qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY[id(port)] = port_registration
        assert ledger.path.read_bytes() == ledger_bytes
        assert qualification._file_identity(ledger.path.stat()) == ledger_identity
        _assert_materializer_vault_scrubbed(vault)
        assert port_registration.state == "failed"
        with pytest.raises(PermissionError):
            qualification._bind_materializer_private_request(port, vault, request)
        manifest.abort()
        assert qualification._TRUTH_AUTHORITY_REGISTRY[id(request._authority)][3] == "revoked"
        if case == "root_latch":

            def restore_root_latch() -> None:
                root.rows = list(original_root_rows)
                root.binding = original_root_binding

            _assert_materializer_cleanup_refusal_then_restore(restore_root_latch)
        else:
            assert qualification._exact_materializer_terminal_tombstone_cut()


@pytest.mark.parametrize(
    "case",
    ("sensor", "row_order", "child_state", "digest", "evaluation_hash"),
)
def test_materializer_private_completion_reopens_exact_commitment_inventory(
    case: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker=f"private-completion-inventory-{case}",
    ) as (ledger, _pin, config):
        _authority, _evaluator, port = _construct_synthetic_materializer(ledger, config)
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        vault = qualification._open_materializer_split_vault(port, split="development")
        _evaluations, _receipts, seal = _materializer_fill_and_seal_public_split(
            manifest=manifest,
            port=port,
            vault=vault,
        )
        request = manifest.begin_private_scoring(public_seal=seal)
        qualification._bind_materializer_private_request(port, vault, request)
        registration = qualification._MATERIALIZER_SPLIT_VAULT_REGISTRY[id(vault)]
        original_rows = tuple(registration.rows)
        first_envelope = registration.envelopes[0]
        envelope_registration = qualification._PUBLIC_MATERIALIZATION_ENVELOPE_REGISTRY[
            id(first_envelope)
        ]
        if case == "sensor":
            sensor_batch = registration.sensor_batches[0]
            tensors = qualification._PUBLIC_SENSOR_BATCH_REGISTRY[id(sensor_batch)].tensors
            assert tensors is not None
            tensors["rgb"][0, 0, 0, 0, 0] += 0.01
        elif case == "row_order":
            registration.rows[0], registration.rows[1] = (
                registration.rows[1],
                registration.rows[0],
            )
        elif case == "child_state":
            envelope_registration.state = "evaluated"
        elif case == "digest":
            registration.commitment_inventory_sha256 = "f" * 64
        else:
            assert envelope_registration.evaluation_hashes is not None
            envelope_registration.evaluation_hashes["public_evidence_sha256"] = "f" * 64
            envelope_registration.evaluation_hashes_binding_sha256 = qualification.canonical_sha256(
                envelope_registration.evaluation_hashes
            )
        with pytest.raises((PermissionError, ValueError)):
            qualification._complete_materializer_private_consumption(
                port,
                vault,
                request,
            )
        _assert_materializer_vault_scrubbed(vault)
        assert qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY[id(port)].state == "failed"
        manifest.abort()
        if case == "row_order":
            _assert_materializer_cleanup_refusal_then_restore(
                lambda: setattr(registration, "rows", list(original_rows))
            )
        else:
            assert qualification._exact_materializer_terminal_tombstone_cut()


@pytest.mark.parametrize("ordinals", ((1, 0, 2, 3), (4, 5, 6, 7)))
def test_materializer_public_request_gap_or_reorder_fails_before_consumption(
    ordinals: tuple[int, int, int, int],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker=f"request-order-{ordinals[0]}",
    ) as (ledger, _pin, config):
        _authority, evaluator, port = _construct_synthetic_materializer(ledger, config)
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        vault = qualification._open_materializer_split_vault(port, split="development")
        batch = manifest.begin_batch((0, 1, 2, 3))
        request = qualification.PublicBatchEvaluationRequest(
            split="development",
            ordinals=batch.ordinals,
            tokens=batch.tokens,
            _manifest=manifest,
            _batch=batch,
        )
        object.__setattr__(request, "ordinals", ordinals)
        with pytest.raises(PermissionError, match="request/vault/ledger"):
            qualification._issue_public_materialization_envelope(
                port,
                vault,
                request,
                sensor_tensors=_materializer_sensor_tensors(),
                safe_task_tensors=_materializer_safe_task_tensors(),
                public_runtime_tensors=_materializer_public_runtime_tensors(),
            )
        assert qualification._CONSUMED_ORDINAL_REGISTRY == {}
        assert qualification._MATERIALIZER_SPLIT_VAULT_REGISTRY[id(vault)].state == (
            "collecting_public"
        )
        assert qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY[id(port)].state == "active"
        object.__setattr__(request, "ordinals", batch.ordinals)
        envelope = qualification._issue_public_materialization_envelope(
            port,
            vault,
            request,
            sensor_tensors=_materializer_sensor_tensors(),
            safe_task_tensors=_materializer_safe_task_tensors(),
            public_runtime_tensors=_materializer_public_runtime_tensors(),
        )
        assert type(envelope) is qualification._PublicMaterializationEnvelope
        evaluator.abort_split("development")
        manifest.abort()


@pytest.mark.parametrize(
    "case",
    ("sensor", "action", "entropy", "authorize", "authorize_registry_loss"),
)
def test_materializer_issue_post_request_failures_burn_and_scrub_nonretryably(
    case: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker=f"issue-fail-burn-{case}",
    ) as (ledger, _pin, config):
        _authority, evaluator, port = _construct_synthetic_materializer(ledger, config)
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        vault = qualification._open_materializer_split_vault(port, split="development")
        batch = manifest.begin_batch((0, 1, 2, 3))
        request = qualification.PublicBatchEvaluationRequest(
            split="development",
            ordinals=batch.ordinals,
            tokens=batch.tokens,
            _manifest=manifest,
            _batch=batch,
        )
        sensors = _materializer_sensor_tensors()
        tasks = _materializer_safe_task_tensors()
        runtime = _materializer_public_runtime_tensors()
        before_bytes = ledger.path.read_bytes()
        before_identity = ledger._last_identity
        before_record = qualification.copy.deepcopy(ledger.record)
        batch_registration = qualification._BATCH_REGISTRY[id(batch)]
        authorize_witness: list[tuple[int, ...]] = []
        if case == "sensor":
            sensors["rgb"][0, 0, 0, 0, 0] = 2.0
        elif case == "action":
            tasks["candidate_impulses_world"].fill_(1.0)
        elif case == "entropy":
            monkeypatch.setattr(
                qualification,
                "_generate_materializer_blinding_nonces",
                lambda: (_ for _ in ()).throw(RuntimeError("synthetic entropy failure")),
            )
        else:
            original_authorize = qualification._authorize_formal_batch

            def authorize_then_fail(
                candidate: qualification.PublicBatchEvaluationRequest,
            ) -> None:
                recognized = original_authorize(candidate)
                authorize_witness.append(tuple(item.ordinal for item in recognized))
                assert len(qualification._CONSUMED_ORDINAL_REGISTRY) == (qualification.BATCH_SIZE)
                if case == "authorize_registry_loss":
                    qualification._BATCH_REGISTRY.pop(id(batch), None)
                    for token in batch.tokens:
                        qualification._TOKEN_REGISTRY.pop(id(token), None)
                raise RuntimeError("synthetic post-authorize failure")

            monkeypatch.setattr(
                qualification,
                "_authorize_formal_batch",
                authorize_then_fail,
            )
        with pytest.raises((PermissionError, RuntimeError, ValueError)):
            qualification._issue_public_materialization_envelope(
                port,
                vault,
                request,
                sensor_tensors=sensors,
                safe_task_tensors=tasks,
                public_runtime_tensors=runtime,
            )
        registration = qualification._MATERIALIZER_SPLIT_VAULT_REGISTRY[id(vault)]
        assert ledger.path.read_bytes() == before_bytes
        assert ledger._last_identity == before_identity
        assert ledger.record == before_record
        if case in {"authorize", "authorize_registry_loss"}:
            assert authorize_witness == [(0, 1, 2, 3)]
        assert qualification._CONSUMED_ORDINAL_REGISTRY == {}
        assert qualification._BATCH_REGISTRY == {}
        assert qualification._TOKEN_REGISTRY == {}
        assert qualification._MATERIALIZER_PUBLIC_BATCH_EVALUATION_REGISTRY == {}
        assert batch_registration.status == "revoked"
        assert manifest._active is None
        assert registration.state == "failed"
        assert qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY[id(port)].state == "failed"
        _assert_materializer_vault_scrubbed(vault)
        with pytest.raises(PermissionError):
            qualification._issue_public_materialization_envelope(
                port,
                vault,
                request,
                sensor_tensors=_materializer_sensor_tensors(),
                safe_task_tensors=_materializer_safe_task_tensors(),
                public_runtime_tensors=_materializer_public_runtime_tensors(),
            )
        with suppress(BaseException):
            evaluator.abort_split("development")
        manifest.abort()


@pytest.mark.parametrize("state", sorted(qualification.MATERIALIZER_LIVE_VAULT_STATES))
def test_materializer_abort_scrubs_every_declared_live_vault_state(
    state: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker=f"abort-live-state-{state}",
    ) as (ledger, _pin, config):
        _authority, evaluator, port = _construct_synthetic_materializer(ledger, config)
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        vault = qualification._open_materializer_split_vault(port, split="development")
        _request, _envelope, _evaluation = _materializer_issue_public_batch(
            manifest=manifest,
            port=port,
            vault=vault,
            start=0,
        )
        qualification._MATERIALIZER_SPLIT_VAULT_REGISTRY[id(vault)].state = state
        # Some named live states require a later durable/child phase than this
        # issued-envelope fixture.  Abort must scrub them all, while reporting
        # a phase-inconsistent injected state when appropriate.
        with suppress(PermissionError):
            evaluator.abort_split("development")
        _assert_materializer_vault_scrubbed(vault)
        assert qualification._MATERIALIZER_SPLIT_VAULT_REGISTRY[id(vault)].state in {
            "failed",
            "revoked",
        }
        manifest.abort()


def test_materializer_blinding_nonce_generation_is_internal_distinct_and_captured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert "blinding_nonces" not in (
        qualification._issue_public_materialization_envelope.__code__.co_varnames
    )
    captured = qualification._generate_materializer_blinding_nonces.__kwdefaults__["_urandom"]

    def forbidden(_size: int) -> bytes:
        raise AssertionError("mutable os.urandom binding was used")

    with monkeypatch.context() as scoped:
        scoped.setattr(qualification.os, "urandom", forbidden)
        values = qualification._generate_materializer_blinding_nonces()
    assert (
        qualification._generate_materializer_blinding_nonces.__kwdefaults__["_urandom"] is captured
    )
    assert len(values) == len(set(values)) == qualification.BATCH_SIZE
    assert all(type(value) is bytes and len(value) == 32 for value in values)
    with pytest.raises(ValueError, match="four distinct"):
        qualification._validated_materializer_blinding_nonces(None)


@pytest.mark.parametrize(
    "case",
    (
        "noncontiguous",
        "requires_grad",
        "oversized_storage",
        "full_span_view",
        "frombuffer",
        "from_numpy",
        "shared_storage",
        "tensor_subclass",
    ),
)
def test_materializer_tensor_validator_rejects_nonowned_or_hookable_storage(
    case: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sensors = _materializer_sensor_tensors()
    rgb = sensors["rgb"]
    if case == "noncontiguous":
        sensors["rgb"] = rgb.transpose(-1, -2)
    elif case == "requires_grad":
        sensors["rgb"] = rgb.requires_grad_(True)
    elif case == "oversized_storage":
        backing = qualification.torch.zeros(
            rgb.numel() + 1,
            dtype=qualification.torch.float32,
        )
        sensors["rgb"] = backing[: rgb.numel()].reshape(rgb.shape)
    elif case == "full_span_view":
        sensors["rgb"] = rgb.view(rgb.shape)
    elif case == "frombuffer":
        backing = bytearray(rgb.numel() * rgb.element_size())
        sensors["rgb"] = qualification.torch.frombuffer(
            backing,
            dtype=qualification.torch.float32,
        ).reshape(rgb.shape)
    elif case == "from_numpy":
        sensors["rgb"] = qualification.torch.from_numpy(
            np.zeros(tuple(rgb.shape), dtype=np.float32)
        )
    elif case == "shared_storage":
        try:
            sensors["rgb"] = rgb.share_memory_()
        except RuntimeError as error:
            if "Operation not permitted" not in str(error):
                raise
            target_pointer = rgb.untyped_storage().data_ptr()
            original_is_shared = qualification.torch.UntypedStorage.is_shared
            monkeypatch.setattr(
                qualification.torch.UntypedStorage,
                "is_shared",
                lambda storage: storage.data_ptr() == target_pointer or original_is_shared(storage),
            )
    else:

        class TensorSubclass(qualification.torch.Tensor):
            pass

        sensors["rgb"] = rgb.as_subclass(TensorSubclass)
    with pytest.raises((TypeError, ValueError)):
        qualification._validated_public_sensor_tensors(sensors)


def test_materializer_clones_are_fresh_full_span_nonshared_and_sibling_disjoint() -> None:
    source = _materializer_sensor_tensors()
    clones = qualification._clone_materializer_tensor_mapping(source)
    qualification._validated_public_sensor_tensors(clones)
    for value in clones.values():
        storage = value.untyped_storage()
        assert value._base is None
        assert value.storage_offset() == 0
        assert storage.resizable() is True
        assert storage.is_shared() is False
        assert storage.nbytes() == value.numel() * value.element_size()
    qualification._materializer_tensor_storage_inventory(
        clones,
        label="fresh materializer clones",
    )


def test_materializer_retained_cut_isolated_from_hidden_detach_and_set_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker="retained-cut-hidden-source-aliases",
    ) as (ledger, _pin, config):
        _authority, evaluator, port = _construct_synthetic_materializer(ledger, config)
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        vault = qualification._open_materializer_split_vault(port, split="development")
        batch = manifest.begin_batch((0, 1, 2, 3))
        request = qualification.PublicBatchEvaluationRequest(
            split="development",
            ordinals=batch.ordinals,
            tokens=batch.tokens,
            _manifest=manifest,
            _batch=batch,
        )
        sensors = _materializer_sensor_tensors()
        tasks = _materializer_safe_task_tensors()
        runtime = _materializer_public_runtime_tensors()
        detached_alias = sensors["rgb"].detach()
        set_alias = qualification.torch.empty(
            0,
            dtype=tasks["candidate_impulses_world"].dtype,
        ).set_(
            tasks["candidate_impulses_world"].untyped_storage(),
            tasks["candidate_impulses_world"].storage_offset(),
            tasks["candidate_impulses_world"].size(),
            tasks["candidate_impulses_world"].stride(),
        )
        assert detached_alias.untyped_storage().data_ptr() == (
            sensors["rgb"].untyped_storage().data_ptr()
        )
        assert set_alias.untyped_storage().data_ptr() == (
            tasks["candidate_impulses_world"].untyped_storage().data_ptr()
        )
        source_rgb_before = sensors["rgb"].clone()
        source_impulses_before = tasks["candidate_impulses_world"].clone()
        envelope = qualification._issue_public_materialization_envelope(
            port,
            vault,
            request,
            sensor_tensors=sensors,
            safe_task_tensors=tasks,
            public_runtime_tensors=runtime,
        )
        root = qualification._MATERIALIZER_SPLIT_VAULT_REGISTRY[id(vault)]
        envelope_registration = qualification._PUBLIC_MATERIALIZATION_ENVELOPE_REGISTRY[
            id(envelope)
        ]
        sensor_batch = envelope_registration.sensor_batch
        task_batch = envelope_registration.task_batch
        assert sensor_batch is not None and task_batch is not None
        retained_sensors = qualification._PUBLIC_SENSOR_BATCH_REGISTRY[id(sensor_batch)].tensors
        retained_tasks = qualification._SAFE_TASK_BATCH_REGISTRY[id(task_batch)].tensors
        retained_runtime = envelope_registration.runtime_tensors
        assert retained_sensors is not None
        assert retained_tasks is not None
        assert retained_runtime is not None
        retained_snapshots = (
            {f"sensor:{key}": value.clone() for key, value in retained_sensors.items()}
            | {f"task:{key}": value.clone() for key, value in retained_tasks.items()}
            | {f"runtime:{key}": value.clone() for key, value in retained_runtime.items()}
        )
        public_bodies = qualification.copy.deepcopy(envelope_registration.public_bodies)
        commitments = envelope.blinded_commitments
        detached_alias.zero_()
        set_alias.zero_()
        assert not qualification.torch.equal(sensors["rgb"], source_rgb_before)
        assert not qualification.torch.equal(
            tasks["candidate_impulses_world"],
            source_impulses_before,
        )
        source_storage = {
            value.untyped_storage().data_ptr()
            for mapping in (sensors, tasks, runtime)
            for value in mapping.values()
        }
        retained_storage = {
            value.untyped_storage().data_ptr()
            for mapping in (retained_sensors, retained_tasks, retained_runtime)
            for value in mapping.values()
        }
        assert source_storage.isdisjoint(retained_storage)
        for key, retained in (
            {
                **{f"sensor:{name}": value for name, value in retained_sensors.items()},
                **{f"task:{name}": value for name, value in retained_tasks.items()},
                **{f"runtime:{name}": value for name, value in retained_runtime.items()},
            }
        ).items():
            assert qualification.torch.equal(retained, retained_snapshots[key])
        assert envelope_registration.public_bodies == public_bodies
        assert envelope.blinded_commitments == commitments
        qualification._materializer_tensor_storage_inventory(
            {
                **{f"source-sensor:{name}": value for name, value in sensors.items()},
                **{f"source-task:{name}": value for name, value in tasks.items()},
                **{f"source-runtime:{name}": value for name, value in runtime.items()},
                **{f"retained-sensor:{name}": value for name, value in retained_sensors.items()},
                **{f"retained-task:{name}": value for name, value in retained_tasks.items()},
                **{f"retained-runtime:{name}": value for name, value in retained_runtime.items()},
            },
            label="source and retained materializer tensor cut",
        )
        qualification._validated_open_materializer_envelope(
            port,
            root,
            envelope_registration,
        )
        evaluator.abort_split("development")
        manifest.abort()


def test_materializer_copy_cut_rejects_a_storage_sharing_clone_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        qualification,
        "_clone_materializer_tensor_mapping",
        lambda mapping: dict(mapping),
    )
    with pytest.raises(PermissionError, match="alias storage"):
        qualification._validated_materializer_issue_payload(
            split="development",
            ordinals=(0, 1, 2, 3),
            sensor_tensors=_materializer_sensor_tensors(),
            safe_task_tensors=_materializer_safe_task_tensors(),
            public_runtime_tensors=_materializer_public_runtime_tensors(),
        )


def test_materializer_safe_impulse_and_runtime_selection_are_exactly_guarded() -> None:
    tasks = _materializer_safe_task_tensors()
    tasks["candidate_impulses_world"] = qualification.torch.ones_like(
        tasks["candidate_impulses_world"]
    )
    with pytest.raises(ValueError, match="sign pairs"):
        qualification._validated_safe_task_batch_tensors(tasks)
    runtime = _materializer_public_runtime_tensors()
    runtime["candidate_total_cost"] = qualification.torch.arange(
        qualification.CANDIDATE_COUNT,
        dtype=qualification.torch.float32,
    ).repeat(qualification.BATCH_SIZE, 1)
    runtime["selected_display_index"].fill_(7)
    with pytest.raises(ValueError, match="planner output"):
        qualification._validated_public_runtime_tensors(runtime)


def test_cleanup_refuses_live_formal_materializer_without_mutating_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker="cleanup-live-materializer",
    ) as (ledger, _pin, config):
        _authority, evaluator, port = _construct_synthetic_materializer(ledger, config)
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        vault = qualification._open_materializer_split_vault(port, split="development")
        _request, envelope, _evaluation, _hashes = _materializer_issue_and_register_public_batch(
            manifest=manifest,
            port=port,
            vault=vault,
            start=0,
        )
        envelope_registration = qualification._PUBLIC_MATERIALIZATION_ENVELOPE_REGISTRY[
            id(envelope)
        ]
        assert id(envelope) in qualification._PUBLIC_MATERIALIZATION_ENVELOPE_REGISTRY
        assert any(
            registration[2] is envelope
            for registration in (
                qualification._MATERIALIZER_PUBLIC_BATCH_EVALUATION_REGISTRY.values()
            )
        )
        # Isolate the cleanup oracle's materializer clause: earlier formal
        # invocation/runner authorities are temporarily hidden, then restored
        # before this context's targeted teardown runs.
        earlier_formal_registries = (
            "_INVOCATION_SEAL_REGISTRY",
            "_RUN_AUTHORIZATION_REGISTRY",
            "_OUTER_RUNNER_AUTHORITY_REGISTRY",
            "_OUTER_RECEIPT_IDENTITY_REGISTRY",
            "_OUTER_AUTHORIZATION_IDENTITY_REGISTRY",
            "_FORMAL_PIN_AUTHORITY_REGISTRY",
            "_LEDGER_RECOVERY_AUTHORIZATION_REGISTRY",
            "_EVALUATOR_CONSTRUCTION_AUTHORITY_REGISTRY",
            "_MATERIALIZER_CONSTRUCTION_AUTHORITY_REGISTRY",
            "_FORMAL_EVALUATOR_CONSTRUCTION_IN_PROGRESS",
            "_MATERIALIZER_CONSTRUCTION_IN_PROGRESS",
        )
        with monkeypatch.context() as isolated:
            for name in earlier_formal_registries:
                isolated.setattr(qualification, name, {})
            with pytest.raises(PermissionError, match="cannot mutate formal"):
                qualification._clear_ephemeral_registries_for_tests()
        assert envelope_registration.runtime_tensors is not None
        assert envelope_registration.public_bodies is not None
        assert envelope_registration.blinding_nonces is not None
        evaluator.abort_split("development")
        manifest.abort()
        _assert_materializer_vault_scrubbed(vault)


@pytest.mark.parametrize("case", ("root_binding", "child_binding", "slot"))
def test_cleanup_refuses_crossbound_terminal_materializer_tombstone(
    case: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _formal_materializer_ledger(
        tmp_path,
        monkeypatch,
        marker=f"cleanup-terminal-crossbind-{case}",
    ) as (ledger, _pin, config):
        _authority, evaluator, port = _construct_synthetic_materializer(ledger, config)
        manifest = qualification._ManifestCapability(split="development", ledger=ledger)
        vault = qualification._open_materializer_split_vault(port, split="development")
        _request, envelope, _evaluation = _materializer_issue_public_batch(
            manifest=manifest,
            port=port,
            vault=vault,
            start=0,
        )
        evaluator.abort_split("development")
        manifest.abort()
        _assert_materializer_vault_scrubbed(vault)
        root = qualification._MATERIALIZER_SPLIT_VAULT_REGISTRY[id(vault)]
        envelope_registration = qualification._PUBLIC_MATERIALIZATION_ENVELOPE_REGISTRY[
            id(envelope)
        ]
        slot = (id(ledger), "development")
        assert qualification._exact_materializer_terminal_tombstone_cut() is True
        if case == "root_binding":
            original: Any = root.binding
            root.binding = (id(port) + 1, *root.binding[1:])
        elif case == "child_binding":
            original = envelope_registration.binding
            envelope_registration.binding = (
                id(port) + 1,
                *envelope_registration.binding[1:],
            )
        else:
            original = qualification._MATERIALIZER_SPLIT_VAULT_SLOT_REGISTRY[slot]
            qualification._MATERIALIZER_SPLIT_VAULT_SLOT_REGISTRY[slot] = id(vault) + 1
        registry_sizes = (
            len(qualification._MATERIALIZER_CONSTRUCTION_AUTHORITY_REGISTRY),
            len(qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY),
            len(qualification._MATERIALIZER_SPLIT_VAULT_REGISTRY),
            len(qualification._PUBLIC_MATERIALIZATION_ENVELOPE_REGISTRY),
        )
        assert qualification._exact_materializer_terminal_tombstone_cut() is False
        earlier_formal_registries = (
            "_INVOCATION_SEAL_REGISTRY",
            "_RUN_AUTHORIZATION_REGISTRY",
            "_OUTER_RUNNER_AUTHORITY_REGISTRY",
            "_OUTER_RECEIPT_IDENTITY_REGISTRY",
            "_OUTER_AUTHORIZATION_IDENTITY_REGISTRY",
            "_FORMAL_PIN_AUTHORITY_REGISTRY",
            "_LEDGER_RECOVERY_AUTHORIZATION_REGISTRY",
            "_FORMAL_EVALUATOR_CONSTRUCTION_IN_PROGRESS",
            "_MATERIALIZER_CONSTRUCTION_IN_PROGRESS",
            "_LEDGER_REGISTRY",
            "_LEDGER_SLOT_REGISTRY",
        )
        with monkeypatch.context() as isolated:
            for name in earlier_formal_registries:
                isolated.setattr(qualification, name, {})
            with pytest.raises(PermissionError, match="cannot mutate formal"):
                qualification._clear_ephemeral_registries_for_tests()
        assert registry_sizes == (
            len(qualification._MATERIALIZER_CONSTRUCTION_AUTHORITY_REGISTRY),
            len(qualification._TRUSTED_MATERIALIZER_PORT_REGISTRY),
            len(qualification._MATERIALIZER_SPLIT_VAULT_REGISTRY),
            len(qualification._PUBLIC_MATERIALIZATION_ENVELOPE_REGISTRY),
        )
        if case == "root_binding":
            root.binding = original
        elif case == "child_binding":
            envelope_registration.binding = original
        else:
            qualification._MATERIALIZER_SPLIT_VAULT_SLOT_REGISTRY[slot] = original
        assert qualification._exact_materializer_terminal_tombstone_cut() is True
