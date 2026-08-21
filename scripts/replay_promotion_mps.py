#!/usr/bin/env python3
"""Replay a candidate/reference selector pair on the active-Aqua MPS backend.

This is deliberately a promotion gate rather than a generic evaluation tool.
It reruns the trainer's exact fixed validation manifest for both immutable
checkpoints, derives the existing physical selector metrics from raw additive
evidence, and applies the same lifecycle, identity, event, horizon, axis, and
calibration guardrails used during training.  It never mutates a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from world_model.datasets.splits import make_seed_manifest
from world_model.evaluation.evaluator import (
    _EVALUATION_METRIC_SCHEMA_VERSION,
    _EVALUATION_PROTOCOL_SCHEMA_VERSION,
    _PER_SCENARIO_METRIC_SCHEMA,
    _capture_checkpoint_snapshot,
    _primary_physical_metrics,
    _primary_physical_metrics_hash_exclusion_declaration,
)
from world_model.evaluation.latency import paired_latency_guardrail
from world_model.runtime import OnlineWorldModel
from world_model.training.checkpointing import capture_git_metadata, load_model_weights
from world_model.training.trainer import (
    _handoff_training_support_failures,
    _make_loader,
    _mutable_causal_training_support_failures,
    _rollout_selection_guardrail_failures,
    _rollout_selection_improves,
    _rollout_selection_metrics,
    _rollout_validation_protocol_from_mapping,
    _rollout_validation_protocol_hash,
    _selection_horizon_keys,
    _validate_validation_support_schema,
    _validation_loader_result,
)
from world_model.utils.artifacts import timestamped_artifact_path
from world_model.utils.config import OrpheusConfig, load_config
from world_model.utils.device import select_device
from world_model.utils.io import atomic_write_text
from world_model.utils.seeds import seed_everything


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_json_value(value: Any) -> Any:
    """Normalize tuples and other JSON-compatible containers as reports do."""

    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


@dataclass(frozen=True)
class _CapturedJsonReport:
    """One report parsed and hashed from the same single immutable byte read."""

    source_path: Path
    payload: dict[str, Any]
    sha256: str
    byte_count: int


def _capture_json_report(path: Path, *, description: str) -> _CapturedJsonReport:
    source = path.expanduser().resolve()
    try:
        with source.open("rb") as handle:
            content = handle.read()
        value = json.loads(content.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is not a readable JSON report: {source}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object: {source}")
    return _CapturedJsonReport(
        source_path=source,
        payload=value,
        sha256=hashlib.sha256(content).hexdigest(),
        byte_count=len(content),
    )


def _physical_replay_contract(
    config: OrpheusConfig,
    *,
    expected_device: str,
    expected_precision: str,
    runtime_source_fingerprint: str,
) -> dict[str, Any]:
    """Resolve the exact physical replay/evaluator evidence boundary."""

    if config.evaluation.episodes != config.training.validation_episodes:
        raise ValueError(
            "promotion replay requires evaluation.episodes to equal training.validation_episodes"
        )
    if not runtime_source_fingerprint:
        raise ValueError("promotion replay requires a current runtime source fingerprint")
    manifest = make_seed_manifest("validation", config.training.validation_episodes)
    seeds = list(manifest.seeds)
    scenarios = list(config.simulator.scenario_mixture)
    if not scenarios:
        raise ValueError("promotion replay requires at least one validation scenario")
    episode_scenarios = [scenarios[int(seed) % len(scenarios)] for seed in seeds]
    horizon_grid = [suffix for suffix, _ in _selection_horizon_keys(config)]
    rollout_protocol = _rollout_validation_protocol_from_mapping(config.to_dict())
    evaluator_batch_size = min(config.training.batch_size, config.evaluation.episodes)
    evaluator_batch_count = math.ceil(config.evaluation.episodes / evaluator_batch_size)
    evaluator_protocol = {
        "schema_version": _EVALUATION_PROTOCOL_SCHEMA_VERSION,
        "metric_schema_version": _EVALUATION_METRIC_SCHEMA_VERSION,
        "per_scenario_metric_schema": _PER_SCENARIO_METRIC_SCHEMA,
        # Filled per arm from its immutable captured bytes.
        "checkpoint_sha256": None,
        "resolved_config_sha256": _canonical_sha256(config.to_dict()),
        "split": "validation",
        "seed_protocol": "standard",
        "seed_manifest": seeds,
        "horizons_seconds_requested": list(config.evaluation.horizons_seconds),
        "horizons_observation_grid": horizon_grid,
        "batch_size": evaluator_batch_size,
        "episode_count": config.evaluation.episodes,
        "runtime_intervention": {
            "evaluator_state_perturbation_in_primary": False,
            "runtime_hypothesis_pool": False,
            "recovery_probe_enabled": False,
            "recovery_probe_position_std": config.evaluation.perturbation_position_std,
            "recovery_probe_velocity_std": config.evaluation.perturbation_velocity_std,
        },
    }
    return {
        "resolved_config_sha256": _canonical_sha256(config.to_dict()),
        "rollout_validation_protocol_hash": _rollout_validation_protocol_hash(config),
        "rollout_validation_protocol": rollout_protocol,
        "validation_split": "validation",
        "validation_seed_manifest": seeds,
        "validation_episode_count": config.training.validation_episodes,
        "validation_batch_size": 1,
        "scenario_mixture": scenarios,
        "resolved_scenarios": _canonical_json_value(rollout_protocol["resolved_scenarios"]),
        "evaluation_episode_scenarios": episode_scenarios,
        "horizons_observation_grid": horizon_grid,
        "horizons_seconds_requested": list(config.evaluation.horizons_seconds),
        "evaluator_batch_size": evaluator_batch_size,
        "evaluator_batch_count": evaluator_batch_count,
        "primary_posterior_trace_frame_count": (
            config.evaluation.episodes * config.simulator.sequence_frames
        ),
        "runtime_source_fingerprint": runtime_source_fingerprint,
        "device": expected_device,
        "precision": expected_precision,
        "evaluator_protocol": evaluator_protocol,
    }


def _require_exact_report_value(
    mapping: Mapping[str, Any],
    key: str,
    expected: Any,
    *,
    role: str,
    scope: str,
) -> None:
    if mapping.get(key) != expected:
        raise ValueError(f"{role} latency report {scope} {key!r} does not match replay")


def _validate_primary_physical_metrics_hash(
    metadata: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    role: str,
) -> None:
    if metadata.get("primary_physical_metrics_hash_excludes") != (
        _primary_physical_metrics_hash_exclusion_declaration()
    ):
        raise ValueError(
            f"{role} latency report has an invalid primary physical hash exclusion scope"
        )
    if metadata.get("primary_physical_metrics_scope") != (
        "clean_primary_metrics_before_isolated_recovery_probe_append"
    ):
        raise ValueError(f"{role} latency report has an invalid primary physical hash scope")
    primary_metrics = _primary_physical_metrics(metrics)
    expected_keys = sorted(primary_metrics)
    if metadata.get("primary_physical_metrics_hashed_keys") != expected_keys:
        raise ValueError(f"{role} latency report primary physical hashed-key evidence is invalid")
    try:
        expected_hash = _canonical_sha256(primary_metrics)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{role} latency report primary physical metrics are not canonical JSON"
        ) from error
    if metadata.get("primary_physical_metrics_sha256") != expected_hash:
        raise ValueError(f"{role} latency report primary physical metrics hash is invalid")


def _validated_latency_evaluation_report(
    captured_report: _CapturedJsonReport,
    *,
    role: str,
    checkpoint_sha256: str,
    checkpoint_byte_count: int,
    replay_contract: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load one evaluator report and bind it to this exact replay input."""

    payload = captured_report.payload
    metadata = payload.get("metadata")
    metrics = payload.get("metrics")
    if not isinstance(metadata, dict) or not isinstance(metrics, dict):
        raise ValueError(f"{role} latency report requires metadata and metrics objects")
    protocol = metadata.get("resolved_evaluation_protocol")
    if not isinstance(protocol, dict):
        raise ValueError(f"{role} latency report lacks its resolved evaluation protocol")
    declared_protocol_hash = metadata.get("resolved_evaluation_protocol_sha256")
    if declared_protocol_hash != _canonical_sha256(protocol):
        raise ValueError(f"{role} latency report protocol hash is invalid")
    if metadata.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError(f"{role} latency report checkpoint identity does not match replay input")
    if protocol.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError(
            f"{role} latency report protocol checkpoint identity does not match replay input"
        )
    if metadata.get("checkpoint_byte_count") != checkpoint_byte_count:
        raise ValueError(f"{role} latency report checkpoint byte count does not match replay input")
    if metadata.get("checkpoint_identity_source") != (
        "captured_pre_evaluation_immutable_byte_snapshot"
    ):
        raise ValueError(f"{role} latency report checkpoint identity source is unsupported")
    expected_protocol = dict(replay_contract["evaluator_protocol"])
    expected_protocol["checkpoint_sha256"] = checkpoint_sha256
    if protocol != expected_protocol:
        raise ValueError(f"{role} latency report evaluation protocol does not match replay")
    expected_metadata = {
        "evaluation_metric_schema_version": _EVALUATION_METRIC_SCHEMA_VERSION,
        "resolved_evaluation_config_sha256": replay_contract["resolved_config_sha256"],
        "simulator_version": replay_contract["rollout_validation_protocol"]["simulator_version"],
        "evaluation_simulator_version": replay_contract["rollout_validation_protocol"][
            "simulator_version"
        ],
        "scenario_mixture": replay_contract["scenario_mixture"],
        "resolved_scenarios": replay_contract["resolved_scenarios"],
        "per_scenario_metrics_schema": _PER_SCENARIO_METRIC_SCHEMA,
        "per_scenario_metrics_scenarios": replay_contract["scenario_mixture"],
        "per_scenario_metrics_horizons": replay_contract["horizons_observation_grid"],
        "evaluation_episode_scenarios": replay_contract["evaluation_episode_scenarios"],
        "split": replay_contract["validation_split"],
        "episodes": replay_contract["validation_episode_count"],
        "batches": replay_contract["evaluator_batch_count"],
        "device": replay_contract["device"],
        "precision": replay_contract["precision"],
        "evaluation_seed_protocol": "standard",
        "evaluation_seed_role": "standard_validation_evaluation",
        "evaluation_seed_offset": 0,
        "evaluation_seed_count": replay_contract["validation_episode_count"],
        "evaluation_seed_first": replay_contract["validation_seed_manifest"][0],
        "evaluation_seed_last": replay_contract["validation_seed_manifest"][-1],
        "evaluation_episode_seeds": replay_contract["validation_seed_manifest"],
        "evaluation_seed_overlaps_training_validation": True,
        "evaluation_seed_overlaps_test_range": False,
        "primary_posterior_trace_frame_count": replay_contract[
            "primary_posterior_trace_frame_count"
        ],
        "primary_posterior_trace_schema": "world_belief_tensor_fields_v1",
        "recovery_probe_enabled": False,
        "runtime_hypothesis_pool_enabled": False,
        "evaluation_perturbations_applied": False,
    }
    for key, expected in expected_metadata.items():
        _require_exact_report_value(
            metadata,
            key,
            expected,
            role=role,
            scope="metadata",
        )
    source_provenance = metadata.get("evaluation_source_provenance")
    if (
        not isinstance(source_provenance, Mapping)
        or source_provenance.get("runtime_source_fingerprint")
        != replay_contract["runtime_source_fingerprint"]
    ):
        raise ValueError(f"{role} latency report runtime source fingerprint does not match replay")
    trace_hash = metadata.get("primary_posterior_trace_sha256")
    if not isinstance(trace_hash, str) or len(trace_hash) != 64:
        raise ValueError(f"{role} latency report primary posterior trace hash is invalid")
    if metadata.get("rgb_only") is not True:
        raise ValueError(f"{role} latency report is not an RGB-only evaluation")
    if metadata.get("oracle_runtime_input_used") is not False:
        raise ValueError(f"{role} latency report used or omitted the oracle-input declaration")
    if metadata.get("primary_online_pass_evaluator_state_perturbation_free") is not True:
        raise ValueError(f"{role} latency report primary pass is not perturbation-free")
    _validate_primary_physical_metrics_hash(metadata, metrics, role=role)
    try:
        nonfinite_count = float(metrics["nonfinite_output_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{role} latency report lacks finite-output support") from error
    if nonfinite_count != 0.0:
        raise ValueError(f"{role} latency report contains non-finite runtime outputs")
    try:
        evaluated_episodes = float(metrics["evaluated_episodes"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{role} latency report lacks evaluated-episode evidence") from error
    if (
        not math.isfinite(evaluated_episodes)
        or not evaluated_episodes.is_integer()
        or evaluated_episodes != replay_contract["validation_episode_count"]
    ):
        raise ValueError(f"{role} latency report evaluated-episode count does not match replay")
    expected_scenario_counts = {
        scenario: replay_contract["evaluation_episode_scenarios"].count(scenario)
        for scenario in replay_contract["scenario_mixture"]
    }
    for scenario, expected_count in expected_scenario_counts.items():
        try:
            actual_count = float(metrics[f"scenario_{scenario}_episode_count"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"{role} latency report lacks scenario episode count for {scenario!r}"
            ) from error
        if not actual_count.is_integer() or actual_count != expected_count:
            raise ValueError(f"{role} latency report scenario episode count does not match replay")
    for name in (
        "injected_perturbation_batch_updates",
        "recovery_probe_evaluated_episodes",
        "recovery_probe_nonfinite_output_count",
    ):
        try:
            value = float(metrics[name])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"{role} latency report lacks disabled-intervention evidence"
            ) from error
        if value != 0.0:
            raise ValueError(f"{role} latency report contains recovery intervention evidence")
    for prefix in ("rgb_global_update", "rgb_fast_update", "future_rollout"):
        try:
            mean = float(metrics[f"{prefix}_latency_mean_ms"])
            total = float(metrics[f"{prefix}_latency_sum_ms"])
            count = float(metrics[f"{prefix}_latency_sample_count"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{role} latency report lacks complete latency evidence") from error
        if (
            not math.isfinite(mean)
            or mean < 0.0
            or not math.isfinite(total)
            or total < 0.0
            or not math.isfinite(count)
            or count <= 0.0
            or not count.is_integer()
            or not math.isclose(total, mean * count, rel_tol=1.0e-9, abs_tol=1.0e-7)
        ):
            raise ValueError(f"{role} latency report has contradictory latency counts")
    return metadata, metrics


_MATCHED_LATENCY_METADATA_KEYS = (
    "evaluation_metric_schema_version",
    "evaluation_source_provenance",
    "scenario_mixture",
    "resolved_scenarios",
    "per_scenario_metrics_schema",
    "per_scenario_metrics_horizons",
    "evaluation_episode_scenarios",
    "split",
    "episodes",
    "device",
    "precision",
    "primary_posterior_trace_frame_count",
    "primary_posterior_trace_schema",
    "batches",
    "primary_physical_metrics_hashed_keys",
    "primary_physical_metrics_scope",
    "primary_physical_metrics_hash_excludes",
)


def _load_paired_latency_evidence(
    *,
    reference_report: Path,
    candidate_report: Path,
    reference_checkpoint_sha256: str,
    candidate_checkpoint_sha256: str,
    reference_checkpoint_byte_count: int,
    candidate_checkpoint_byte_count: int,
    replay_contract: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate independently measured reports as one matched latency pair."""

    captured_reference_report = _capture_json_report(
        reference_report,
        description="reference latency report",
    )
    captured_candidate_report = _capture_json_report(
        candidate_report,
        description="candidate latency report",
    )
    reference_metadata, reference_metrics = _validated_latency_evaluation_report(
        captured_reference_report,
        role="reference",
        checkpoint_sha256=reference_checkpoint_sha256,
        checkpoint_byte_count=reference_checkpoint_byte_count,
        replay_contract=replay_contract,
    )
    candidate_metadata, candidate_metrics = _validated_latency_evaluation_report(
        captured_candidate_report,
        role="candidate",
        checkpoint_sha256=candidate_checkpoint_sha256,
        checkpoint_byte_count=candidate_checkpoint_byte_count,
        replay_contract=replay_contract,
    )
    reference_protocol = dict(reference_metadata["resolved_evaluation_protocol"])
    candidate_protocol = dict(candidate_metadata["resolved_evaluation_protocol"])
    reference_protocol.pop("checkpoint_sha256", None)
    candidate_protocol.pop("checkpoint_sha256", None)
    if reference_protocol != candidate_protocol:
        raise ValueError(
            "candidate/reference latency reports do not share the same evaluation protocol"
        )
    mismatches = [
        key
        for key in _MATCHED_LATENCY_METADATA_KEYS
        if reference_metadata.get(key) != candidate_metadata.get(key)
    ]
    if mismatches:
        raise ValueError(
            "candidate/reference latency reports are not matched for: " + ", ".join(mismatches)
        )
    binding = {
        "reference_latency_report": str(captured_reference_report.source_path),
        "reference_latency_report_sha256": captured_reference_report.sha256,
        "reference_latency_report_byte_count": captured_reference_report.byte_count,
        "candidate_latency_report": str(captured_candidate_report.source_path),
        "candidate_latency_report_sha256": captured_candidate_report.sha256,
        "candidate_latency_report_byte_count": captured_candidate_report.byte_count,
        "matched_protocol_without_checkpoint_sha256": _canonical_sha256(reference_protocol),
        "device": replay_contract["device"],
        "precision": replay_contract["precision"],
        "runtime_source_fingerprint": replay_contract["runtime_source_fingerprint"],
        "resolved_config_sha256": replay_contract["resolved_config_sha256"],
        "rollout_validation_protocol_hash": replay_contract["rollout_validation_protocol_hash"],
        "seed_manifest": list(reference_protocol["seed_manifest"]),
        "evaluation_episode_scenarios": list(reference_metadata["evaluation_episode_scenarios"]),
        "horizons_observation_grid": list(reference_protocol.get("horizons_observation_grid", [])),
        "validation_batch_size": replay_contract["validation_batch_size"],
        "evaluator_batch_size": replay_contract["evaluator_batch_size"],
        "evaluator_batch_count": replay_contract["evaluator_batch_count"],
    }
    return candidate_metrics, reference_metrics, binding


def _promotion_eligibility(
    *,
    accuracy_improves: bool,
    reference_guardrail_failures: list[dict[str, Any]],
    training_support_failures: list[dict[str, Any]],
    mutable_support_failures: list[dict[str, Any]],
    latency_promotion_eligible: bool,
) -> tuple[bool, bool]:
    """Keep deterministic physical qualification separate from the cost gate."""

    physical = bool(
        accuracy_improves
        and not reference_guardrail_failures
        and not training_support_failures
        and not mutable_support_failures
    )
    return physical, physical and latency_promotion_eligible


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--reference",
        required=True,
        help="Protected incumbent/reference checkpoint to replay on MPS.",
    )
    parser.add_argument(
        "--candidate",
        required=True,
        help="Candidate checkpoint to replay on the same MPS manifest.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Report label/path; its basename receives a UTC timestamp prefix.",
    )
    parser.add_argument(
        "--reference-latency-report",
        help="Evaluator evaluation.json for the reference checkpoint on the matched MPS protocol.",
    )
    parser.add_argument(
        "--candidate-latency-report",
        help="Evaluator evaluation.json for the candidate checkpoint on the matched MPS protocol.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Validation workers; zero keeps the MPS replay maximally auditable.",
    )
    return parser.parse_args()


def _replay_checkpoint(
    checkpoint: Path,
    *,
    config: OrpheusConfig,
    device: torch.device,
    output: Path,
    role: str,
) -> tuple[Any, dict[str, Any]]:
    """Return selector metrics plus raw evidence from one isolated runtime."""

    model = OnlineWorldModel.from_config(config, device=device)
    load_model_weights(checkpoint, model=model, expected_config=config)
    model.eval()
    model.reset()
    loader = _make_loader(
        config,
        split="validation",
        episodes=config.training.validation_episodes,
        shuffle=False,
        batch_size_override=1,
    )
    validation = _validation_loader_result(
        model,
        loader,
        config,
        device=device,
        closed_loop=True,
        progress_path=output / f"{role}_validation_progress.json",
        progress_split=f"mps_{role}",
    )
    _validate_validation_support_schema(validation.metrics, config)
    selector = _rollout_selection_metrics(
        validation.metrics,
        config,
        require_scenarios=True,
    )
    raw_metrics = dict(validation.metrics)
    atomic_write_text(
        output / f"{role}_validation_metrics.json",
        json.dumps(raw_metrics, indent=2, sort_keys=True) + "\n",
    )
    return selector, raw_metrics


def _reset_physical_replay_rng(*, seed: int, deterministic: bool) -> None:
    """Restore every supported RNG to the same arm-local starting state."""

    seed_everything(seed, deterministic=deterministic)
    mps = getattr(torch, "mps", None)
    backend = getattr(torch.backends, "mps", None)
    if (
        mps is not None
        and backend is not None
        and backend.is_available()
        and hasattr(mps, "manual_seed")
    ):
        mps.manual_seed(seed)


def _replay_checkpoint_with_rng_reset(
    checkpoint: Path,
    *,
    config: OrpheusConfig,
    device: torch.device,
    output: Path,
    role: str,
) -> tuple[Any, dict[str, Any]]:
    """Start one physical arm from an identical complete RNG state."""

    _reset_physical_replay_rng(
        seed=config.project.seed,
        deterministic=config.project.deterministic,
    )
    return _replay_checkpoint(
        checkpoint,
        config=config,
        device=device,
        output=output,
        role=role,
    )


def main() -> int:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("--num-workers must be nonnegative")
    if bool(args.reference_latency_report) != bool(args.candidate_latency_report):
        raise ValueError(
            "--reference-latency-report and --candidate-latency-report must be supplied together"
        )
    config = load_config(
        args.config,
        overrides=[
            "device.preference=mps",
            "device.closed_loop_preference=mps",
            f"training.num_workers={args.num_workers}",
        ],
    )
    config.validate()
    device_info = select_device("mps")
    if device_info.device.type != "mps":
        raise RuntimeError("promotion replay requires an active Aqua MPS device")
    device = device_info.device
    source_provenance = capture_git_metadata(Path(__file__).resolve().parents[1])
    runtime_source_fingerprint = source_provenance.get("runtime_source_fingerprint")
    if not isinstance(runtime_source_fingerprint, str) or not runtime_source_fingerprint:
        raise RuntimeError("promotion replay could not capture current runtime source identity")
    replay_contract = _physical_replay_contract(
        config,
        expected_device=str(device),
        expected_precision=device_info.precision,
        runtime_source_fingerprint=runtime_source_fingerprint,
    )

    reference_path = Path(args.reference).expanduser().resolve()
    candidate_path = Path(args.candidate).expanduser().resolve()
    if reference_path == candidate_path:
        raise ValueError("reference and candidate must be distinct checkpoints")
    output = timestamped_artifact_path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)

    with (
        _capture_checkpoint_snapshot(reference_path) as captured_reference,
        _capture_checkpoint_snapshot(candidate_path) as captured_candidate,
    ):
        reference, _ = _replay_checkpoint_with_rng_reset(
            captured_reference.snapshot_path,
            config=config,
            device=device,
            output=output,
            role="reference",
        )
        candidate, _ = _replay_checkpoint_with_rng_reset(
            captured_candidate.snapshot_path,
            config=config,
            device=device,
            output=output,
            role="candidate",
        )
        reference_checkpoint_sha256 = captured_reference.sha256
        candidate_checkpoint_sha256 = captured_candidate.sha256
        reference_checkpoint_byte_count = captured_reference.byte_count
        candidate_checkpoint_byte_count = captured_candidate.byte_count
    reference_guardrail_failures = _rollout_selection_guardrail_failures(candidate, reference)
    training_support_failures = _handoff_training_support_failures(
        candidate,
        reference,
        config,
    )
    mutable_support_failures = _mutable_causal_training_support_failures(candidate, config)
    latency_binding: dict[str, Any] | None = None
    if args.reference_latency_report and args.candidate_latency_report:
        reference_latency_report = Path(args.reference_latency_report).expanduser().resolve()
        candidate_latency_report = Path(args.candidate_latency_report).expanduser().resolve()
        candidate_latency_metrics, reference_latency_metrics, latency_binding = (
            _load_paired_latency_evidence(
                reference_report=reference_latency_report,
                candidate_report=candidate_latency_report,
                reference_checkpoint_sha256=reference_checkpoint_sha256,
                candidate_checkpoint_sha256=candidate_checkpoint_sha256,
                reference_checkpoint_byte_count=reference_checkpoint_byte_count,
                candidate_checkpoint_byte_count=candidate_checkpoint_byte_count,
                replay_contract=replay_contract,
            )
        )
        latency_guardrail = paired_latency_guardrail(
            candidate_latency_metrics,
            reference_latency_metrics,
        )
    else:
        # No truthful paired wall-clock control exists inside trainer validation.
        # Missing evaluator evidence therefore fails comprehensive promotion closed.
        latency_guardrail = paired_latency_guardrail({}, {})
    physical_promotion_eligible, comprehensive_promotion_eligible = _promotion_eligibility(
        accuracy_improves=_rollout_selection_improves(candidate, reference),
        reference_guardrail_failures=reference_guardrail_failures,
        training_support_failures=training_support_failures,
        mutable_support_failures=mutable_support_failures,
        latency_promotion_eligible=latency_guardrail.promotion_eligible,
    )
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "promotion_backend": str(device),
        "pytorch_version": torch.__version__,
        "precision": device_info.precision,
        "physical_promotion_eligible": physical_promotion_eligible,
        # Compatibility alias retained for existing automation. Its explicit
        # scope below prevents it from being read as physical-only acceptance.
        "promotion_eligible": comprehensive_promotion_eligible,
        "promotion_eligible_scope": "alias_of_comprehensive_promotion_eligible",
        "reference_checkpoint": str(reference_path),
        "reference_checkpoint_sha256": reference_checkpoint_sha256,
        "reference_checkpoint_byte_count": reference_checkpoint_byte_count,
        "reference_checkpoint_identity_source": "captured_once_immutable_byte_snapshot",
        "candidate_checkpoint": str(candidate_path),
        "candidate_checkpoint_sha256": candidate_checkpoint_sha256,
        "candidate_checkpoint_byte_count": candidate_checkpoint_byte_count,
        "candidate_checkpoint_identity_source": "captured_once_immutable_byte_snapshot",
        "runtime_source_fingerprint": runtime_source_fingerprint,
        "resolved_config_sha256": replay_contract["resolved_config_sha256"],
        "physical_replay_contract": replay_contract,
        "physical_replay_contract_sha256": _canonical_sha256(replay_contract),
        "validation_episode_count": config.training.validation_episodes,
        "validation_protocol_hash": _rollout_validation_protocol_hash(config),
        "scenario_mixture": list(config.simulator.scenario_mixture),
        "reference": reference.validation_metrics(),
        "candidate": candidate.validation_metrics(),
        "reference_guardrail_failures": reference_guardrail_failures,
        "training_support_failures": training_support_failures,
        "mutable_support_failures": mutable_support_failures,
        **latency_guardrail.metrics(),
        "latency_report_binding": latency_binding,
        "comprehensive_promotion_eligible": comprehensive_promotion_eligible,
    }
    atomic_write_text(output / "report.json", json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if comprehensive_promotion_eligible else 2


if __name__ == "__main__":
    raise SystemExit(main())
