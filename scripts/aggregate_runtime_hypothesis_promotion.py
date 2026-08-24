#!/usr/bin/env python3
"""Fail-closed aggregate gate for all runtime-hypothesis promotion splits."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.compare_runtime_hypothesis_evaluations import (
    CapturedReport,
    _canonical_sha256,
    _expected_runtime_environment,
    _expected_runtime_policy,
    _require_tracked_source_file,
    compare_evaluation_reports,
)
from world_model.evaluation.seed_protocol import make_evaluation_seed_protocol
from world_model.training.checkpointing import capture_git_metadata
from world_model.utils.config import OrpheusConfig, load_config
from world_model.utils.io import atomic_write_text

_COMPARISON_SCHEMA = "runtime_hypothesis_paired_promotion_v3"
_SUITE_SCHEMA = "runtime_hypothesis_comprehensive_suite_v1"
_SPLIT_CONTRACTS: dict[str, tuple[str, str, int]] = {
    "standard_validation": ("validation", "standard", 0),
    "fresh_validation": ("validation", "fresh_validation", 32),
    "test": ("test", "standard", 0),
    "ood": ("ood", "standard", 0),
}


def _require_true(payload: Mapping[str, Any], name: str, *, role: str) -> None:
    if payload.get(name) is not True:
        raise ValueError(f"{role} does not prove {name}=true")


def _expected_protocol(
    config: OrpheusConfig,
    *,
    split: str,
    seed_protocol: str,
    seed_offset: int,
    expected_device: str,
) -> dict[str, Any]:
    resolved = make_evaluation_seed_protocol(
        name=seed_protocol,
        split=split,
        episode_count=config.evaluation.episodes,
        training_validation_episodes=config.training.validation_episodes,
        seed_offset=seed_offset,
    )
    policy = _expected_runtime_policy(config)
    candidate_names = [candidate["name"] for candidate in policy["candidates"]]
    base_candidates = {
        "learned",
        "constant_velocity",
        "damped_constant_velocity",
        "ballistic_contact",
    }
    return {
        "split": resolved.split,
        "seed_protocol": resolved.name,
        "seed_offset": resolved.seed_offset,
        "seed_role": resolved.intended_use,
        "seed_manifest": list(resolved.manifest.seeds),
        "scenario_mixture": list(config.simulator.scenario_mixture),
        "horizons": [f"{value:.3f}s" for value in config.evaluation.horizons_seconds],
        "device": expected_device,
        "precision": "float32",
        "reference_runtime_hypothesis_pool": False,
        "candidate_runtime_hypothesis_pool": True,
        "absolute_tolerance": 1.0e-9,
        "relative_tolerance": 1.0e-6,
        "sharpness_maximum_ratio": 1.05,
        "latency_maximum_ratio": 1.10,
        "minimum_pooled_position_improvement_m": 1.0e-5,
        "runtime_candidate_names": candidate_names,
        "required_runtime_candidate_names": [
            name for name in candidate_names if name not in base_candidates
        ],
    }


def aggregate_comparisons(
    comparisons: Mapping[str, CapturedReport],
    *,
    config: OrpheusConfig,
    current_source: Mapping[str, Any],
    expected_device: str = "mps",
    _raw_capture_sink: list[CapturedReport] | None = None,
) -> dict[str, Any]:
    """Validate four complete pair decisions and emit one promotion decision."""

    if set(comparisons) != set(_SPLIT_CONTRACTS):
        raise ValueError(
            "promotion suite requires exactly standard/fresh validation, test, and OOD"
        )
    if config.evaluation.episodes != 32 or config.training.validation_episodes != 32:
        raise ValueError("promotion suite requires exact fixed-32 evaluation semantics")
    if not config.runtime.hypothesis_local_applicability_enabled:
        raise ValueError("promotion suite requires local applicability enabled")
    if config.runtime.hypothesis_composition_step_seconds is None:
        raise ValueError("promotion suite requires bounded composition enabled")

    expected_source = dict(current_source)
    expected_config_sha = _canonical_sha256(config.to_dict())
    expected_environment = _expected_runtime_environment(expected_device)
    expected_policy = _expected_runtime_policy(config)
    common_checkpoint: tuple[str, int, int] | None = None
    validated: dict[str, Any] = {}
    raw_report_captures: list[CapturedReport] = []

    for role, contract in _SPLIT_CONTRACTS.items():
        captured = comparisons[role]
        payload = captured.payload
        if payload.get("schema_version") != _COMPARISON_SCHEMA:
            raise ValueError(f"{role} comparison schema is not {_COMPARISON_SCHEMA}")
        for name in (
            "passed",
            "physical_promotion_eligible",
            "latency_guardrail_supported",
            "latency_guardrail_passed",
            "comprehensive_promotion_eligible",
        ):
            _require_true(payload, name, role=role)
        physical = payload.get("physical")
        latency = payload.get("latency")
        if not isinstance(physical, Mapping) or not isinstance(latency, Mapping):
            raise ValueError(f"{role} comparison omits physical or latency evidence")
        if physical.get("failure_count") != 0:
            raise ValueError(f"{role} physical comparison contains failures")
        if latency.get("latency_guardrail_failures") != []:
            raise ValueError(f"{role} latency comparison contains failures")
        if payload.get("source_provenance") != expected_source:
            raise ValueError(f"{role} comparison source provenance is not current")
        if payload.get("resolved_evaluation_config_sha256") != expected_config_sha:
            raise ValueError(f"{role} comparison config digest is not current")
        if payload.get("evaluation_runtime_environment") != expected_environment:
            raise ValueError(f"{role} comparison runtime environment is not current")
        if payload.get("runtime_hypothesis_pool_policy") != expected_policy:
            raise ValueError(f"{role} comparison runtime policy is not current")
        split, seed_protocol, seed_offset = contract
        if payload.get("protocol") != _expected_protocol(
            config,
            split=split,
            seed_protocol=seed_protocol,
            seed_offset=seed_offset,
            expected_device=expected_device,
        ):
            raise ValueError(f"{role} comparison protocol is not exact")
        raw_arms: dict[str, CapturedReport] = {}
        for arm in ("reference", "candidate"):
            identity = payload.get(f"{arm}_report")
            if not isinstance(identity, Mapping) or not isinstance(identity.get("path"), str):
                raise ValueError(f"{role} comparison omits the {arm} report identity")
            raw_capture = CapturedReport.capture(
                identity["path"],
                role=f"{role} {arm} evaluator report",
            )
            if raw_capture.identity() != dict(identity):
                raise ValueError(f"{role} {arm} evaluator report identity changed")
            raw_arms[arm] = raw_capture
            raw_report_captures.append(raw_capture)
        reproduced = compare_evaluation_reports(
            raw_arms["reference"],
            raw_arms["candidate"],
            config=config,
            current_source=current_source,
            expected_device=expected_device,
            split=split,
            seed_protocol=seed_protocol,
            seed_offset=seed_offset,
        )
        supplied_stable = {name: value for name, value in payload.items() if name != "created_utc"}
        reproduced_stable = {
            name: value for name, value in reproduced.items() if name != "created_utc"
        }
        if supplied_stable != reproduced_stable:
            raise ValueError(f"{role} comparison does not reproduce from its evaluator reports")
        checkpoint_sha = payload.get("checkpoint_sha256")
        checkpoint_bytes = payload.get("checkpoint_byte_count")
        checkpoint_step = payload.get("checkpoint_step")
        if (
            not isinstance(checkpoint_sha, str)
            or len(checkpoint_sha) != 64
            or isinstance(checkpoint_bytes, bool)
            or not isinstance(checkpoint_bytes, int)
            or checkpoint_bytes <= 0
            or isinstance(checkpoint_step, bool)
            or not isinstance(checkpoint_step, int)
            or checkpoint_step < 0
        ):
            raise ValueError(f"{role} comparison checkpoint identity is invalid")
        try:
            int(checkpoint_sha, 16)
        except ValueError as error:
            raise ValueError(f"{role} comparison checkpoint digest is invalid") from error
        identity = (checkpoint_sha, checkpoint_bytes, checkpoint_step)
        if common_checkpoint is None:
            common_checkpoint = identity
        elif identity != common_checkpoint:
            raise ValueError("promotion suite comparisons do not share one checkpoint")
        validated[role] = {
            "comparison": captured.identity(),
            "protocol": payload["protocol"],
            "primary_physical_metrics_sha256": payload.get("primary_physical_metrics_sha256"),
            "posterior_trace_sha256": payload.get("posterior_trace_sha256"),
            "physical": physical,
            "latency": latency,
        }

    for raw_capture in raw_report_captures:
        raw_capture.assert_path_identity()
    if _raw_capture_sink is not None:
        _raw_capture_sink.extend(raw_report_captures)

    assert common_checkpoint is not None
    return {
        "schema_version": _SUITE_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "comprehensive_promotion_eligible": True,
        "checkpoint_sha256": common_checkpoint[0],
        "checkpoint_byte_count": common_checkpoint[1],
        "checkpoint_step": common_checkpoint[2],
        "source_provenance": expected_source,
        "resolved_evaluation_config_sha256": expected_config_sha,
        "evaluation_runtime_environment": expected_environment,
        "runtime_hypothesis_pool_policy": expected_policy,
        "required_comparison_roles": list(_SPLIT_CONTRACTS),
        "comparisons": validated,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--standard-validation", required=True)
    parser.add_argument("--fresh-validation", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--ood", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository = Path(__file__).resolve().parents[1]
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"promotion-suite output must be fresh: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"promotion-suite output parent does not exist: {output.parent}")
    try:
        output.relative_to(repository)
    except ValueError:
        pass
    else:
        raise ValueError("promotion-suite output must remain outside the source repository")
    config_path = Path(args.config).expanduser().resolve()
    _require_tracked_source_file(repository, config_path, role="promotion-suite config")
    config = load_config(config_path, overrides=args.set)
    current_source = capture_git_metadata(repository)
    if current_source.get("dirty") is not False:
        raise ValueError("promotion-suite aggregation requires a clean source tree")
    paths = {
        "standard_validation": args.standard_validation,
        "fresh_validation": args.fresh_validation,
        "test": args.test,
        "ood": args.ood,
    }
    captures = {
        role: CapturedReport.capture(path, role=f"{role} comparison")
        for role, path in paths.items()
    }
    if len({capture.path for capture in captures.values()}) != len(captures):
        raise ValueError("promotion-suite comparison artifacts must be distinct")
    raw_captures: list[CapturedReport] = []
    result = aggregate_comparisons(
        captures,
        config=config,
        current_source=current_source,
        expected_device=args.device,
        _raw_capture_sink=raw_captures,
    )
    for capture in captures.values():
        capture.assert_path_identity()
    for raw_capture in raw_captures:
        raw_capture.assert_path_identity()
    if capture_git_metadata(repository) != current_source:
        raise RuntimeError("source provenance changed during promotion-suite aggregation")
    encoded = json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n"
    atomic_write_text(output, encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
