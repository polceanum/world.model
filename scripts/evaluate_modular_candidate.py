#!/usr/bin/env python3
"""Compose and strictly qualify selected checkpoint modules on trainer seeds."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from world_model.runtime import OnlineWorldModel
from world_model.training.checkpoint_composition import (
    compose_model_state,
    compose_model_state_rows,
)
from world_model.training.checkpointing import (
    capture_git_metadata,
    save_checkpoint,
    validate_checkpoint_config,
)
from world_model.training.trainer import (
    _handoff_training_support_failures,
    _make_loader,
    _mutable_causal_training_support_failures,
    _rollout_selection_from_checkpoint,
    _rollout_selection_guardrail_failures,
    _rollout_selection_improves,
    _rollout_selection_metrics,
    _rollout_selection_passes_guardrails,
    _validate_validation_support_schema,
    _validation_loader_result,
)
from world_model.utils.artifacts import timestamped_artifact_path
from world_model.utils.config import load_config
from world_model.utils.io import atomic_write_text
from world_model.utils.seeds import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--base", required=True, help="Accepted base checkpoint")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--donor", help="Candidate donor checkpoint")
    source.add_argument(
        "--fresh-initialization",
        action="store_true",
        help=(
            "Use the deterministic fresh model initialization as the donor. "
            "This is required when a changed module's parameter meaning is "
            "incompatible with inherited weights."
        ),
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--module",
        action="append",
        dest="modules",
        help="Top-level or dotted state-dict module prefix; may be repeated",
    )
    selection.add_argument(
        "--tensor-row",
        action="append",
        dest="tensor_rows",
        metavar="STATE_DICT_NAME:ROW",
        help=(
            "Import one leading-dimension tensor row from the donor; may be "
            "repeated for axis-local output-head qualification"
        ),
    )
    parser.add_argument("--donor-weight", type=float, default=1.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def _load_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model_state"), dict):
        raise ValueError(f"checkpoint does not contain a model state: {path}")
    return payload


def _selection_from_numbered_checkpoint(payload: dict[str, Any], config: Any) -> Any:
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("base checkpoint does not contain selection metrics")
    selection = _rollout_selection_from_checkpoint(metrics, config)
    if selection is None:
        raise ValueError("base checkpoint does not contain a complete accepted selection")
    return selection


def _parse_tensor_rows(values: list[str] | None) -> dict[str, list[int]]:
    parsed: dict[str, list[int]] = {}
    for value in values or []:
        name, separator, row_text = value.rpartition(":")
        if not separator or not name or not row_text:
            raise ValueError(f"--tensor-row must use STATE_DICT_NAME:ROW, received {value!r}")
        try:
            row = int(row_text)
        except ValueError as error:
            raise ValueError(f"tensor row must be an integer, received {value!r}") from error
        parsed.setdefault(name, []).append(row)
    return parsed


def main() -> int:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("--num-workers must be nonnegative")
    config = load_config(
        args.config,
        overrides=[f"training.num_workers={args.num_workers}"],
    )
    config.validate()
    seed_everything(config.project.seed, deterministic=config.project.deterministic)

    base_path = Path(args.base).expanduser().resolve()
    base_payload = _load_payload(base_path)
    validate_checkpoint_config(base_payload, config)
    reference = _selection_from_numbered_checkpoint(base_payload, config)
    model = OnlineWorldModel.from_config(config, device=torch.device(args.device))
    donor_path: Path | None = None
    donor_payload: dict[str, Any] | None = None
    if args.fresh_initialization:
        if args.donor_weight != 1.0:
            raise ValueError("fresh initialization requires --donor-weight=1")
        donor_state = model.state_dict()
        donor_step = 0
    else:
        donor_path = Path(args.donor).expanduser().resolve()
        donor_payload = _load_payload(donor_path)
        validate_checkpoint_config(donor_payload, config)
        donor_state = donor_payload["model_state"]
        donor_step = int(donor_payload["step"])
    tensor_rows = _parse_tensor_rows(args.tensor_rows)
    if args.modules:
        composed_state, selected = compose_model_state(
            base_payload["model_state"],
            donor_state,
            module_prefixes=args.modules,
            donor_weight=args.donor_weight,
        )
    else:
        composed_state, selected = compose_model_state_rows(
            base_payload["model_state"],
            donor_state,
            tensor_rows=tensor_rows,
            donor_weight=args.donor_weight,
        )

    model.load_state_dict(composed_state)
    model.eval()
    loader = _make_loader(
        config,
        split="validation",
        episodes=config.training.validation_episodes,
        shuffle=False,
        batch_size_override=1,
    )
    output = timestamped_artifact_path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    validation = _validation_loader_result(
        model,
        loader,
        config,
        device=torch.device(args.device),
        closed_loop=True,
        progress_path=output / "validation_progress.json",
        progress_split="validation",
    )
    atomic_write_text(
        output / "validation_metrics.json",
        json.dumps(validation.metrics, indent=2, sort_keys=True) + "\n",
    )
    _validate_validation_support_schema(validation.metrics, config)
    candidate = _rollout_selection_metrics(
        validation.metrics,
        config,
        require_scenarios=True,
    )
    reference_failures = _rollout_selection_guardrail_failures(candidate, reference)
    training_support_failures = _handoff_training_support_failures(
        candidate,
        reference,
        config,
    )
    mutable_support_failures = _mutable_causal_training_support_failures(candidate, config)
    accepted = (
        not training_support_failures
        and _rollout_selection_improves(candidate, reference)
        and _rollout_selection_passes_guardrails(candidate, reference)
    )
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "accepted_against_base": accepted,
        "base_checkpoint": str(base_path),
        "base_step": int(base_payload["step"]),
        "donor_checkpoint": None if donor_path is None else str(donor_path),
        "donor_step": donor_step,
        "fresh_initialization": bool(args.fresh_initialization),
        "modules": list(args.modules or []),
        "tensor_rows": tensor_rows,
        "donor_weight": args.donor_weight,
        "selected_tensor_count": len(selected),
        "candidate": candidate.validation_metrics(),
        "base": reference.validation_metrics(),
        "reference_guardrail_failures": reference_failures,
        "training_support_failures": training_support_failures,
        "mutable_support_failures": mutable_support_failures,
    }
    atomic_write_text(
        output / "report.json",
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    save_checkpoint(
        output / "candidate.pt",
        model=model,
        optimizer=None,
        config=config,
        step=donor_step,
        metrics={
            "modular_candidate": 1.0,
            "accepted_against_base": float(accepted),
            **candidate.validation_metrics(),
        },
        device=args.device,
        source_provenance={
            **capture_git_metadata(Path(__file__).resolve().parents[1]),
            "checkpoint_composition": {
                "base_checkpoint": str(base_path),
                "donor_checkpoint": None if donor_path is None else str(donor_path),
                "fresh_initialization": bool(args.fresh_initialization),
                "project_seed": config.project.seed,
                "modules": list(args.modules or []),
                "tensor_rows": tensor_rows,
                "donor_weight": args.donor_weight,
            },
        },
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "accepted_against_base": accepted,
                "base_score": reference.score,
                "candidate_score": candidate.score,
                "base_horizon_position_rmse_m": reference.horizon_position_rmse_m,
                "candidate_horizon_position_rmse_m": candidate.horizon_position_rmse_m,
                "base_horizon_forecast_target_coverage": (
                    reference.horizon_forecast_target_coverage
                ),
                "candidate_horizon_forecast_target_coverage": (
                    candidate.horizon_forecast_target_coverage
                ),
                "reference_guardrail_failure_count": len(reference_failures),
                "training_support_failure_count": len(training_support_failures),
                "mutable_support_failure_count": len(mutable_support_failures),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
