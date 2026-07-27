"""Explicit seed manifests for standard and checkpoint-selection evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from world_model.datasets.splits import (
    SPLIT_SEED_RANGES,
    SeedManifest,
    canonical_split_name,
    make_seed_manifest,
)

STANDARD_SEED_PROTOCOL = "standard"
FRESH_VALIDATION_SEED_PROTOCOL = "fresh_validation"
EVALUATION_SEED_PROTOCOLS = (
    STANDARD_SEED_PROTOCOL,
    FRESH_VALIDATION_SEED_PROTOCOL,
)


@dataclass(frozen=True)
class EvaluationSeedProtocol:
    """Resolved deterministic episode manifest and its model-selection role."""

    name: str
    split: str
    manifest: SeedManifest
    seed_offset: int
    intended_use: str
    overlaps_training_validation: bool
    overlaps_test_range: bool

    def metadata(self) -> dict[str, Any]:
        """Return JSON-safe provenance for an evaluation report."""

        return {
            "evaluation_seed_protocol": self.name,
            "evaluation_seed_role": self.intended_use,
            "evaluation_seed_offset": self.seed_offset,
            "evaluation_seed_count": len(self.manifest),
            "evaluation_seed_first": self.manifest.seeds[0],
            "evaluation_seed_last": self.manifest.seeds[-1],
            "evaluation_episode_seeds": list(self.manifest.seeds),
            "evaluation_seed_overlaps_training_validation": (self.overlaps_training_validation),
            "evaluation_seed_overlaps_test_range": self.overlaps_test_range,
        }


def make_evaluation_seed_protocol(
    *,
    name: str,
    split: str,
    episode_count: int,
    training_validation_episodes: int,
    seed_offset: int | None = None,
) -> EvaluationSeedProtocol:
    """Resolve a fixed evaluation manifest without silently reusing test seeds.

    ``fresh_validation`` is intended for comparing candidate checkpoints.  It
    begins immediately after the validation episodes used by the trainer, stays
    inside the reserved validation range, and is therefore disjoint from both
    training-time validation and the complete test seed range.
    """

    if name not in EVALUATION_SEED_PROTOCOLS:
        raise ValueError(
            f"unknown evaluation seed protocol {name!r}; "
            f"expected one of {list(EVALUATION_SEED_PROTOCOLS)}"
        )
    canonical_split = canonical_split_name(split)
    if episode_count <= 0:
        raise ValueError("evaluation episode_count must be positive")
    if training_validation_episodes < 0:
        raise ValueError("training_validation_episodes must be nonnegative")
    if seed_offset is not None and seed_offset < 0:
        raise ValueError("evaluation seed_offset must be nonnegative")

    if name == FRESH_VALIDATION_SEED_PROTOCOL:
        if canonical_split != "validation":
            raise ValueError(
                "fresh_validation seed protocol requires split='validation'; "
                "test seeds must not be used for checkpoint selection"
            )
        if seed_offset is None:
            seed_offset = training_validation_episodes
        elif seed_offset < training_validation_episodes:
            raise ValueError(
                "fresh validation seed_offset must not overlap trainer validation episodes"
            )
        intended_use = "checkpoint_selection_validation"
    else:
        seed_offset = 0 if seed_offset is None else seed_offset
        intended_use = f"standard_{canonical_split}_evaluation"

    manifest = make_seed_manifest(
        canonical_split,
        episode_count,
        offset=seed_offset,
    )
    training_validation_manifest = make_seed_manifest(
        "validation",
        training_validation_episodes,
    )
    training_validation_seeds = set(training_validation_manifest.seeds)
    overlaps_training_validation = bool(training_validation_seeds.intersection(manifest.seeds))
    test_lower, test_upper = SPLIT_SEED_RANGES["test"]
    overlaps_test_range = any(test_lower <= seed <= test_upper for seed in manifest.seeds)

    if name == FRESH_VALIDATION_SEED_PROTOCOL:
        if overlaps_training_validation:
            raise RuntimeError("fresh validation manifest overlaps trainer validation seeds")
        if overlaps_test_range:
            raise RuntimeError("fresh validation manifest overlaps reserved test seeds")

    return EvaluationSeedProtocol(
        name=name,
        split=canonical_split,
        manifest=manifest,
        seed_offset=seed_offset,
        intended_use=intended_use,
        overlaps_training_validation=overlaps_training_validation,
        overlaps_test_range=overlaps_test_range,
    )


__all__ = [
    "EVALUATION_SEED_PROTOCOLS",
    "FRESH_VALIDATION_SEED_PROTOCOL",
    "STANDARD_SEED_PROTOCOL",
    "EvaluationSeedProtocol",
    "make_evaluation_seed_protocol",
]
