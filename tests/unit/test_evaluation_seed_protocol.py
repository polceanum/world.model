from __future__ import annotations

import pytest

from world_model.datasets.splits import SPLIT_SEED_RANGES
from world_model.evaluation.seed_protocol import (
    FRESH_VALIDATION_SEED_PROTOCOL,
    STANDARD_SEED_PROTOCOL,
    make_evaluation_seed_protocol,
)


def test_fresh_validation_starts_after_trainer_manifest_and_avoids_test() -> None:
    protocol = make_evaluation_seed_protocol(
        name=FRESH_VALIDATION_SEED_PROTOCOL,
        split="validation",
        episode_count=8,
        training_validation_episodes=4,
    )

    validation_start = SPLIT_SEED_RANGES["validation"][0]
    test_lower, test_upper = SPLIT_SEED_RANGES["test"]
    assert protocol.manifest.seeds == tuple(range(validation_start + 4, validation_start + 12))
    assert not protocol.overlaps_training_validation
    assert not protocol.overlaps_test_range
    assert all(not (test_lower <= seed <= test_upper) for seed in protocol.manifest)
    assert protocol.metadata()["evaluation_seed_role"] == "checkpoint_selection_validation"


def test_fresh_validation_rejects_test_split_for_model_selection() -> None:
    with pytest.raises(ValueError, match="requires split='validation'"):
        make_evaluation_seed_protocol(
            name=FRESH_VALIDATION_SEED_PROTOCOL,
            split="test",
            episode_count=8,
            training_validation_episodes=4,
        )


def test_standard_test_protocol_preserves_existing_seed_manifest() -> None:
    protocol = make_evaluation_seed_protocol(
        name=STANDARD_SEED_PROTOCOL,
        split="test",
        episode_count=3,
        training_validation_episodes=4,
    )

    assert protocol.manifest.seeds == (200_000, 200_001, 200_002)
    assert protocol.overlaps_test_range
    assert not protocol.overlaps_training_validation


def test_fresh_validation_fails_when_reserved_range_is_exhausted() -> None:
    with pytest.raises(ValueError, match="exceeds the reserved validation range"):
        make_evaluation_seed_protocol(
            name=FRESH_VALIDATION_SEED_PROTOCOL,
            split="validation",
            episode_count=2,
            training_validation_episodes=10_000,
        )
