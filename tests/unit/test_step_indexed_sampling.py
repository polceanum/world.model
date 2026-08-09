from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader

from world_model.training.sampling import (
    ScenarioBalancedStepIndexedBatchSampler,
    StepIndexedBatchSampler,
)


def _legacy_loader_batches(
    *,
    dataset_size: int,
    batch_size: int,
    seed: int,
    steps: int,
) -> list[list[int]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    loader = DataLoader(
        list(range(dataset_size)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=generator,
    )
    batches: list[list[int]] = []
    while len(batches) < steps:
        for batch in loader:
            batches.append([int(index) for index in batch])
            if len(batches) == steps:
                break
    return batches


def _step_batches(
    *,
    dataset_size: int,
    batch_size: int,
    seed: int,
    start_step: int,
    stop_step: int,
) -> list[list[int]]:
    return list(
        StepIndexedBatchSampler(
            dataset_size=dataset_size,
            batch_size=batch_size,
            seed=seed,
            shuffle=True,
            start_step=start_step,
            stop_step=stop_step,
        )
    )


def test_resumed_sample_order_matches_uninterrupted_legacy_loader() -> None:
    dataset_size = 11
    batch_size = 4
    seed = 731
    stop_step = 13
    resume_step = 5  # Deliberately interrupt in the middle of an epoch.

    legacy = _legacy_loader_batches(
        dataset_size=dataset_size,
        batch_size=batch_size,
        seed=seed,
        steps=stop_step,
    )
    uninterrupted = _step_batches(
        dataset_size=dataset_size,
        batch_size=batch_size,
        seed=seed,
        start_step=0,
        stop_step=stop_step,
    )
    resumed = [
        *_step_batches(
            dataset_size=dataset_size,
            batch_size=batch_size,
            seed=seed,
            start_step=0,
            stop_step=resume_step,
        ),
        *_step_batches(
            dataset_size=dataset_size,
            batch_size=batch_size,
            seed=seed,
            start_step=resume_step,
            stop_step=stop_step,
        ),
    ]

    assert uninterrupted == legacy
    assert resumed == uninterrupted


def test_unshuffled_step_order_resumes_across_partial_last_batch() -> None:
    common = {
        "dataset_size": 5,
        "batch_size": 2,
        "seed": 19,
        "shuffle": False,
    }
    uninterrupted = list(
        StepIndexedBatchSampler(
            **common,
            start_step=0,
            stop_step=7,
        )
    )
    resumed = [
        *StepIndexedBatchSampler(
            **common,
            start_step=0,
            stop_step=2,
        ),
        *StepIndexedBatchSampler(
            **common,
            start_step=2,
            stop_step=7,
        ),
    ]

    assert uninterrupted == [
        [0, 1],
        [2, 3],
        [4],
        [0, 1],
        [2, 3],
        [4],
        [0, 1],
    ]
    assert resumed == uninterrupted


def test_scenario_balanced_batches_are_exact_and_resume_deterministically() -> None:
    common = {
        "scenario_index_by_dataset_index": [index % 4 for index in range(32)],
        "scenario_count": 4,
        "batch_size": 8,
        "seed": 733,
        "shuffle": True,
    }
    uninterrupted = list(
        ScenarioBalancedStepIndexedBatchSampler(
            **common,
            start_step=0,
            stop_step=11,
        )
    )
    resumed = [
        *ScenarioBalancedStepIndexedBatchSampler(
            **common,
            start_step=0,
            stop_step=5,
        ),
        *ScenarioBalancedStepIndexedBatchSampler(
            **common,
            start_step=5,
            stop_step=11,
        ),
    ]

    assert resumed == uninterrupted
    assert all(len(batch) == 8 for batch in uninterrupted)
    assert all(
        [sum(index % 4 == scenario for index in batch) for scenario in range(4)] == [2, 2, 2, 2]
        for batch in uninterrupted
    )
    assert all(
        len({index for batch in uninterrupted[epoch : epoch + 4] for index in batch}) == 32
        for epoch in (0, 4)
    )


def test_scenario_balanced_sampler_rejects_incomplete_or_unequal_protocols() -> None:
    with pytest.raises(ValueError, match="multiple of scenario_count"):
        ScenarioBalancedStepIndexedBatchSampler(
            scenario_index_by_dataset_index=[0, 1, 2, 3],
            scenario_count=4,
            batch_size=6,
            seed=0,
            shuffle=True,
            start_step=0,
            stop_step=1,
        )
    with pytest.raises(ValueError, match="equal nonempty scenario pools"):
        ScenarioBalancedStepIndexedBatchSampler(
            scenario_index_by_dataset_index=[0, 0, 1],
            scenario_count=2,
            batch_size=2,
            seed=0,
            shuffle=True,
            start_step=0,
            stop_step=1,
        )
