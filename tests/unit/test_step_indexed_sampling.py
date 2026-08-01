from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from world_model.training.sampling import StepIndexedBatchSampler


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
