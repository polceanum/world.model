from __future__ import annotations

import pytest

from world_model.training.trainer import measurement_pretrain_frame_index


def test_fixed_pretraining_sweeps_every_frame_for_every_loader_batch() -> None:
    loader_batches = 4
    total_frames = 16
    visited = {batch_index: [] for batch_index in range(loader_batches)}

    for step in range(loader_batches * total_frames):
        batch_index = step % loader_batches
        visited[batch_index].append(
            measurement_pretrain_frame_index(
                step,
                loader_batches=loader_batches,
                total_frames=total_frames,
                fixed_dataset=True,
            )
        )

    expected = list(range(total_frames))
    assert all(frame_indices == expected for frame_indices in visited.values())


def test_pretraining_frame_index_rejects_empty_axes() -> None:
    for loader_batches, total_frames in ((0, 16), (4, 0)):
        with pytest.raises(ValueError, match="must be positive"):
            measurement_pretrain_frame_index(
                0,
                loader_batches=loader_batches,
                total_frames=total_frames,
                fixed_dataset=True,
            )


def test_streaming_pretraining_samples_a_valid_frame() -> None:
    sampled = {
        measurement_pretrain_frame_index(
            step,
            loader_batches=4,
            total_frames=7,
            fixed_dataset=False,
        )
        for step in range(32)
    }
    assert sampled
    assert sampled <= set(range(7))
