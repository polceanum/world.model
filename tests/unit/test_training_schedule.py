from __future__ import annotations

import pytest
from torch import nn

from world_model.runtime import OnlineWorldModel
from world_model.training.trainer import (
    measurement_pretrain_frame_index,
    set_global_perception_trainable,
)
from world_model.utils.config import load_config


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


def test_global_perception_freeze_leaves_fast_roi_trainable() -> None:
    model = OnlineWorldModel.from_config(load_config("configs/tiny_overfit.yaml"))
    rgb = model.observation_modules["rgb"]

    set_global_perception_trainable(model, trainable=False)

    assert not any(parameter.requires_grad for parameter in rgb.backbone.parameters())
    assert not any(parameter.requires_grad for parameter in rgb.global_detector.parameters())
    assert all(parameter.requires_grad for parameter in rgb.roi_updater.parameters())
    assert isinstance(rgb.roi_updater, nn.Module)

    set_global_perception_trainable(model, trainable=True)
    assert all(parameter.requires_grad for parameter in rgb.backbone.parameters())
    assert all(parameter.requires_grad for parameter in rgb.global_detector.parameters())
