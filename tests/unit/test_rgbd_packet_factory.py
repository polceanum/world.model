from __future__ import annotations

import pytest
import torch

from world_model.evaluation.evaluator import _make_runtime_packet
from world_model.training.loop import make_rgbd_packet
from world_model.utils.config import load_config


def _batch() -> dict[str, object]:
    batch, frames, height, width = 2, 3, 8, 10
    return {
        "rgb": torch.zeros(batch, frames, 3, height, width),
        "depth": torch.ones(batch, frames, 1, height, width),
        "timestamps": torch.tensor([[0.0, 0.05, 0.1], [0.0, 0.05, 0.1]]),
        "camera": {
            "world_from_camera": torch.eye(4).expand(batch, frames, 4, 4).clone(),
            "intrinsics": torch.eye(3).expand(batch, frames, 3, 3).clone(),
        },
    }


def test_make_rgbd_packet_is_one_batched_composite_stream() -> None:
    packet = make_rgbd_packet(_batch(), 1)

    assert packet.modality == "rgbd"
    assert packet.sensor_id == "camera0:rgbd"
    assert packet.frame_id == "camera:camera0:rgbd"
    assert packet.timestamp == pytest.approx(0.05)
    assert packet.metadata["image_size"] == (8, 10)
    assert packet.metadata["depth_semantics"] == (
        "observable_camera_z_surface_depth_zero_means_no_return"
    )
    assert isinstance(packet.payload, dict)
    assert packet.payload["rgb"].shape == (2, 3, 8, 10)
    assert packet.payload["depth"].shape == (2, 1, 8, 10)
    assert packet.calibration["world_from_camera"].shape == (2, 4, 4)
    assert packet.calibration["intrinsics"].shape == (2, 3, 3)


def test_make_rgbd_packet_rejects_misaligned_depth_before_packet_creation() -> None:
    batch = _batch()
    batch["depth"] = torch.ones(2, 3, 1, 7, 10)

    with pytest.raises(ValueError, match="RGB and depth sequence dimensions must match"):
        make_rgbd_packet(batch, 0)


def test_make_rgbd_packet_requires_shared_batched_timestamp() -> None:
    batch = _batch()
    batch["timestamps"][1, 1] = 0.051

    with pytest.raises(ValueError, match="shared frame timestamp"):
        make_rgbd_packet(batch, 1)


def test_evaluator_routes_rgbd_profile_through_composite_packet_factory() -> None:
    config = load_config("configs/rgbd_online_free_motion_cpu.yaml")

    packet = _make_runtime_packet(config, _batch(), 2)

    assert packet.modality == "rgbd"
    assert packet.sensor_id == "camera0:rgbd"
    assert isinstance(packet.payload, dict)
    assert packet.payload["rgb"].shape == (2, 3, 8, 10)
    assert packet.payload["depth"].shape == (2, 1, 8, 10)
