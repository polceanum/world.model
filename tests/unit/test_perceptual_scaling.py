from __future__ import annotations

from pathlib import Path

import pytest

from world_model.evaluation.perceptual_scaling import scaled_set_config
from world_model.training.dynamic_set_config import load_config

CONFIG = Path(__file__).parents[2] / "configs" / "rgbd_dynamic_set_planning_cpu.yaml"


def test_n8_profile_derives_ten_proposals_without_mutating_n6_config() -> None:
    base = load_config(CONFIG)

    scaled = scaled_set_config(base, max_objects=8)

    assert base.model.max_objects == 6
    assert base.model.rgbd.max_objects == 6
    assert base.model.rgbd.proposal_count == 8
    assert scaled.model.max_objects == 8
    assert scaled.model.rgbd.max_objects == 8
    assert scaled.model.rgbd.birth_proposals == 2
    assert scaled.model.rgbd.proposal_count == 10
    assert scaled.simulator.max_objects == 8


@pytest.mark.parametrize("count", [0, -1, True])
def test_scaled_profile_rejects_invalid_capacity(count: int) -> None:
    with pytest.raises(ValueError, match="max_objects"):
        scaled_set_config(load_config(CONFIG), max_objects=count)
