from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from world_model.training.dynamic_set_optimization import (
    DynamicSetOptimizerConfig,
    build_dynamic_set_optimizer,
    dynamic_set_capacity,
    learning_rate_multiplier,
)


class _TinyStructuredModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.perception = nn.Sequential(nn.Linear(8, 32), nn.SiLU(), nn.Linear(32, 8))
        self.relation = nn.Sequential(nn.Linear(13, 16), nn.SiLU(), nn.Linear(16, 5))
        self.frozen_other = nn.Linear(6, 6)


def test_schedule_is_step_one_warmup_and_cosine_to_ten_percent() -> None:
    config = DynamicSetOptimizerConfig()
    assert learning_rate_multiplier(0, config) == pytest.approx(1.0 / 512.0)
    assert learning_rate_multiplier(511, config) == pytest.approx(1.0)
    assert learning_rate_multiplier(512, config) == pytest.approx(1.0)
    midpoint = (config.maximum_updates + config.warmup_updates) // 2
    assert learning_rate_multiplier(midpoint, config) == pytest.approx(0.55)
    assert learning_rate_multiplier(config.maximum_updates, config) == pytest.approx(0.10)
    assert learning_rate_multiplier(config.maximum_updates + 1_000, config) == pytest.approx(0.10)


def test_optimizer_has_exact_owner_groups_and_freezes_everything_else() -> None:
    model = _TinyStructuredModel()
    optimizer, scheduler, capacity = build_dynamic_set_optimizer(
        model,
        perception=model.perception,
        relation=model.relation,
        legacy_modules=(model.frozen_other,),
    )

    assert [group["name"] for group in optimizer.param_groups] == ["perception", "relation"]
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3.0e-4 / 512.0)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(1.0e-4 / 512.0)
    assert all(parameter.requires_grad for parameter in model.perception.parameters())
    assert all(parameter.requires_grad for parameter in model.relation.parameters())
    assert not any(parameter.requires_grad for parameter in model.frozen_other.parameters())
    assert capacity.total_new_parameters == sum(
        parameter.numel()
        for module in (model.perception, model.relation)
        for parameter in module.parameters()
    )

    before = [group["lr"] for group in optimizer.param_groups]
    optimizer.step()
    scheduler.step()
    after = [group["lr"] for group in optimizer.param_groups]
    assert all(
        math.isfinite(value) and value > old for old, value in zip(before, after, strict=True)
    )


def test_capacity_rejects_overlapping_owners() -> None:
    model = _TinyStructuredModel()
    with pytest.raises(ValueError, match="overlap"):
        dynamic_set_capacity(
            model,
            perception=model.perception,
            relation=model.perception,
            legacy_modules=(model.frozen_other,),
        )


def test_capacity_counts_float32_weight_bytes() -> None:
    model = _TinyStructuredModel().to(dtype=torch.float32)
    capacity = dynamic_set_capacity(
        model,
        perception=model.perception,
        relation=model.relation,
        legacy_modules=(model.frozen_other,),
    )
    assert capacity.complete_model_weight_bytes == 4 * capacity.complete_model_parameters


def test_capacity_rejects_an_unclassified_hidden_parameter() -> None:
    model = _TinyStructuredModel()
    model.hidden_new_owner = nn.Linear(3, 3)

    with pytest.raises(ValueError, match="unclassified=2"):
        dynamic_set_capacity(
            model,
            perception=model.perception,
            relation=model.relation,
            legacy_modules=(model.frozen_other,),
        )
