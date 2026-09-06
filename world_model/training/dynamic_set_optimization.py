"""Strict optimizer ownership and schedule for specification 1.61."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

PERCEPTION_PARAMETER_LIMIT = 100_000
RELATION_PARAMETER_LIMIT = 50_000
TOTAL_NEW_PARAMETER_LIMIT = 250_000
LEARNED_WEIGHT_BYTE_LIMIT = 1 << 20


@dataclass(frozen=True)
class DynamicSetOptimizerConfig:
    perception_learning_rate: float = 3.0e-4
    relation_learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    warmup_updates: int = 512
    maximum_updates: int = 32_768
    final_learning_rate_fraction: float = 0.10

    def validate(self) -> DynamicSetOptimizerConfig:
        for name, value in (
            ("perception_learning_rate", self.perception_learning_rate),
            ("relation_learning_rate", self.relation_learning_rate),
            ("weight_decay", self.weight_decay),
        ):
            if isinstance(value, bool) or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.perception_learning_rate == 0.0 or self.relation_learning_rate == 0.0:
            raise ValueError("dynamic-set learning rates must be positive")
        for name, value in (
            ("warmup_updates", self.warmup_updates),
            ("maximum_updates", self.maximum_updates),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.warmup_updates >= self.maximum_updates:
            raise ValueError("warmup_updates must precede maximum_updates")
        if not 0.0 < self.final_learning_rate_fraction <= 1.0:
            raise ValueError("final_learning_rate_fraction must lie in (0,1]")
        return self


DEFAULT_DYNAMIC_SET_OPTIMIZER_CONFIG = DynamicSetOptimizerConfig()


@dataclass(frozen=True)
class DynamicSetCapacity:
    perception_parameters: int
    relation_parameters: int
    legacy_parameters: int
    total_new_parameters: int
    complete_model_parameters: int
    complete_model_weight_bytes: int

    def validate(self) -> DynamicSetCapacity:
        values = (
            self.perception_parameters,
            self.relation_parameters,
            self.legacy_parameters,
            self.total_new_parameters,
            self.complete_model_parameters,
            self.complete_model_weight_bytes,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values
        ):
            raise ValueError("dynamic-set capacity values must be nonnegative integers")
        if self.total_new_parameters != self.perception_parameters + self.relation_parameters:
            raise ValueError("total new capacity differs from the two declared new owners")
        if self.complete_model_parameters != self.total_new_parameters + self.legacy_parameters:
            raise ValueError("complete model capacity differs from the exact owner partition")
        if self.perception_parameters > PERCEPTION_PARAMETER_LIMIT:
            raise ValueError("set perception exceeds 100,000 parameters")
        if self.relation_parameters > RELATION_PARAMETER_LIMIT:
            raise ValueError("relation residual exceeds 50,000 parameters")
        if self.total_new_parameters > TOTAL_NEW_PARAMETER_LIMIT:
            raise ValueError("total new capacity exceeds 250,000 parameters")
        if self.complete_model_weight_bytes > LEARNED_WEIGHT_BYTE_LIMIT:
            raise ValueError("complete learned model weights exceed one MiB")
        return self


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _parameter_bytes(module: nn.Module) -> int:
    return sum(parameter.numel() * parameter.element_size() for parameter in module.parameters())


def dynamic_set_capacity(
    complete_model: nn.Module,
    *,
    perception: nn.Module,
    relation: nn.Module,
    legacy_modules: Sequence[nn.Module],
) -> DynamicSetCapacity:
    """Audit an exact, explicit partition of every learned parameter.

    The frozen legacy modules are named by the caller rather than inferred as
    the complement of the two new owners.  Consequently a newly introduced,
    undeclared parameter fails closed instead of evading the new-capacity cap.
    """

    if not isinstance(complete_model, nn.Module):
        raise TypeError("complete_model must be an nn.Module")
    if not isinstance(perception, nn.Module) or not isinstance(relation, nn.Module):
        raise TypeError("perception and relation owners must be nn.Modules")
    if isinstance(legacy_modules, (str, bytes)) or not isinstance(legacy_modules, Sequence):
        raise TypeError("legacy_modules must be a sequence of nn.Modules")
    legacy = tuple(legacy_modules)
    if not legacy or any(not isinstance(module, nn.Module) for module in legacy):
        raise ValueError("legacy_modules must explicitly name at least one nn.Module")

    complete_parameters = tuple(complete_model.parameters())
    complete_ids = {id(parameter) for parameter in complete_parameters}
    if len(complete_ids) != len(complete_parameters):
        raise ValueError("complete model exposes duplicate parameter identities")

    perception_parameters = tuple(perception.parameters())
    relation_parameters = tuple(relation.parameters())
    legacy_parameters = tuple(parameter for module in legacy for parameter in module.parameters())
    owner_groups = {
        "perception": {id(parameter) for parameter in perception_parameters},
        "relation": {id(parameter) for parameter in relation_parameters},
        "legacy": {id(parameter) for parameter in legacy_parameters},
    }
    if not owner_groups["perception"] or not owner_groups["relation"]:
        raise ValueError("perception and relation owners must be nonempty")
    if len(owner_groups["legacy"]) != len(legacy_parameters):
        raise ValueError("legacy modules overlap each other")
    if any(
        owner_groups[left] & owner_groups[right]
        for left, right in (
            ("perception", "relation"),
            ("perception", "legacy"),
            ("relation", "legacy"),
        )
    ):
        raise ValueError("dynamic-set parameter owners overlap")
    declared_ids = set().union(*owner_groups.values())
    if declared_ids != complete_ids:
        missing = len(complete_ids - declared_ids)
        foreign = len(declared_ids - complete_ids)
        raise ValueError(
            "dynamic-set parameter registry is not exhaustive "
            f"(unclassified={missing}, foreign={foreign})"
        )
    result = DynamicSetCapacity(
        perception_parameters=sum(parameter.numel() for parameter in perception_parameters),
        relation_parameters=sum(parameter.numel() for parameter in relation_parameters),
        legacy_parameters=sum(parameter.numel() for parameter in legacy_parameters),
        total_new_parameters=sum(
            parameter.numel() for parameter in (*perception_parameters, *relation_parameters)
        ),
        complete_model_parameters=_parameter_count(complete_model),
        complete_model_weight_bytes=_parameter_bytes(complete_model),
    )
    return result.validate()


def learning_rate_multiplier(
    update_index: int,
    config: DynamicSetOptimizerConfig = DEFAULT_DYNAMIC_SET_OPTIMIZER_CONFIG,
) -> float:
    """Return the step-one warmup then cosine-to-ten-percent schedule."""

    config.validate()
    if isinstance(update_index, bool) or not isinstance(update_index, int) or update_index < 0:
        raise ValueError("update_index must be a nonnegative integer")
    if update_index < config.warmup_updates:
        return (update_index + 1) / config.warmup_updates
    progress = min(
        1.0,
        (update_index - config.warmup_updates) / (config.maximum_updates - config.warmup_updates),
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return (
        config.final_learning_rate_fraction + (1.0 - config.final_learning_rate_fraction) * cosine
    )


def build_dynamic_set_optimizer(
    complete_model: nn.Module,
    *,
    perception: nn.Module,
    relation: nn.Module,
    legacy_modules: Sequence[nn.Module],
    config: DynamicSetOptimizerConfig = DEFAULT_DYNAMIC_SET_OPTIMIZER_CONFIG,
) -> tuple[torch.optim.AdamW, torch.optim.lr_scheduler.LambdaLR, DynamicSetCapacity]:
    """Freeze non-owners and return exactly two AdamW parameter groups."""

    config.validate()
    capacity = dynamic_set_capacity(
        complete_model,
        perception=perception,
        relation=relation,
        legacy_modules=legacy_modules,
    )
    complete_model.requires_grad_(False)
    perception.requires_grad_(True)
    relation.requires_grad_(True)
    perception_parameters = tuple(perception.parameters())
    relation_parameters = tuple(relation.parameters())
    if not perception_parameters or not relation_parameters:
        raise ValueError("both dynamic-set optimizer owners must have parameters")
    optimizer = torch.optim.AdamW(
        (
            {
                "name": "perception",
                "params": perception_parameters,
                "lr": config.perception_learning_rate,
            },
            {
                "name": "relation",
                "params": relation_parameters,
                "lr": config.relation_learning_rate,
            },
        ),
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda update_index: learning_rate_multiplier(update_index, config),
    )
    return optimizer, scheduler, capacity


__all__ = [
    "DynamicSetCapacity",
    "DynamicSetOptimizerConfig",
    "DEFAULT_DYNAMIC_SET_OPTIMIZER_CONFIG",
    "LEARNED_WEIGHT_BYTE_LIMIT",
    "PERCEPTION_PARAMETER_LIMIT",
    "RELATION_PARAMETER_LIMIT",
    "TOTAL_NEW_PARAMETER_LIMIT",
    "build_dynamic_set_optimizer",
    "dynamic_set_capacity",
    "learning_rate_multiplier",
]
