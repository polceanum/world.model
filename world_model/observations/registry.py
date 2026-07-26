"""Explicit observation-module registry."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TypeVar

from world_model.observations.base import ObservationModule

ModuleType = TypeVar("ModuleType", bound=type[ObservationModule])
OBSERVATION_MODULES: dict[str, type[ObservationModule]] = {}


def register_observation_module(
    name: str,
) -> Callable[[ModuleType], ModuleType]:
    if not name:
        raise ValueError("observation module name must be non-empty")

    def decorator(module_type: ModuleType) -> ModuleType:
        existing = OBSERVATION_MODULES.get(name)
        if existing is not None and existing is not module_type:
            raise ValueError(f"observation modality {name!r} is already registered")
        if getattr(module_type, "modality_name", None) != name:
            raise ValueError(
                f"{module_type.__name__}.modality_name must equal registered name {name!r}"
            )
        OBSERVATION_MODULES[name] = module_type
        return module_type

    return decorator


def observation_module_type(name: str) -> type[ObservationModule]:
    try:
        return OBSERVATION_MODULES[name]
    except KeyError as exc:
        available = ", ".join(sorted(OBSERVATION_MODULES)) or "<none>"
        raise KeyError(f"unknown observation modality {name!r}; registered: {available}") from exc


def validate_module_mapping(modules: Mapping[str, ObservationModule]) -> None:
    for name, module in modules.items():
        if name != module.modality_name:
            raise ValueError(
                f"module mapping key {name!r} does not match modality {module.modality_name!r}"
            )
