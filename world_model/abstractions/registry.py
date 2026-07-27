"""Registry of implemented executable predictive abstractions."""

from __future__ import annotations

from collections.abc import Iterable

from world_model.abstractions.contracts import (
    AbstractionKind,
    AbstractionSpec,
)


class AbstractionRegistry:
    """Small explicit registry; no dynamic plugin infrastructure is required."""

    def __init__(self, specs: Iterable[AbstractionSpec] = ()) -> None:
        self._specs: dict[AbstractionKind, AbstractionSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: AbstractionSpec) -> None:
        if spec.kind in self._specs:
            raise ValueError(f"abstraction {spec.kind.name} is already registered")
        self._specs[spec.kind] = spec

    def resolve(self, kind: AbstractionKind | int) -> AbstractionSpec:
        try:
            normalized = AbstractionKind(int(kind))
            return self._specs[normalized]
        except (KeyError, TypeError, ValueError) as exc:
            raise KeyError(f"unregistered predictive abstraction {kind!r}") from exc

    @property
    def kinds(self) -> tuple[AbstractionKind, ...]:
        return tuple(self._specs)


def default_abstraction_registry() -> AbstractionRegistry:
    """Return the executable abstractions supported by the current runtime."""

    return AbstractionRegistry(
        (
            AbstractionSpec(
                kind=AbstractionKind.POINT_TRAJECTORY,
                name="point_trajectory",
                execution_operator="analytic_kinematics",
                required_state_fields=("position", "velocity", "fast_log_variance"),
                complexity_cost=1.0,
            ),
            AbstractionSpec(
                kind=AbstractionKind.RIGID_SPHERE,
                name="rigid_sphere",
                execution_operator="sphere_contact_resolver",
                required_state_fields=(
                    "position",
                    "velocity",
                    "geometry",
                    "log_mass",
                    "restitution_logit",
                    "friction_logit",
                ),
                complexity_cost=2.0,
            ),
        )
    )
