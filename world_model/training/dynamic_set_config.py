"""Strict specification-1.61 configuration without mutating frozen config source.

The specification-1.60 scene certificate binds ``world_model.utils.config``
byte-for-byte.  These dataclass extensions and their loader are consequently
owned by the dormant dynamic-set rung rather than the accepted package-wide
configuration module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml

from world_model.utils.config import (
    DynamicsConfig as _BaseDynamicsConfig,
)
from world_model.utils.config import (
    ModelConfig as _BaseModelConfig,
)
from world_model.utils.config import (
    OrpheusConfig as _BaseOrpheusConfig,
)
from world_model.utils.config import (
    RGBDConfig as _BaseRGBDConfig,
)
from world_model.utils.config import _deep_merge, _strict_construct, parse_overrides
from world_model.utils.config import save_resolved_config as _save_resolved_config


@dataclass(frozen=True)
class RGBDConfig(_BaseRGBDConfig):
    """Dynamic-set additions to the frozen public RGB-D configuration."""

    observation_mode: str = "legacy"
    max_objects: int = 6
    birth_proposals: int = 2
    set_feature_dim: int = 32
    set_log_variance_residual_limit: float = 4.0


@dataclass(frozen=True)
class DynamicsConfig(_BaseDynamicsConfig):
    """Relation-only/contact execution switches for specification 1.61."""

    relation_hidden_dim: int | None = None
    modal_dynamics_enabled: bool = True
    continuous_pair_force_enabled: bool = True
    node_acceleration_enabled: bool = True
    event_driven_state_only_enabled: bool = False
    relation_process_uncertainty_enabled: bool = False
    packed_interactions_enabled: bool = False


@dataclass(frozen=True)
class ModelConfig(_BaseModelConfig):
    rgbd: RGBDConfig = field(default_factory=RGBDConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)


def _base_dataclass(cls: type[Any], value: object, **overrides: object) -> Any:
    values = {item.name: getattr(value, item.name) for item in fields(cls)}
    values.update(overrides)
    return cls(**values)


@dataclass(frozen=True)
class OrpheusConfig(_BaseOrpheusConfig):
    """Package configuration plus the explicitly scoped dynamic-set fields."""

    model: ModelConfig = field(default_factory=ModelConfig)

    def _base_validation_projection(self) -> _BaseOrpheusConfig:
        """Project onto the frozen validator without weakening legacy profiles."""

        rgbd = _base_dataclass(_BaseRGBDConfig, self.model.rgbd)
        dynamics = _base_dataclass(_BaseDynamicsConfig, self.model.dynamics)
        simulator = self.simulator
        model_max_objects = self.model.max_objects
        filter_config = self.model.filter
        if self.model.rgbd.observation_mode == "set":
            # The frozen validator intentionally knows only one/two-proposal
            # RGB-D.  Validate all shared fields through an inert one-slot
            # projection, then validate the true set/count semantics below.
            rgbd = replace(
                rgbd,
                proposal_count=1,
                temporal_min_samples=rgbd.temporal_history_size,
            )
            simulator = replace(
                simulator,
                min_objects=1,
                max_objects=1,
                radius_range=(rgbd.world_radius, rgbd.world_radius),
                drag_range=(rgbd.linear_drag, rgbd.linear_drag),
            )
            model_max_objects = 1
            filter_config = replace(filter_config, direct_metric_position_update=False)
        model = _base_dataclass(
            _BaseModelConfig,
            self.model,
            max_objects=model_max_objects,
            rgbd=rgbd,
            dynamics=dynamics,
            filter=filter_config,
        )
        return _BaseOrpheusConfig(
            project=self.project,
            device=self.device,
            simulator=simulator,
            model=model,
            runtime=self.runtime,
            training=self.training,
            evaluation=self.evaluation,
            demo=self.demo,
            source_path=self.source_path,
        )

    def validate(self) -> None:
        self._base_validation_projection().validate()
        simulator = self.simulator
        model = self.model
        rgbd = model.rgbd
        dynamics = model.dynamics

        if rgbd.observation_mode not in {"legacy", "set"}:
            raise ValueError("model.rgbd.observation_mode must be 'legacy' or 'set'")
        if (
            isinstance(rgbd.max_objects, bool)
            or not isinstance(rgbd.max_objects, int)
            or (rgbd.max_objects <= 0)
        ):
            raise ValueError("model.rgbd.max_objects must be a positive integer")
        if (
            isinstance(rgbd.birth_proposals, bool)
            or not isinstance(rgbd.birth_proposals, int)
            or rgbd.birth_proposals <= 0
        ):
            raise ValueError("model.rgbd.birth_proposals must be a positive integer")
        if isinstance(rgbd.proposal_count, bool) or not isinstance(rgbd.proposal_count, int):
            if rgbd.observation_mode == "legacy":
                raise ValueError("model.rgbd.proposal_count must be integer one or two")
            raise ValueError("model.rgbd.proposal_count must be an integer")
        if rgbd.observation_mode == "legacy" and rgbd.proposal_count not in {1, 2}:
            raise ValueError("model.rgbd.proposal_count must be integer one or two in legacy mode")
        if rgbd.observation_mode == "set" and rgbd.proposal_count != (
            rgbd.max_objects + rgbd.birth_proposals
        ):
            raise ValueError(
                "set model.rgbd proposal_count must equal max_objects + birth_proposals"
            )
        if rgbd.observation_mode == "legacy" and rgbd.birth_proposals != 2:
            raise ValueError("legacy model.rgbd requires birth_proposals=2")
        if (
            isinstance(rgbd.set_feature_dim, bool)
            or not isinstance(rgbd.set_feature_dim, int)
            or rgbd.set_feature_dim not in {32, 64}
        ):
            raise ValueError("model.rgbd.set_feature_dim must be 32 or 64")
        if rgbd.observation_mode == "legacy" and rgbd.set_feature_dim != 32:
            raise ValueError("legacy model.rgbd requires set_feature_dim=32")
        if (
            isinstance(rgbd.set_log_variance_residual_limit, bool)
            or not isinstance(rgbd.set_log_variance_residual_limit, (int, float))
            or not math.isfinite(float(rgbd.set_log_variance_residual_limit))
            or not 0.0 < float(rgbd.set_log_variance_residual_limit) <= 32.0
        ):
            raise ValueError(
                "model.rgbd.set_log_variance_residual_limit must be finite and lie in (0,32]"
            )
        if rgbd.observation_mode == "legacy" and rgbd.set_log_variance_residual_limit != 4.0:
            raise ValueError("legacy model.rgbd requires set_log_variance_residual_limit=4")
        if rgbd.observation_mode == "legacy" and (
            rgbd.temporal_min_samples != rgbd.temporal_history_size
        ):
            raise ValueError(
                "legacy model.rgbd.temporal_min_samples must equal temporal_history_size"
            )
        if rgbd.observation_mode == "set" and rgbd.temporal_min_samples != 3:
            raise ValueError("set model.rgbd.temporal_min_samples must equal three")

        if rgbd.enabled:
            if rgbd.observation_mode == "set":
                if model.max_objects != rgbd.max_objects:
                    raise ValueError("set model.rgbd max_objects must equal model.max_objects")
                if simulator.min_objects < 1 or simulator.max_objects > rgbd.max_objects:
                    raise ValueError(
                        "set simulator object counts must fit inside model.rgbd.max_objects"
                    )
                if model.state.appearance_dim != 8:
                    raise ValueError("set model.rgbd requires model.state.appearance_dim=8")
            if model.filter.direct_metric_position_update:
                if self.runtime.modality != "rgbd":
                    raise ValueError("direct metric position updates require the RGB-D runtime")
                if model.rgb.enabled or self.runtime.enable_debug_oracle:
                    raise ValueError(
                        "direct metric position updates require an exclusive RGB-D observation path"
                    )
                if model.filter.enable_learned_corrector:
                    raise ValueError(
                        "direct metric position updates require the learned corrector disabled"
                    )

        for name, value in (
            ("modal_dynamics_enabled", dynamics.modal_dynamics_enabled),
            ("continuous_pair_force_enabled", dynamics.continuous_pair_force_enabled),
            ("node_acceleration_enabled", dynamics.node_acceleration_enabled),
            ("event_driven_state_only_enabled", dynamics.event_driven_state_only_enabled),
            (
                "relation_process_uncertainty_enabled",
                dynamics.relation_process_uncertainty_enabled,
            ),
            ("packed_interactions_enabled", dynamics.packed_interactions_enabled),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"model.dynamics.{name} must be boolean")
        if dynamics.relation_hidden_dim is not None and (
            isinstance(dynamics.relation_hidden_dim, bool)
            or not isinstance(dynamics.relation_hidden_dim, int)
            or dynamics.relation_hidden_dim <= 0
        ):
            raise ValueError("model.dynamics.relation_hidden_dim must be a positive integer")
        if dynamics.event_driven_state_only_enabled:
            if (
                dynamics.modal_dynamics_enabled
                or dynamics.continuous_pair_force_enabled
                or dynamics.node_acceleration_enabled
                or dynamics.attention_residual_enabled
            ):
                raise ValueError(
                    "event-driven state-only dynamics requires relation-only pair impulses"
                )
            if tuple(float(value) for value in simulator.gravity) != (0.0, 0.0, 0.0):
                raise ValueError("event-driven state-only dynamics requires zero gravity")


DynamicSetOrpheusConfig = OrpheusConfig


def load_config(
    path: str | Path,
    *,
    overrides: list[str] | tuple[str, ...] = (),
) -> OrpheusConfig:
    """Load and validate a configuration through the 1.61 extension types."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Configuration file not found: {source}")
    loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise TypeError("Top-level configuration must be a mapping")
    merged = _deep_merge(loaded, parse_overrides(overrides))
    model_values = merged.get("model")
    if isinstance(model_values, dict):
        rgbd_values = model_values.get("rgbd")
        if isinstance(rgbd_values, dict) and rgbd_values.get("observation_mode") == "set":
            max_objects = int(rgbd_values.get("max_objects", model_values.get("max_objects", 6)))
            birth_proposals = int(rgbd_values.get("birth_proposals", 2))
            rgbd_values.setdefault("max_objects", max_objects)
            rgbd_values.setdefault("birth_proposals", birth_proposals)
            rgbd_values.setdefault("proposal_count", max_objects + birth_proposals)
    merged["source_path"] = str(source)
    config = _strict_construct(OrpheusConfig, merged, "config")
    config.validate()
    return config


load_dynamic_set_config = load_config


def save_resolved_config(config: OrpheusConfig, path: str | Path) -> None:
    """Persist a validated dynamic-set config using the shared YAML encoding."""

    _save_resolved_config(config, path)


__all__ = [
    "DynamicSetOrpheusConfig",
    "DynamicsConfig",
    "ModelConfig",
    "OrpheusConfig",
    "RGBDConfig",
    "load_config",
    "load_dynamic_set_config",
    "save_resolved_config",
]
