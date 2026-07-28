"""Strict, dependency-light YAML configuration for Project Orpheus."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

import yaml

T = TypeVar("T")


@dataclass(frozen=True)
class ProjectConfig:
    name: str = "orpheus"
    seed: int = 42
    output_dir: str = "runs"
    deterministic: bool = False


@dataclass(frozen=True)
class DeviceConfig:
    preference: str = "auto"
    cuda_amp: bool = True
    mps_float32: bool = True
    compile: bool = False


@dataclass(frozen=True)
class SimulatorConfig:
    type: str = "sphere_world"
    image_size: tuple[int, int] = (96, 96)
    frame_rate: float = 30.0
    physics_rate: float = 120.0
    sequence_frames: int = 72
    min_objects: int = 3
    max_objects: int = 6
    world_bounds: tuple[tuple[float, float], ...] = (
        (-2.25, 2.25),
        (0.0, 3.25),
        (-1.5, 1.5),
    )
    radius_range: tuple[float, float] = (0.16, 0.28)
    mass_range: tuple[float, float] = (0.6, 1.8)
    restitution_range: tuple[float, float] = (0.45, 0.9)
    drag_range: tuple[float, float] = (0.01, 0.16)
    friction_range: tuple[float, float] = (0.05, 0.35)
    initial_speed_range: tuple[float, float] = (0.35, 1.35)
    ensured_pair_height_range: tuple[float, float] = (1.1, 1.35)
    ensured_pair_surface_gap_range: tuple[float, float] = (0.75, 0.9)
    ensured_pair_speed_range: tuple[float, float] = (0.85, 1.25)
    gravity: tuple[float, float, float] = (0.0, -9.81, 0.0)
    camera_motion: str = "orbit"
    known_camera_pose: bool = True
    render_noise_std: float = 0.01
    ensure_collision: bool = True
    external_impulse_probability: float = 0.0
    external_impulse_range: tuple[float, float] = (0.15, 0.6)
    scenario_mixture: tuple[str, ...] = ("baseline",)
    split_train_start: int = 0
    split_validation_start: int = 100000
    split_test_start: int = 200000
    split_ood_start: int = 300000


@dataclass(frozen=True)
class StateConfig:
    geometry_dim: int = 8
    appearance_dim: int = 32
    residual_dynamics_dim: int = 16
    modal_count: int = 4
    modal_dim: int = 3
    parameter_memory_dim: int = 48
    global_dim: int = 16
    fast_log_variance_min: float = -12.0
    fast_log_variance_max: float = 6.0
    slow_log_variance_min: float = -12.0
    slow_log_variance_max: float = 8.0


@dataclass(frozen=True)
class RGBConfig:
    enabled: bool = True
    backbone_channels: tuple[int, ...] = (32, 64, 96, 128)
    feature_dim: int = 96
    proposal_queries: int = 10
    global_every_steps: int = 12
    roi_size: int = 20
    fast_depth_residual_enabled: bool = False
    temporal_velocity_enabled: bool = False
    temporal_velocity_history_size: int = 3
    temporal_velocity_min_samples: int = 3
    temporal_velocity_min_dt: float = 1.0e-3
    temporal_velocity_variance_scale: float = 1.0
    temporal_velocity_variance_floor: float = 0.25
    temporal_velocity_variance_ceiling: float | None = None
    temporal_velocity_lateral_only: bool = False
    temporal_velocity_unobserved_variance: float = 1.0e4
    temporal_velocity_reset_on_collision: bool = False
    temporal_velocity_max_age_steps: int | None = None
    temporal_velocity_post_event_max_samples: int | None = None
    temporal_velocity_measurement_position_blend: float = 0.0
    temporal_velocity_position_innovation_coupling: bool = False
    structured_disc_center_enabled: bool = False
    structured_disc_threshold: float = 0.04
    structured_disc_min_pixels: int = 4
    structured_disc_max_assignment_distance: float = 0.75
    structured_disc_center_std_pixels: float = 0.75
    structured_disc_depth_relative_std: float | None = None
    structured_disc_depth_outlier_relative_threshold: float | None = None
    structured_disc_depth_outlier_variance_scale: float = 9.0
    structured_disc_position_confidence: float | None = None
    roi_uncertainty_scale: float = 2.5
    global_uncertainty_threshold: float = 4.0
    surprise_threshold: float = 8.0
    existence_threshold: float = 0.45
    measurement_log_variance_min: float = -9.0
    measurement_log_variance_max: float = 4.0


@dataclass(frozen=True)
class DynamicsConfig:
    max_substep: float = 1.0 / 120.0
    hidden_dim: int = 96
    interaction_radius: float = 1.0
    process_noise_position: float = 1e-4
    process_noise_velocity: float = 2e-3
    modal_acceleration_scale: float = 0.25
    residual_acceleration_scale: float = 0.5
    penetration_slop: float = 1e-3
    max_penetration_correction: float = 0.08
    contact_confidence_sigma: float = 0.25
    sleep_speed: float = 0.05
    allow_large_substep: bool = False


@dataclass(frozen=True)
class FilterConfig:
    hidden_dim: int = 128
    robust_clip: float = 5.0
    min_log_variance: float = -12.0
    max_log_variance: float = 8.0
    learned_residual_scale: float = 0.15
    missed_variance_growth: float = 0.08


@dataclass(frozen=True)
class AssociationConfig:
    geometry_weight: float = 1.0
    appearance_weight: float = 0.25
    existence_weight: float = 0.05
    mahalanobis_gate: float = 16.0
    maximum_cost: float = 25.0
    ambiguity_margin: float = 0.02


@dataclass(frozen=True)
class LifecycleConfig:
    birth_confidence: float = 0.55
    birth_confirmations: int = 1
    max_missed_steps: int = 12
    existence_decay: float = 0.35
    occlusion_existence_decay: float = 0.04


@dataclass(frozen=True)
class IdentificationConfig:
    enabled: bool = True
    hidden_dim: int = 48
    slow_learning_rate: float = 0.04
    drag_speed_threshold: float = 0.25
    restitution_event_threshold: float = 0.4
    ambiguity_gate: float = 0.2


@dataclass(frozen=True)
class ModelConfig:
    max_objects: int = 8
    state: StateConfig = field(default_factory=StateConfig)
    rgb: RGBConfig = field(default_factory=RGBConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    association: AssociationConfig = field(default_factory=AssociationConfig)
    lifecycle: LifecycleConfig = field(default_factory=LifecycleConfig)
    identification: IdentificationConfig = field(default_factory=IdentificationConfig)


@dataclass(frozen=True)
class RuntimeConfig:
    modality: str = "rgb"
    enable_debug_oracle: bool = False
    strict_timestamps: bool = True
    modality_order: tuple[str, ...] = ("debug_oracle", "rgb")


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 4
    steps: int = 1000
    learning_rate: float = 3e-4
    closed_loop_learning_rate_scale: float = 0.1
    closed_loop_global_trainable_steps: int = 50
    closed_loop_trainable_scope: str = "all"
    weight_decay: float = 1e-4
    tbptt_steps: int = 24
    grad_clip_norm: float = 1.0
    checkpoint_every: int = 100
    eval_every: int = 100
    log_every: int = 10
    train_episodes: int = 256
    validation_episodes: int = 16
    num_workers: int = 0
    fixed_dataset: bool = False
    rgb_pretrain_steps: int = 100
    measurement_validation_frames: int = 8
    perturbation_probability: float = 0.25
    perturbation_position_std: float = 0.12
    perturbation_velocity_std: float = 0.20
    collision_window_probability: float = 0.50
    long_horizon_window_probability: float = 0.50
    collision_positive_weight_max: float = 10.0
    horizon_weights: tuple[float, ...] = (1.0, 1.0, 1.2, 1.5, 1.5)
    measurement_loss_weights: dict[str, float] = field(
        default_factory=lambda: {
            "rgb_existence": 1.0,
            "rgb_geometry": 1.0,
            "rgb_colour": 0.25,
            "rgb_nll": 0.05,
            "rgb_visibility": 0.25,
            "rgb_appearance": 0.25,
            "rgb_raw_centre": 2.0,
            "rgb_world_position": 8.0,
            "rgb_world_position_nll": 0.05,
        }
    )
    loss_weights: dict[str, float] = field(
        default_factory=lambda: {
            "measurement": 1.0,
            "state_position": 2.0,
            "state_velocity": 0.25,
            "rollout_position": 4.0,
            "rollout_velocity": 0.1,
            "event": 0.2,
            "parameter": 0.1,
            "existence": 0.2,
            "uncertainty": 0.05,
            "correction": 0.02,
        }
    )


@dataclass(frozen=True)
class EvaluationConfig:
    horizons_seconds: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0, 2.0)
    episodes: int = 16
    rgb_only: bool = True
    perturbation_position_std: float = 0.15
    perturbation_velocity_std: float = 0.25
    confidence_level: float = 0.90
    benchmark_warmup: int = 2


@dataclass(frozen=True)
class DemoConfig:
    seed: int = 200000
    max_frames: int = 48
    future_horizon_seconds: float = 1.0
    fps: int = 12


@dataclass(frozen=True)
class OrpheusConfig:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    demo: DemoConfig = field(default_factory=DemoConfig)
    source_path: str | None = field(default=None, repr=False, compare=False)

    def validate(self) -> None:
        """Validate architecture-sensitive constraints from ``PROJECT_SPEC.md``."""

        simulator = self.simulator
        model = self.model
        if simulator.type != "sphere_world":
            raise ValueError(f"Unsupported simulator type {simulator.type!r}")
        if len(simulator.image_size) != 2 or any(size <= 0 for size in simulator.image_size):
            raise ValueError("simulator.image_size must contain two positive integers")
        if simulator.frame_rate <= 0 or simulator.physics_rate <= 0:
            raise ValueError("simulator frame_rate and physics_rate must be positive")
        if simulator.sequence_frames < 2:
            raise ValueError("simulator.sequence_frames must be at least 2")
        if simulator.min_objects <= 0 or simulator.max_objects < simulator.min_objects:
            raise ValueError("invalid simulator object-count range")
        if len(simulator.world_bounds) != 3 or any(
            len(bounds) != 2 or bounds[0] >= bounds[1] for bounds in simulator.world_bounds
        ):
            raise ValueError("simulator.world_bounds must contain three increasing pairs")
        if model.max_objects < simulator.max_objects:
            raise ValueError("model.max_objects must be >= simulator.max_objects")
        if model.state.modal_count < 0 or model.state.modal_dim <= 0:
            raise ValueError("modal_count must be nonnegative and modal_dim positive")
        if model.lifecycle.max_missed_steps <= 0:
            raise ValueError("model.lifecycle.max_missed_steps must be positive")
        if (
            not math.isfinite(model.rgb.temporal_velocity_min_dt)
            or model.rgb.temporal_velocity_min_dt <= 0
        ):
            raise ValueError("model.rgb.temporal_velocity_min_dt must be finite and positive")
        if model.rgb.temporal_velocity_history_size < 3:
            raise ValueError("model.rgb.temporal_velocity_history_size must be at least three")
        if (
            not 2
            <= model.rgb.temporal_velocity_min_samples
            <= (model.rgb.temporal_velocity_history_size)
        ):
            raise ValueError(
                "model.rgb.temporal_velocity_min_samples must lie between two and history_size"
            )
        if (
            not math.isfinite(model.rgb.temporal_velocity_variance_scale)
            or model.rgb.temporal_velocity_variance_scale < 1
        ):
            raise ValueError(
                "model.rgb.temporal_velocity_variance_scale must be finite and at least one"
            )
        if (
            not math.isfinite(model.rgb.temporal_velocity_variance_floor)
            or model.rgb.temporal_velocity_variance_floor <= 0
        ):
            raise ValueError(
                "model.rgb.temporal_velocity_variance_floor must be finite and positive"
            )
        variance_ceiling = model.rgb.temporal_velocity_variance_ceiling
        if variance_ceiling is not None and (
            not math.isfinite(variance_ceiling)
            or variance_ceiling < model.rgb.temporal_velocity_variance_floor
        ):
            raise ValueError(
                "model.rgb.temporal_velocity_variance_ceiling must be finite "
                "and no smaller than temporal_velocity_variance_floor"
            )
        if (
            not math.isfinite(model.rgb.temporal_velocity_unobserved_variance)
            or model.rgb.temporal_velocity_unobserved_variance
            < model.rgb.temporal_velocity_variance_floor
        ):
            raise ValueError(
                "model.rgb.temporal_velocity_unobserved_variance must be finite "
                "and no smaller than temporal_velocity_variance_floor"
            )
        if (
            model.rgb.temporal_velocity_max_age_steps is not None
            and model.rgb.temporal_velocity_max_age_steps < model.rgb.temporal_velocity_min_samples
        ):
            raise ValueError(
                "model.rgb.temporal_velocity_max_age_steps must be no smaller "
                "than temporal_velocity_min_samples"
            )
        if (
            model.rgb.temporal_velocity_post_event_max_samples is not None
            and model.rgb.temporal_velocity_post_event_max_samples
            < model.rgb.temporal_velocity_min_samples
        ):
            raise ValueError(
                "model.rgb.temporal_velocity_post_event_max_samples must be no smaller "
                "than temporal_velocity_min_samples"
            )
        if (
            not math.isfinite(model.rgb.temporal_velocity_measurement_position_blend)
            or not 0.0 <= model.rgb.temporal_velocity_measurement_position_blend <= 1.0
        ):
            raise ValueError(
                "model.rgb.temporal_velocity_measurement_position_blend must lie in [0, 1]"
            )
        if (
            not math.isfinite(model.rgb.structured_disc_threshold)
            or not 0 < model.rgb.structured_disc_threshold < 2
        ):
            raise ValueError("model.rgb.structured_disc_threshold must lie in (0, 2)")
        if model.rgb.structured_disc_min_pixels <= 0:
            raise ValueError("model.rgb.structured_disc_min_pixels must be positive")
        if (
            not math.isfinite(model.rgb.structured_disc_max_assignment_distance)
            or model.rgb.structured_disc_max_assignment_distance <= 0
        ):
            raise ValueError(
                "model.rgb.structured_disc_max_assignment_distance must be finite and positive"
            )
        if (
            not math.isfinite(model.rgb.structured_disc_center_std_pixels)
            or model.rgb.structured_disc_center_std_pixels <= 0
        ):
            raise ValueError(
                "model.rgb.structured_disc_center_std_pixels must be finite and positive"
            )
        if model.rgb.structured_disc_depth_relative_std is not None and (
            not math.isfinite(model.rgb.structured_disc_depth_relative_std)
            or not 0.0 < model.rgb.structured_disc_depth_relative_std <= 1.0
        ):
            raise ValueError("model.rgb.structured_disc_depth_relative_std must lie in (0, 1]")
        if model.rgb.structured_disc_depth_outlier_relative_threshold is not None and (
            not math.isfinite(model.rgb.structured_disc_depth_outlier_relative_threshold)
            or model.rgb.structured_disc_depth_outlier_relative_threshold <= 0.0
        ):
            raise ValueError(
                "model.rgb.structured_disc_depth_outlier_relative_threshold "
                "must be finite and positive"
            )
        if (
            not math.isfinite(model.rgb.structured_disc_depth_outlier_variance_scale)
            or model.rgb.structured_disc_depth_outlier_variance_scale < 1.0
        ):
            raise ValueError(
                "model.rgb.structured_disc_depth_outlier_variance_scale "
                "must be finite and at least one"
            )
        if model.rgb.structured_disc_position_confidence is not None and (
            not math.isfinite(model.rgb.structured_disc_position_confidence)
            or not 0.0 < model.rgb.structured_disc_position_confidence <= 1.0
        ):
            raise ValueError("model.rgb.structured_disc_position_confidence must lie in (0, 1]")
        if not (
            0.0 <= model.lifecycle.occlusion_existence_decay <= model.lifecycle.existence_decay
        ):
            raise ValueError(
                "model.lifecycle occlusion_existence_decay must lie between "
                "zero and existence_decay"
            )
        if model.dynamics.max_substep <= 0:
            raise ValueError("model.dynamics.max_substep must be positive")
        if model.dynamics.contact_confidence_sigma < 0:
            raise ValueError("model.dynamics.contact_confidence_sigma must be nonnegative")
        observation_dt = 1.0 / simulator.frame_rate
        if model.dynamics.max_substep > observation_dt and not model.dynamics.allow_large_substep:
            raise ValueError(
                "model.dynamics.max_substep exceeds the observation timestep; "
                "set allow_large_substep only for an explicit ablation"
            )
        for name, bounds in (
            ("radius", simulator.radius_range),
            ("mass", simulator.mass_range),
            ("restitution", simulator.restitution_range),
            ("drag", simulator.drag_range),
            ("friction", simulator.friction_range),
            ("initial_speed", simulator.initial_speed_range),
            ("ensured_pair_height", simulator.ensured_pair_height_range),
            ("ensured_pair_surface_gap", simulator.ensured_pair_surface_gap_range),
            ("ensured_pair_speed", simulator.ensured_pair_speed_range),
            ("external_impulse", simulator.external_impulse_range),
        ):
            if len(bounds) != 2 or bounds[0] > bounds[1]:
                raise ValueError(f"invalid simulator {name}_range")
        if not (0 <= simulator.restitution_range[0] <= simulator.restitution_range[1] <= 1):
            raise ValueError("restitution_range must lie in [0, 1]")
        if not (0 <= simulator.friction_range[0] <= simulator.friction_range[1] <= 1):
            raise ValueError("friction_range must lie in [0, 1]")
        if not simulator.scenario_mixture:
            raise ValueError("simulator.scenario_mixture must contain at least one scenario")
        supported_scenarios = {
            "baseline",
            "reference_pairs",
            "elastic_pairs",
            "damped_contacts",
            "impulse_perturbation",
            "camera_parallax",
            "glancing_impacts",
            "heavy_light_impacts",
        }
        unknown_scenarios = set(simulator.scenario_mixture) - supported_scenarios
        if unknown_scenarios:
            raise ValueError(f"unsupported simulator scenarios: {sorted(unknown_scenarios)}")
        if self.device.preference not in {"auto", "cpu", "mps", "cuda"}:
            raise ValueError(f"Unsupported device preference {self.device.preference!r}")
        if self.runtime.modality not in {"rgb", "debug_oracle"}:
            raise ValueError(f"Unsupported runtime modality {self.runtime.modality!r}")
        if not self.runtime.strict_timestamps:
            raise ValueError(
                "Milestone 1 requires runtime.strict_timestamps=true; "
                "out-of-sequence buffering is not implemented"
            )
        if self.runtime.modality == "rgb" and not simulator.known_camera_pose:
            raise ValueError("Milestone 1 RGB requires known_camera_pose=true")
        if self.evaluation.rgb_only and (
            self.runtime.modality == "debug_oracle" or self.runtime.enable_debug_oracle
        ):
            raise ValueError("RGB-only evaluation cannot enable debug oracle input")
        if any(horizon <= 0 for horizon in self.evaluation.horizons_seconds):
            raise ValueError("evaluation horizons must be positive")
        if len(self.training.horizon_weights) != len(self.evaluation.horizons_seconds):
            raise ValueError("training.horizon_weights must match evaluation.horizons_seconds")
        if any(weight <= 0 for weight in self.training.horizon_weights):
            raise ValueError("training horizon weights must be positive")
        if not self.training.measurement_loss_weights or any(
            not math.isfinite(weight) or weight < 0
            for weight in self.training.measurement_loss_weights.values()
        ):
            raise ValueError(
                "training.measurement_loss_weights must be nonempty, finite, and nonnegative"
            )
        episode_duration = (simulator.sequence_frames - 1) / simulator.frame_rate
        if max(self.evaluation.horizons_seconds) > episode_duration + 1e-9:
            raise ValueError(
                "evaluation horizon exceeds generated episode duration "
                f"({max(self.evaluation.horizons_seconds):.3f}s > {episode_duration:.3f}s)"
            )
        if self.training.batch_size <= 0 or self.training.steps < 0:
            raise ValueError("training batch_size must be positive and steps nonnegative")
        if self.training.learning_rate <= 0:
            raise ValueError("training.learning_rate must be positive")
        if not 0 < self.training.closed_loop_learning_rate_scale <= 1:
            raise ValueError("training.closed_loop_learning_rate_scale must lie in (0, 1]")
        if self.training.closed_loop_global_trainable_steps < 0:
            raise ValueError("training.closed_loop_global_trainable_steps must be nonnegative")
        if self.training.closed_loop_trainable_scope not in {"all", "dynamics"}:
            raise ValueError("training.closed_loop_trainable_scope must be 'all' or 'dynamics'")
        if self.training.tbptt_steps <= 0:
            raise ValueError("training.tbptt_steps must be positive")
        if self.training.measurement_validation_frames <= 0:
            raise ValueError("training.measurement_validation_frames must be positive")
        if not 0 <= self.training.perturbation_probability <= 1:
            raise ValueError("training.perturbation_probability must lie in [0, 1]")
        if (
            not math.isfinite(self.training.perturbation_position_std)
            or self.training.perturbation_position_std <= 0
        ):
            raise ValueError("training.perturbation_position_std must be finite and positive")
        if (
            not math.isfinite(self.training.perturbation_velocity_std)
            or self.training.perturbation_velocity_std <= 0
        ):
            raise ValueError("training.perturbation_velocity_std must be finite and positive")
        if not 0 <= self.training.collision_window_probability <= 1:
            raise ValueError("training.collision_window_probability must lie in [0, 1]")
        if not 0 <= self.training.long_horizon_window_probability <= 1:
            raise ValueError("training.long_horizon_window_probability must lie in [0, 1]")
        if self.training.collision_positive_weight_max < 1:
            raise ValueError("training.collision_positive_weight_max must be at least one")

    def to_dict(self) -> dict[str, Any]:
        """Return a YAML-safe resolved representation."""

        data = asdict(self)
        data.pop("source_path", None)
        return _yaml_safe(data)


def _yaml_safe(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_yaml_safe(item) for item in value]
    if isinstance(value, list):
        return [_yaml_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _yaml_safe(item) for key, item in value.items()}
    return value


def _convert_value(annotation: Any, value: Any, path: str) -> Any:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if is_dataclass(annotation):
        if not isinstance(value, Mapping):
            raise TypeError(f"{path} must be a mapping")
        return _strict_construct(annotation, value, path)
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{path} must be a sequence")
        item_type = args[0] if args else Any
        return tuple(_convert_value(item_type, item, f"{path}[]") for item in value)
    if origin is list:
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{path} must be a sequence")
        item_type = args[0] if args else Any
        return [_convert_value(item_type, item, f"{path}[]") for item in value]
    if origin is dict:
        if not isinstance(value, Mapping):
            raise TypeError(f"{path} must be a mapping")
        return dict(value)
    if origin in {Union, UnionType}:
        non_none = [argument for argument in args if argument is not type(None)]
        if value is None and len(non_none) != len(args):
            return None
        if len(non_none) == 1:
            return _convert_value(non_none[0], value, path)
    return value


def _strict_construct(cls: type[T], data: Mapping[str, Any], path: str) -> T:
    names = {item.name for item in fields(cls)}
    unknown = sorted(set(data) - names)
    if unknown:
        raise KeyError(f"Unknown configuration key(s) at {path}: {', '.join(unknown)}")
    hints = get_type_hints(cls)
    values: dict[str, Any] = {}
    for name, value in data.items():
        values[name] = _convert_value(hints[name], value, f"{path}.{name}")
    return cls(**values)


def _set_dotted(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor: dict[str, Any] = data
    for part in parts[:-1]:
        nested = cursor.get(part)
        if nested is None:
            nested = {}
            cursor[part] = nested
        if not isinstance(nested, dict):
            raise KeyError(f"Cannot set {dotted_key!r}; {part!r} is not a mapping")
        cursor = nested
    cursor[parts[-1]] = value


def parse_overrides(overrides: list[str] | tuple[str, ...]) -> dict[str, Any]:
    """Parse conservative ``key=value`` overrides using YAML scalar decoding."""

    result: dict[str, Any] = {}
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must have key=value form: {override!r}")
        key, raw_value = override.split("=", 1)
        if not key or any(not part for part in key.split(".")):
            raise ValueError(f"Invalid override key {key!r}")
        _set_dotted(result, key, yaml.safe_load(raw_value))
    return result


def _deep_merge(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def load_config(
    path: str | Path,
    *,
    overrides: list[str] | tuple[str, ...] = (),
) -> OrpheusConfig:
    """Load strict YAML, apply dotted overrides, and validate the resolved config."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Configuration file not found: {source}")
    loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise TypeError("Top-level configuration must be a mapping")
    merged = _deep_merge(loaded, parse_overrides(overrides))
    merged["source_path"] = str(source)
    config = _strict_construct(OrpheusConfig, merged, "config")
    config.validate()
    return config


def save_resolved_config(config: OrpheusConfig, path: str | Path) -> None:
    """Save the validated configuration in plain YAML."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
