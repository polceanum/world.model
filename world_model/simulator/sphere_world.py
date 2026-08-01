"""Deterministic padded 3-D sphere world used by Project Orpheus."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass, replace
from typing import Any

import torch
from torch import Tensor

from world_model.simulator.camera import (
    CameraFrame,
    CameraTrajectory,
    CameraTrajectoryConfig,
)
from world_model.simulator.physics import (
    PhysicsConfig,
    PhysicsStepEvents,
    SphereState,
    advance_spheres,
)
from world_model.simulator.renderer import RenderOutput, render_spheres


def _plain_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "__dict__"):
        return {key: item for key, item in vars(value).items() if not key.startswith("_")}
    raise TypeError("simulator config must be a mapping or dataclass-like object")


def _tuple_range(value: Any, *, name: str) -> tuple[float, float]:
    result = tuple(float(item) for item in value)
    if len(result) != 2 or result[1] < result[0]:
        raise ValueError(f"{name} must be an increasing pair")
    return result


@dataclass(frozen=True)
class SphereWorldConfig:
    """Resolved simulator configuration for one dataset family."""

    image_size: tuple[int, int] = (64, 64)
    frame_rate: float = 30.0
    physics_rate: float = 120.0
    sequence_frames: int = 24
    min_objects: int = 2
    max_objects: int = 3
    padding_max_objects: int | None = None
    gravity: tuple[float, float, float] = (0.0, -9.81, 0.0)
    world_bounds: tuple[tuple[float, float], ...] = (
        (-2.25, 2.25),
        (0.0, 3.25),
        (-1.5, 1.5),
    )
    radius_range: tuple[float, float] = (0.16, 0.26)
    mass_range: tuple[float, float] = (0.65, 1.8)
    restitution_range: tuple[float, float] = (0.45, 0.9)
    drag_range: tuple[float, float] = (0.01, 0.16)
    friction_range: tuple[float, float] = (0.04, 0.3)
    initial_speed_range: tuple[float, float] = (0.35, 1.35)
    ensured_pair_height_range: tuple[float, float] = (1.1, 1.35)
    ensured_pair_surface_gap_range: tuple[float, float] = (0.75, 0.9)
    ensured_pair_speed_range: tuple[float, float] = (0.85, 1.25)
    ensured_pair_lateral_offset_range: tuple[float, float] = (0.0, 0.0)
    camera_motion: str = "fixed"
    camera_fov_degrees: float = 48.0
    camera_translation_amplitude: float = 0.35
    camera_orbit_amplitude: float = 0.32
    camera_zoom_amplitude: float = 0.0
    render_noise_std: float = 0.0
    edge_softness_pixels: float = 1.0
    solver_iterations: int = 2
    ensure_collision: bool = True
    allow_births: bool = False
    birth_probability: float = 0.2
    allow_removals: bool = False
    removal_probability: float = 0.05
    external_impulse_probability: float = 0.0
    external_impulse_range: tuple[float, float] = (0.15, 0.6)
    scenario_mixture: tuple[str, ...] = ("baseline",)
    distribution: str = "in_distribution"

    @property
    def n_max(self) -> int:
        return self.max_objects if self.padding_max_objects is None else self.padding_max_objects

    @property
    def observation_dt(self) -> float:
        return 1.0 / self.frame_rate

    def validate(self) -> None:
        height, width = self.image_size
        if height <= 0 or width <= 0:
            raise ValueError("image_size must contain positive dimensions")
        if self.frame_rate <= 0 or self.physics_rate <= 0:
            raise ValueError("frame_rate and physics_rate must be positive")
        if self.sequence_frames < 2:
            raise ValueError("sequence_frames must be at least two")
        if self.min_objects < 1 or self.max_objects < self.min_objects:
            raise ValueError("object count range is invalid")
        if self.n_max < self.max_objects:
            raise ValueError("padding_max_objects cannot be below max_objects")
        if len(self.gravity) != 3:
            raise ValueError("gravity must have three components")
        bounds = torch.as_tensor(self.world_bounds)
        if bounds.shape != (3, 2) or bool(torch.any(bounds[:, 1] <= bounds[:, 0])):
            raise ValueError("world_bounds must be increasing [3, 2] bounds")
        for name in (
            "radius_range",
            "mass_range",
            "restitution_range",
            "drag_range",
            "friction_range",
            "initial_speed_range",
            "ensured_pair_height_range",
            "ensured_pair_surface_gap_range",
            "ensured_pair_speed_range",
            "ensured_pair_lateral_offset_range",
            "external_impulse_range",
        ):
            lower, upper = getattr(self, name)
            if lower < 0 or upper < lower:
                raise ValueError(f"{name} must be nonnegative and increasing")
        if self.radius_range[0] <= 0 or self.mass_range[0] <= 0:
            raise ValueError("radius and mass must be strictly positive")
        if not 0 <= self.restitution_range[0] <= self.restitution_range[1] <= 1:
            raise ValueError("restitution must lie in [0, 1]")
        for probability_name in (
            "birth_probability",
            "removal_probability",
            "external_impulse_probability",
        ):
            probability = getattr(self, probability_name)
            if not 0 <= probability <= 1:
                raise ValueError(f"{probability_name} must lie in [0, 1]")
        if not self.scenario_mixture:
            raise ValueError("scenario_mixture must contain at least one scenario")
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
        unknown_scenarios = set(self.scenario_mixture) - supported_scenarios
        if unknown_scenarios:
            raise ValueError(f"unsupported scenarios: {sorted(unknown_scenarios)}")
        if self.distribution not in {"in_distribution", "ood"}:
            raise ValueError("distribution must be 'in_distribution' or 'ood'")

    @classmethod
    def from_config(cls, config: Any) -> SphereWorldConfig:
        """Coerce a top-level project config or simulator subsection.

        This keeps the simulator independent of the repository's config loader
        while accepting its typed dataclasses and plain resolved YAML mappings.
        """

        if isinstance(config, cls):
            config.validate()
            return config
        root = _plain_mapping(config)
        simulator_raw = root.get("simulator", root)
        simulator = _plain_mapping(simulator_raw)
        model_raw = root.get("model", {})
        model = _plain_mapping(model_raw) if model_raw else {}
        camera_raw = simulator.get("camera", {})
        camera = _plain_mapping(camera_raw) if camera_raw else {}

        field_names = set(cls.__dataclass_fields__)
        values: dict[str, Any] = {key: simulator[key] for key in field_names if key in simulator}
        if "padding_max_objects" not in values and "max_objects" in model:
            values["padding_max_objects"] = int(model["max_objects"])
        if "camera_motion" not in values:
            if "motion" in camera:
                values["camera_motion"] = camera["motion"]
            elif simulator.get("moving_camera") is True:
                values["camera_motion"] = "mixed"
        aliases = {
            "camera_fov_degrees": "vertical_fov_degrees",
            "camera_translation_amplitude": "translation_amplitude",
            "camera_orbit_amplitude": "orbit_amplitude",
            "camera_zoom_amplitude": "zoom_amplitude",
        }
        for destination, source in aliases.items():
            if destination not in values and source in camera:
                values[destination] = camera[source]

        if "image_size" in values:
            values["image_size"] = tuple(int(item) for item in values["image_size"])
        if "gravity" in values:
            values["gravity"] = tuple(float(item) for item in values["gravity"])
        if "world_bounds" in values:
            values["world_bounds"] = tuple(
                tuple(float(item) for item in axis) for axis in values["world_bounds"]
            )
        for range_name in (
            "radius_range",
            "mass_range",
            "restitution_range",
            "drag_range",
            "friction_range",
            "initial_speed_range",
            "ensured_pair_height_range",
            "ensured_pair_surface_gap_range",
            "ensured_pair_speed_range",
            "ensured_pair_lateral_offset_range",
            "external_impulse_range",
        ):
            if range_name in values:
                values[range_name] = _tuple_range(values[range_name], name=range_name)
        if "scenario_mixture" in values:
            values["scenario_mixture"] = tuple(str(item) for item in values["scenario_mixture"])
        resolved = cls(**values)
        resolved.validate()
        return resolved

    def for_distribution(self, distribution: str) -> SphereWorldConfig:
        """Return an explicit in-distribution or compositional-OOD variant."""

        if distribution == "in_distribution":
            return replace(self, distribution=distribution)
        if distribution != "ood":
            raise ValueError("unknown simulator distribution")
        # Held-out joint ranges remain physically reasonable but do not overlap
        # the default train parameter support.
        return replace(
            self,
            distribution="ood",
            restitution_range=(0.18, 0.4),
            drag_range=(0.19, 0.32),
            camera_motion=("combined" if self.camera_motion == "fixed" else self.camera_motion),
        )

    def scenario_for_seed(self, seed: int) -> str:
        """Select a deterministic interaction regime for an episode seed."""

        return self.scenario_mixture[int(seed) % len(self.scenario_mixture)]

    def for_scenario(self, scenario: str) -> SphereWorldConfig:
        """Resolve named physical ranges without changing tensor contracts."""

        if scenario == "baseline":
            return self
        if scenario == "reference_pairs":
            return replace(
                self,
                radius_range=(0.21, 0.21),
                mass_range=(0.85, 1.15),
                restitution_range=(0.75, 0.9),
                drag_range=(0.005, 0.04),
                friction_range=(0.0, 0.04),
                initial_speed_range=(0.9, 1.25),
                ensured_pair_speed_range=(1.35, 1.6),
                external_impulse_probability=0.0,
            )
        if scenario == "elastic_pairs":
            return replace(
                self,
                restitution_range=(0.78, 0.95),
                drag_range=(0.005, 0.04),
                friction_range=(0.03, 0.12),
                initial_speed_range=(0.85, 1.55),
                external_impulse_probability=0.0,
            )
        if scenario == "damped_contacts":
            return replace(
                self,
                restitution_range=(0.18, 0.42),
                drag_range=(0.18, 0.32),
                friction_range=(0.28, 0.55),
                initial_speed_range=(0.35, 1.0),
                external_impulse_probability=0.0,
            )
        if scenario == "impulse_perturbation":
            return replace(
                self,
                restitution_range=(0.4, 0.8),
                drag_range=(0.02, 0.14),
                friction_range=(0.06, 0.3),
                initial_speed_range=(0.45, 1.3),
                external_impulse_probability=0.12,
                external_impulse_range=(0.25, 0.8),
            )
        if scenario == "camera_parallax":
            return replace(
                self,
                camera_motion="combined",
                camera_translation_amplitude=0.55,
                camera_orbit_amplitude=0.45,
                camera_zoom_amplitude=0.12,
                external_impulse_probability=0.0,
            )
        if scenario == "glancing_impacts":
            return replace(
                self,
                restitution_range=(0.55, 0.88),
                drag_range=(0.005, 0.08),
                friction_range=(0.01, 0.08),
                initial_speed_range=(1.1, 1.8),
                ensured_pair_lateral_offset_range=(0.18, 0.32),
                external_impulse_probability=0.0,
            )
        if scenario == "heavy_light_impacts":
            return replace(
                self,
                mass_range=(0.3, 2.5),
                restitution_range=(0.38, 0.78),
                drag_range=(0.01, 0.12),
                friction_range=(0.04, 0.22),
                initial_speed_range=(0.7, 1.5),
                external_impulse_probability=0.0,
            )
        raise ValueError(f"unsupported scenario {scenario!r}")


@dataclass(frozen=True)
class LifecycleEvents:
    created: Tensor
    removed: Tensor


def _uniform(
    generator: torch.Generator,
    shape: tuple[int, ...],
    value_range: tuple[float, float],
) -> Tensor:
    lower, upper = value_range
    return lower + (upper - lower) * torch.rand(shape, generator=generator)


class SphereWorld:
    """Stateful deterministic simulator with explicit frame-time stepping."""

    def __init__(self, config: SphereWorldConfig | Mapping[str, Any], seed: int) -> None:
        self.seed = int(seed)
        base_config = SphereWorldConfig.from_config(config)
        self.scenario_name = base_config.scenario_for_seed(self.seed)
        # Scenario ranges define the in-distribution physical regime. Apply the
        # held-out distribution last so scenario-specific replacements cannot
        # silently overwrite the OOD restitution/drag contract.
        self.config = base_config.for_scenario(self.scenario_name).for_distribution(
            base_config.distribution
        )
        self.config.validate()
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(self.seed & 0x7FFF_FFFF_FFFF_FFFF)
        # Rendering can consume a frame-sized number of random samples. Keep it
        # independent from the physical/lifecycle stream so changing only
        # observation noise cannot change future external impulses.
        self.render_generator = torch.Generator(device="cpu")
        self.render_generator.manual_seed((self.seed + 104_729) & 0x7FFF_FFFF_FFFF_FFFF)
        self.camera = CameraTrajectory(
            CameraTrajectoryConfig(
                image_size=self.config.image_size,
                mode=self.config.camera_motion,
                vertical_fov_degrees=self.config.camera_fov_degrees,
                translation_amplitude=self.config.camera_translation_amplitude,
                orbit_amplitude=self.config.camera_orbit_amplitude,
                zoom_amplitude=self.config.camera_zoom_amplitude,
            ),
            seed=self.seed + 8_191,
        )
        self.physics_config = PhysicsConfig(
            gravity=self.config.gravity,
            bounds=self.config.world_bounds,
            max_substep=1.0 / self.config.physics_rate,
            solver_iterations=self.config.solver_iterations,
        )
        (
            self.state,
            self._simulator_ids,
            self._spawn_frame,
            self._remove_frame,
            self._spawn_position,
            self._spawn_velocity,
        ) = self._sample_state()
        self.timestamp = 0.0
        self.frame_index = 0

    def _sample_state(
        self,
    ) -> tuple[SphereState, Tensor, Tensor, Tensor, Tensor, Tensor]:
        config = self.config
        count = int(
            torch.randint(
                config.min_objects,
                config.max_objects + 1,
                (),
                generator=self.generator,
            )
        )
        n_max = config.n_max
        radius = torch.zeros((n_max, 1), dtype=torch.float32)
        mass = torch.ones((n_max, 1), dtype=torch.float32)
        restitution = torch.zeros((n_max, 1), dtype=torch.float32)
        drag = torch.zeros((n_max, 1), dtype=torch.float32)
        friction = torch.zeros((n_max, 1), dtype=torch.float32)
        radius[:count, 0] = _uniform(self.generator, (count,), config.radius_range)
        mass[:count, 0] = _uniform(self.generator, (count,), config.mass_range)
        restitution[:count, 0] = _uniform(self.generator, (count,), config.restitution_range)
        drag[:count, 0] = _uniform(self.generator, (count,), config.drag_range)
        friction[:count, 0] = _uniform(self.generator, (count,), config.friction_range)

        albedo = torch.zeros((n_max, 3), dtype=torch.float32)
        raw_colour = 0.18 + 0.77 * torch.rand((count, 3), generator=self.generator)
        brightest = raw_colour.max(dim=-1).values
        raw_colour = raw_colour + (0.55 - brightest).clamp_min(0.0).unsqueeze(-1)
        albedo[:count] = raw_colour.clamp(0.0, 0.98)

        position = torch.zeros((n_max, 3), dtype=torch.float32)
        velocity = torch.zeros_like(position)
        bounds = torch.tensor(config.world_bounds, dtype=torch.float32)
        placed = 0
        if count >= 2 and config.ensure_collision:
            pair_height = float(_uniform(self.generator, (), config.ensured_pair_height_range))
            pair_z = float(_uniform(self.generator, (), (-0.25, 0.25)))
            lateral_offset = float(
                _uniform(
                    self.generator,
                    (),
                    config.ensured_pair_lateral_offset_range,
                )
            )
            surface_gap = float(
                _uniform(
                    self.generator,
                    (),
                    config.ensured_pair_surface_gap_range,
                )
            )
            center_distance = float(radius[0, 0] + radius[1, 0]) + surface_gap
            if lateral_offset >= center_distance:
                raise ValueError("ensured pair lateral offset must be below center separation")
            half_separation = 0.5 * math.sqrt(
                center_distance * center_distance - lateral_offset * lateral_offset
            )
            position[0] = torch.tensor(
                [-half_separation, pair_height, pair_z - 0.5 * lateral_offset]
            )
            position[1] = torch.tensor(
                [half_separation, pair_height, pair_z + 0.5 * lateral_offset]
            )
            speed = float(_uniform(self.generator, (), config.ensured_pair_speed_range))
            velocity[0] = torch.tensor([speed, 0.15, 0.0])
            velocity[1] = torch.tensor([-speed, 0.15, 0.0])
            placed = 2

        for slot in range(placed, count):
            candidate = None
            for _ in range(128):
                candidate = torch.stack(
                    (
                        _uniform(
                            self.generator,
                            (),
                            (
                                float(bounds[0, 0] + radius[slot, 0] + 0.1),
                                float(bounds[0, 1] - radius[slot, 0] - 0.1),
                            ),
                        ),
                        _uniform(
                            self.generator,
                            (),
                            (
                                float(bounds[1, 0] + radius[slot, 0] + 0.12),
                                min(
                                    2.5,
                                    float(bounds[1, 1] - radius[slot, 0] - 0.1),
                                ),
                            ),
                        ),
                        _uniform(
                            self.generator,
                            (),
                            (
                                float(bounds[2, 0] + radius[slot, 0] + 0.1),
                                float(bounds[2, 1] - radius[slot, 0] - 0.1),
                            ),
                        ),
                    )
                )
                if slot == 0:
                    break
                separation = torch.linalg.vector_norm(position[:slot] - candidate, dim=-1)
                required = radius[:slot, 0] + radius[slot, 0] + 0.08
                if bool(torch.all(separation > required)):
                    break
            if candidate is None:
                raise RuntimeError("failed to sample a sphere position")
            position[slot] = candidate
            direction = torch.randn(3, generator=self.generator)
            direction[1] *= 0.55
            direction = direction / torch.linalg.vector_norm(direction).clamp_min(1.0e-6)
            speed = _uniform(self.generator, (), config.initial_speed_range)
            velocity[slot] = direction * speed

        simulator_ids = torch.arange(n_max, dtype=torch.int64)
        object_id = torch.full((n_max,), -1, dtype=torch.int64)
        object_id[:count] = simulator_ids[:count]
        active = torch.zeros(n_max, dtype=torch.bool)
        active[:count] = True
        spawn_frame = torch.full((n_max,), -1, dtype=torch.int64)
        spawn_frame[:count] = 0
        remove_frame = torch.full((n_max,), -1, dtype=torch.int64)
        spawn_position = position.clone()
        spawn_velocity = velocity.clone()

        if (
            config.allow_births
            and count >= 3
            and float(torch.rand((), generator=self.generator)) < config.birth_probability
        ):
            slot = count - 1
            birth_low = max(2, config.sequence_frames // 4)
            birth_high = max(birth_low + 1, config.sequence_frames // 2 + 1)
            spawn_frame[slot] = int(
                torch.randint(birth_low, birth_high, (), generator=self.generator)
            )
            active[slot] = False
            object_id[slot] = -1
            side = -1.0 if int(torch.randint(0, 2, (), generator=self.generator)) == 0 else 1.0
            spawn_position[slot, 0] = (
                bounds[0, 0] + radius[slot, 0] + 0.02
                if side < 0
                else bounds[0, 1] - radius[slot, 0] - 0.02
            )
            spawn_position[slot, 1] = 1.0
            spawn_position[slot, 2] = 0.0
            spawn_velocity[slot] = torch.tensor([-side * 1.0, 0.25, 0.0])
            position[slot] = spawn_position[slot]
            velocity[slot].zero_()

        if (
            config.allow_removals
            and count >= 2
            and float(torch.rand((), generator=self.generator)) < config.removal_probability
        ):
            slot = count - 1
            earliest = max(int(spawn_frame[slot]) + 3, config.sequence_frames // 2)
            if earliest < config.sequence_frames - 1:
                remove_frame[slot] = int(
                    torch.randint(
                        earliest,
                        config.sequence_frames,
                        (),
                        generator=self.generator,
                    )
                )

        orientation = torch.zeros((n_max, 4), dtype=torch.float32)
        orientation[:, 3] = 1.0
        state = SphereState(
            object_id=object_id,
            active=active,
            position=position,
            velocity=velocity,
            radius=radius,
            mass=mass,
            restitution=restitution,
            drag=drag,
            friction=friction,
            albedo=albedo,
            orientation=orientation,
            angular_velocity=torch.zeros((n_max, 3), dtype=torch.float32),
            sleeping=torch.zeros(n_max, dtype=torch.bool),
            sleep_counter=torch.zeros(n_max, dtype=torch.int64),
        )
        state.validate()
        return (
            state,
            simulator_ids,
            spawn_frame,
            remove_frame,
            spawn_position,
            spawn_velocity,
        )

    def apply_lifecycle(self, frame_index: int | None = None) -> LifecycleEvents:
        """Apply scheduled births/removals at an observation frame."""

        if frame_index is None:
            frame_index = self.frame_index
        created = self._spawn_frame == frame_index
        removed = self._remove_frame == frame_index
        if frame_index == 0:
            created &= self.state.active
        state = self.state
        object_id = state.object_id.clone()
        active = state.active.clone()
        position = state.position.clone()
        velocity = state.velocity.clone()
        sleeping = state.sleeping.clone()
        sleep_counter = state.sleep_counter.clone()
        if bool(created.any()):
            active[created] = True
            object_id[created] = self._simulator_ids[created]
            position[created] = self._spawn_position[created]
            velocity[created] = self._spawn_velocity[created]
            sleeping[created] = False
            sleep_counter[created] = 0
        if bool(removed.any()):
            active[removed] = False
            object_id[removed] = -1
            velocity[removed] = 0
            sleeping[removed] = False
            sleep_counter[removed] = 0
        self.state = replace(
            state,
            object_id=object_id,
            active=active,
            position=position,
            velocity=velocity,
            sleeping=sleeping,
            sleep_counter=sleep_counter,
        )
        self.state.validate()
        return LifecycleEvents(created=created.clone(), removed=removed.clone())

    def sample_external_impulse(self) -> Tensor:
        """Sample a labelled rare impulse from the episode-local generator."""

        impulse = torch.zeros_like(self.state.velocity)
        if self.config.external_impulse_probability <= 0:
            return impulse
        if (
            float(torch.rand((), generator=self.generator))
            >= self.config.external_impulse_probability
        ):
            return impulse
        candidates = torch.nonzero(self.state.active, as_tuple=False).flatten()
        if candidates.numel() == 0:
            return impulse
        selected = int(
            candidates[int(torch.randint(0, candidates.numel(), (), generator=self.generator))]
        )
        direction = torch.randn(3, generator=self.generator)
        direction = direction / torch.linalg.vector_norm(direction).clamp_min(1.0e-6)
        magnitude = _uniform(self.generator, (), self.config.external_impulse_range)
        impulse[selected] = direction * magnitude
        return impulse

    def step(
        self,
        dt: float | None = None,
        *,
        external_impulse: Tensor | None = None,
    ) -> PhysicsStepEvents:
        """Advance physics and return exact interval event labels."""

        if dt is None:
            dt = self.config.observation_dt
        if external_impulse is None:
            external_impulse = self.sample_external_impulse()
        self.state, events = advance_spheres(
            self.state,
            float(dt),
            self.physics_config,
            external_impulse=external_impulse,
        )
        self.timestamp += float(dt)
        self.frame_index += 1
        return events

    def camera_frame(self, timestamp: float | None = None) -> CameraFrame:
        if timestamp is None:
            timestamp = self.timestamp
        return self.camera.at(float(timestamp), dtype=self.state.position.dtype)

    def render(
        self,
        *,
        camera: CameraFrame | None = None,
    ) -> RenderOutput:
        """Render the current world state and exact per-object image labels."""

        if camera is None:
            camera = self.camera_frame()
        return render_spheres(
            self.state,
            camera,
            self.config.image_size,
            edge_softness_pixels=self.config.edge_softness_pixels,
            noise_std=self.config.render_noise_std,
            generator=self.render_generator,
        )

    def reset(self) -> None:
        """Reset this instance to exactly the initial seeded world."""

        replacement = SphereWorld(self.config, self.seed)
        self.__dict__.update(replacement.__dict__)
