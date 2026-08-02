"""Composite hybrid dynamics model."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, fields, replace
from typing import Any

import torch
from torch import Tensor, nn

from world_model.belief import (
    BeliefTrajectory,
    MotionMode,
    ObjectBeliefTensor,
    WorldBelief,
)
from world_model.dynamics.analytic import AnalyticKinematics
from world_model.dynamics.contacts import ContactPlane, SphereContactResolver
from world_model.dynamics.events import EventModel
from world_model.dynamics.graph import InteractionGraph
from world_model.dynamics.modal import ModalDynamics
from world_model.dynamics.rollout import RolloutEngine, RolloutStep
from world_model.dynamics.uncertainty import UncertaintyDynamics


@dataclass(frozen=True)
class DynamicsConfig:
    """Self-contained dynamics dimensions and numerical settings."""

    modal_count: int = 2
    modal_dim: int = 2
    residual_dynamics_dim: int = 8
    global_code_dim: int = 8
    max_substep: float = 1.0 / 120.0
    graph_hidden_dim: int = 64
    uncertainty_hidden_dim: int = 32
    interaction_radius: float = 0.5
    constant_mode_count: int = 0
    max_modal_acceleration: float = 5.0
    max_pair_force: float = 2.0
    max_node_acceleration: float = 2.0
    base_process_variance_per_second: float = 1e-5
    process_noise_position: float | None = None
    process_noise_velocity: float | None = None
    log_variance_min: float = -20.0
    log_variance_max: float = 10.0
    ground_height: float = 0.0
    contact_margin: float = 0.0
    boundary_contact_tolerance: float = 1.0e-4
    penetration_slop: float = 1e-4
    max_penetration_correction: float = 0.05
    contact_confidence_sigma: float = 0.0
    pair_collision_speed_epsilon: float = 1.0e-7
    boundary_collision_speed_epsilon: float = 0.1
    solver_iterations: int = 2
    sleep_speed: float = 0.02
    world_bounds: tuple[tuple[float, float], ...] | None = None

    @property
    def fast_state_dim(self) -> int:
        return 13 + self.modal_count * 2 * self.modal_dim

    def validate(self) -> DynamicsConfig:
        if self.modal_count < 0 or self.modal_dim < 0:
            raise ValueError("modal dimensions must be nonnegative")
        if not 0 <= self.constant_mode_count <= self.modal_count:
            raise ValueError("constant_mode_count is outside modal bank")
        if self.residual_dynamics_dim < 0 or self.global_code_dim < 0:
            raise ValueError("dynamics code dimensions must be nonnegative")
        if self.max_substep <= 0 or not math.isfinite(self.max_substep):
            raise ValueError("max_substep must be finite and positive")
        if self.interaction_radius <= 0:
            raise ValueError("interaction_radius must be positive")
        for name, value in (
            ("base_process_variance_per_second", self.base_process_variance_per_second),
            ("process_noise_position", self.process_noise_position),
            ("process_noise_velocity", self.process_noise_velocity),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.log_variance_min >= self.log_variance_max:
            raise ValueError("invalid log variance bounds")
        for name, value in (
            ("contact_margin", self.contact_margin),
            ("boundary_contact_tolerance", self.boundary_contact_tolerance),
            ("contact_confidence_sigma", self.contact_confidence_sigma),
        ):
            if value < 0 or not math.isfinite(value):
                raise ValueError(f"{name} must be finite and nonnegative")
        for name, value in (
            ("pair_collision_speed_epsilon", self.pair_collision_speed_epsilon),
            (
                "boundary_collision_speed_epsilon",
                self.boundary_collision_speed_epsilon,
            ),
        ):
            if value < 0 or not math.isfinite(value):
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.solver_iterations < 1:
            raise ValueError("solver_iterations must be at least one")
        if self.world_bounds is not None and (
            len(self.world_bounds) != 3
            or any(len(bounds) != 2 or bounds[0] >= bounds[1] for bounds in self.world_bounds)
        ):
            raise ValueError("world_bounds must contain three increasing pairs")
        return self


class DynamicsModel(nn.Module):
    """Analytic + modal + learned interaction + event + uncertainty dynamics."""

    def __init__(
        self,
        config: DynamicsConfig | None = None,
        **overrides: object,
    ) -> None:
        super().__init__()
        if config is not None and overrides:
            config = replace(config, **overrides)
        elif config is None:
            config = DynamicsConfig(**overrides)
        self.config = config.validate()
        self.analytic = AnalyticKinematics()
        self.modal = ModalDynamics(
            self.config.modal_count,
            self.config.modal_dim,
            constant_mode_count=self.config.constant_mode_count,
            max_residual_acceleration=self.config.max_modal_acceleration,
        )
        self.interactions = InteractionGraph(
            self.config.residual_dynamics_dim,
            self.config.global_code_dim,
            hidden_dim=self.config.graph_hidden_dim,
            interaction_radius=self.config.interaction_radius,
            max_pair_force=self.config.max_pair_force,
            max_node_acceleration=self.config.max_node_acceleration,
        )
        resolver = SphereContactResolver(
            planes=self._environment_planes(),
            contact_margin=self.config.contact_margin,
            boundary_contact_tolerance=self.config.boundary_contact_tolerance,
            penetration_slop=self.config.penetration_slop,
            max_position_correction=self.config.max_penetration_correction,
            contact_confidence_sigma=self.config.contact_confidence_sigma,
            collision_speed_epsilon=self.config.pair_collision_speed_epsilon,
            boundary_collision_speed_epsilon=(self.config.boundary_collision_speed_epsilon),
            solver_iterations=self.config.solver_iterations,
        )
        self.events = EventModel(
            resolver,
            sleep_speed_threshold=self.config.sleep_speed,
        )
        self.uncertainty = UncertaintyDynamics(
            self.config.fast_state_dim,
            hidden_dim=self.config.uncertainty_hidden_dim,
            base_process_variance_per_second=(self.config.base_process_variance_per_second),
            position_process_variance_per_second=(self.config.process_noise_position),
            velocity_process_variance_per_second=(self.config.process_noise_velocity),
            log_variance_bounds=(
                self.config.log_variance_min,
                self.config.log_variance_max,
            ),
        )
        self.rollout_engine = RolloutEngine()

    def _environment_planes(self) -> tuple[ContactPlane, ...]:
        if self.config.world_bounds is None:
            return (
                ContactPlane(
                    normal=(0.0, 1.0, 0.0),
                    offset=self.config.ground_height,
                    name="ground",
                    is_ground=True,
                ),
            )
        planes: list[ContactPlane] = []
        axis_names = ("x", "y", "z")
        for axis, (lower, upper) in enumerate(self.config.world_bounds):
            lower_normal = [0.0, 0.0, 0.0]
            lower_normal[axis] = 1.0
            upper_normal = [0.0, 0.0, 0.0]
            upper_normal[axis] = -1.0
            planes.append(
                ContactPlane(
                    normal=tuple(lower_normal),
                    offset=float(lower),
                    name=f"{axis_names[axis]}_minimum",
                    is_ground=axis == 1,
                )
            )
            planes.append(
                ContactPlane(
                    normal=tuple(upper_normal),
                    offset=float(-upper),
                    name=f"{axis_names[axis]}_maximum",
                    is_ground=False,
                )
            )
        return tuple(planes)

    @classmethod
    def from_config(cls, config: Any) -> DynamicsModel:
        """Build from root ``OrpheusConfig`` without depending on its type."""

        model = getattr(config, "model", config)
        state = getattr(model, "state", None)
        dynamics = getattr(model, "dynamics", None)
        if state is None or dynamics is None:
            raise TypeError("config must expose model.state and model.dynamics")
        simulator = getattr(config, "simulator", None)
        ground_height = 0.0
        if simulator is not None:
            ground_height = float(simulator.world_bounds[1][0])
        return cls(
            modal_count=int(state.modal_count),
            modal_dim=int(state.modal_dim),
            residual_dynamics_dim=int(state.residual_dynamics_dim),
            global_code_dim=int(state.global_dim),
            max_substep=float(dynamics.max_substep),
            graph_hidden_dim=int(dynamics.hidden_dim),
            uncertainty_hidden_dim=max(16, int(dynamics.hidden_dim) // 2),
            interaction_radius=float(dynamics.interaction_radius),
            max_modal_acceleration=float(dynamics.modal_acceleration_scale),
            max_pair_force=float(dynamics.residual_acceleration_scale),
            max_node_acceleration=float(dynamics.residual_acceleration_scale),
            process_noise_position=float(dynamics.process_noise_position),
            process_noise_velocity=float(dynamics.process_noise_velocity),
            log_variance_min=float(state.fast_log_variance_min),
            log_variance_max=float(state.fast_log_variance_max),
            ground_height=ground_height,
            contact_margin=float(dynamics.contact_margin),
            boundary_contact_tolerance=float(dynamics.boundary_contact_tolerance),
            penetration_slop=float(dynamics.penetration_slop),
            max_penetration_correction=float(dynamics.max_penetration_correction),
            contact_confidence_sigma=float(dynamics.contact_confidence_sigma),
            pair_collision_speed_epsilon=(float(dynamics.pair_collision_speed_epsilon)),
            boundary_collision_speed_epsilon=(float(dynamics.boundary_collision_speed_epsilon)),
            solver_iterations=(
                int(getattr(simulator, "solver_iterations", 2)) if simulator is not None else 2
            ),
            sleep_speed=float(dynamics.sleep_speed),
            world_bounds=(
                tuple((float(bounds[0]), float(bounds[1])) for bounds in simulator.world_bounds)
                if simulator is not None
                else None
            ),
        )

    @classmethod
    def from_belief(
        cls,
        belief: WorldBelief,
        **settings: object,
    ) -> DynamicsModel:
        """Construct dimensionally compatible dynamics from a belief."""

        return cls(
            modal_count=belief.objects.modal_count,
            modal_dim=belief.objects.modal_dim,
            residual_dynamics_dim=belief.objects.residual_dynamics_dim,
            global_code_dim=belief.global_code.shape[-1],
            **settings,
        )

    def _validate_dimensions(self, belief: WorldBelief) -> None:
        objects = belief.objects
        expected = (
            self.config.modal_count,
            self.config.modal_dim,
            self.config.residual_dynamics_dim,
            self.config.global_code_dim,
        )
        actual = (
            objects.modal_count,
            objects.modal_dim,
            objects.residual_dynamics_dim,
            belief.global_code.shape[-1],
        )
        if actual != expected:
            raise ValueError(
                f"belief/dynamics dimensions differ: expected {expected}, got {actual}"
            )

    def _normalise_dt(self, belief: WorldBelief, dt: float | Tensor) -> Tensor:
        value = torch.as_tensor(dt, device=belief.device, dtype=belief.dtype)
        if value.ndim == 0:
            value = value.expand(belief.batch_size).clone()
        if value.shape != belief.timestamp.shape:
            raise ValueError("dt must be scalar or shape [B]")
        if not torch.isfinite(value).all() or torch.any(value < 0):
            raise ValueError("dt must contain finite nonnegative seconds")
        return value

    def _substep(
        self,
        belief: WorldBelief,
        dt: Tensor,
        *,
        external_acceleration: Tensor | None = None,
    ) -> RolloutStep:
        objects, modal = self.modal(belief.objects, dt)
        interaction = self.interactions(
            objects,
            belief.global_code,
            modal_acceleration=modal.residual_acceleration,
        )
        total_residual = modal.residual_acceleration + interaction.residual_acceleration
        objects = self.analytic(
            objects,
            belief.gravity,
            dt,
            residual_acceleration=total_residual,
            external_acceleration=external_acceleration,
        )
        events = self.events(objects, interaction)
        # Preserve the fixed uncertainty-network input shape while making the
        # graph's zero-centred edge-noise residual operational. Pair count
        # describes how many interactions are possible; the additional term
        # learns whether those interactions need more or less process noise.
        interaction_uncertainty = (
            interaction.interaction_density + interaction.edge_process_noise.sum(dim=-1)
        ).clamp_min(0.0)
        uncertainty = self.uncertainty(
            events.objects,
            dt,
            event_logits=events.event_logits,
            interaction_density=interaction_uncertainty,
            residual_acceleration=total_residual,
        )
        update_batch = dt > 0
        updated_objects = self._blend_objects(
            belief.objects,
            uncertainty.objects,
            update_batch,
        )
        updated = replace(
            belief,
            timestamp=belief.timestamp + dt,
            objects=updated_objects,
        )
        auxiliary = self._mask_auxiliary(
            {
                "pair_contact": events.contacts.pair_contact,
                "interval_pair_contact": events.contacts.interval_pair_contact,
                "pair_collision": events.contacts.pair_collision,
                "boundary_contact": events.contacts.boundary_contact,
                "interval_boundary_contact": events.contacts.interval_boundary_contact,
                "boundary_collision": events.contacts.boundary_collision,
                "ground_contact": events.contacts.ground_contact,
                "interval_ground_contact": events.contacts.interval_ground_contact,
                "ground_collision": events.contacts.ground_collision,
                "pair_impulse": events.contacts.pair_impulse,
                "max_penetration": events.contacts.max_penetration,
                "mean_penetration": events.contacts.mean_penetration,
                "action_reaction_residual": (events.contacts.action_reaction_residual),
                "process_variance": uncertainty.process_variance,
                "edge_process_noise": interaction.edge_process_noise,
                "residual_acceleration": total_residual,
            },
            update_batch,
        )
        event_logits = torch.where(
            update_batch[:, None, None],
            events.event_logits,
            belief.objects.motion_mode_logits,
        )
        return RolloutStep(
            belief=updated,
            event_logits=event_logits,
            auxiliary=auxiliary,
        )

    @staticmethod
    def _blend_objects(
        previous: ObjectBeliefTensor,
        updated: ObjectBeliefTensor,
        update_batch: Tensor,
    ) -> ObjectBeliefTensor:
        values: dict[str, Tensor] = {}
        for item in fields(previous):
            old_value = getattr(previous, item.name)
            new_value = getattr(updated, item.name)
            mask = update_batch
            while mask.ndim < old_value.ndim:
                mask = mask.unsqueeze(-1)
            values[item.name] = torch.where(mask, new_value, old_value)
        return replace(updated, **values)

    @staticmethod
    def _mask_auxiliary(
        values: dict[str, Tensor],
        update_batch: Tensor,
    ) -> dict[str, Tensor]:
        output: dict[str, Tensor] = {}
        for name, value in values.items():
            mask = update_batch
            while mask.ndim < value.ndim:
                mask = mask.unsqueeze(-1)
            output[name] = torch.where(mask, value, torch.zeros_like(value))
        return output

    def _zero_step(self, belief: WorldBelief) -> RolloutStep:
        objects = belief.objects
        batch, count = objects.active.shape
        event_logits = objects.motion_mode_logits.clone()
        # A zero-duration segment contains no event, even if the source belief
        # is instantaneously in COLLISION mode.
        event_logits[..., MotionMode.COLLISION] = -4.0
        auxiliary = {
            "pair_contact": torch.zeros(
                batch,
                count,
                count,
                device=belief.device,
                dtype=torch.bool,
            ),
            "interval_pair_contact": torch.zeros(
                batch,
                count,
                count,
                device=belief.device,
                dtype=torch.bool,
            ),
            "pair_collision": torch.zeros(
                batch,
                count,
                count,
                device=belief.device,
                dtype=torch.bool,
            ),
            "boundary_contact": torch.zeros(
                batch,
                count,
                len(self.events.resolver.plane_names),
                device=belief.device,
                dtype=torch.bool,
            ),
            "interval_boundary_contact": torch.zeros(
                batch,
                count,
                len(self.events.resolver.plane_names),
                device=belief.device,
                dtype=torch.bool,
            ),
            "boundary_collision": torch.zeros(
                batch,
                count,
                len(self.events.resolver.plane_names),
                device=belief.device,
                dtype=torch.bool,
            ),
            "ground_contact": torch.zeros_like(objects.active),
            "interval_ground_contact": torch.zeros_like(objects.active),
            "ground_collision": torch.zeros_like(objects.active),
            "pair_impulse": objects.position.new_zeros(batch, count, count),
            "max_penetration": belief.timestamp.new_zeros(batch),
            "mean_penetration": belief.timestamp.new_zeros(batch),
            "action_reaction_residual": belief.timestamp.new_zeros(batch),
            "process_variance": objects.fast_log_variance.new_zeros(
                batch,
                count,
                objects.fast_state_dim,
            ),
            "edge_process_noise": objects.position.new_zeros(batch, count, count),
            "residual_acceleration": objects.position.new_zeros(batch, count, 3),
        }
        return RolloutStep(
            belief=belief.clone(),
            event_logits=event_logits,
            auxiliary=auxiliary,
        )

    def _predict_step(
        self,
        belief: WorldBelief,
        dt: float | Tensor,
        *,
        external_acceleration: Tensor | None = None,
    ) -> RolloutStep:
        self._validate_dimensions(belief)
        elapsed = self._normalise_dt(belief, dt)
        output = belief.clone()
        if float(elapsed.max().detach().cpu()) == 0.0:
            return self._zero_step(output)
        substeps = max(
            1,
            math.ceil(float(elapsed.max().detach().cpu()) / self.config.max_substep),
        )
        sub_dt = elapsed / substeps
        result: RolloutStep | None = None
        interval_collision_logits: Tensor | None = None
        interval_pair_contact: Tensor | None = None
        interval_pair_collision: Tensor | None = None
        interval_boundary_contact: Tensor | None = None
        interval_boundary_collision: Tensor | None = None
        interval_ground_contact: Tensor | None = None
        interval_ground_collision: Tensor | None = None
        interval_pair_impulse: Tensor | None = None
        interval_max_penetration: Tensor | None = None
        for _ in range(substeps):
            result = self._substep(
                output,
                sub_dt,
                external_acceleration=external_acceleration,
            )
            output = result.belief
            event_valid = (sub_dt > 0).unsqueeze(-1) & output.objects.active
            collision_logits = torch.where(
                event_valid,
                result.event_logits[..., MotionMode.COLLISION],
                result.event_logits.new_full((), -4.0),
            )
            if interval_collision_logits is None:
                interval_collision_logits = collision_logits
                interval_pair_contact = result.auxiliary["interval_pair_contact"]
                interval_pair_collision = result.auxiliary["pair_collision"]
                interval_boundary_contact = result.auxiliary["interval_boundary_contact"]
                interval_boundary_collision = result.auxiliary["boundary_collision"]
                interval_ground_contact = result.auxiliary["interval_ground_contact"]
                interval_ground_collision = result.auxiliary["ground_collision"]
                interval_pair_impulse = result.auxiliary["pair_impulse"]
                interval_max_penetration = result.auxiliary["max_penetration"]
            else:
                interval_collision_logits = torch.maximum(
                    interval_collision_logits,
                    collision_logits,
                )
                interval_pair_contact = (
                    interval_pair_contact | result.auxiliary["interval_pair_contact"]
                )
                interval_pair_collision = (
                    interval_pair_collision | result.auxiliary["pair_collision"]
                )
                interval_boundary_contact = (
                    interval_boundary_contact | result.auxiliary["interval_boundary_contact"]
                )
                interval_boundary_collision = (
                    interval_boundary_collision | result.auxiliary["boundary_collision"]
                )
                interval_ground_contact = (
                    interval_ground_contact | result.auxiliary["interval_ground_contact"]
                )
                interval_ground_collision = (
                    interval_ground_collision | result.auxiliary["ground_collision"]
                )
                interval_pair_impulse = torch.maximum(
                    interval_pair_impulse,
                    result.auxiliary["pair_impulse"],
                )
                interval_max_penetration = torch.maximum(
                    interval_max_penetration,
                    result.auxiliary["max_penetration"],
                )
        assert result is not None
        assert interval_collision_logits is not None
        assert interval_pair_contact is not None
        assert interval_pair_collision is not None
        assert interval_boundary_contact is not None
        assert interval_boundary_collision is not None
        assert interval_ground_contact is not None
        assert interval_ground_collision is not None
        assert interval_pair_impulse is not None
        assert interval_max_penetration is not None
        # Avoid accumulated timestamp roundoff from many substeps.
        final_belief = replace(output, timestamp=belief.timestamp + elapsed)
        # Motion modes on the belief describe the endpoint. Rollout event
        # logits instead describe whether a collision occurred anywhere in
        # this prediction segment.
        event_logits = result.event_logits.clone()
        event_logits[..., MotionMode.COLLISION] = interval_collision_logits
        auxiliary = dict(result.auxiliary)
        auxiliary.update(
            {
                "interval_pair_contact": interval_pair_contact,
                "pair_collision": interval_pair_collision,
                "interval_boundary_contact": interval_boundary_contact,
                "boundary_collision": interval_boundary_collision,
                "interval_ground_contact": interval_ground_contact,
                "ground_collision": interval_ground_collision,
                "pair_impulse": interval_pair_impulse,
                "max_penetration": interval_max_penetration,
            }
        )
        return RolloutStep(
            belief=final_belief,
            event_logits=event_logits,
            auxiliary=auxiliary,
        )

    def predict(
        self,
        belief: WorldBelief,
        dt: float | Tensor,
    ) -> WorldBelief:
        """Predict a new belief after elapsed seconds without mutating input."""

        return self.predict_step(belief, dt).belief

    def predict_step(
        self,
        belief: WorldBelief,
        dt: float | Tensor,
    ) -> RolloutStep:
        """Return the endpoint belief plus events over the elapsed interval."""

        return self._predict_step(belief, dt)

    def rollout(
        self,
        belief: WorldBelief,
        query_times: Tensor | Sequence[float],
        *,
        return_events: bool = True,
    ) -> BeliefTrajectory:
        """Predict at sorted future offsets in seconds without mutating input."""

        return self.rollout_engine.rollout(
            lambda current, dt: self._predict_step(current, dt),
            belief,
            query_times,
            return_events=return_events,
        )
