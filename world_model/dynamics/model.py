"""Composite hybrid dynamics model."""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence
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
from world_model.dynamics.applicability import (
    PairApplicability,
    PairApplicabilityConfig,
    apply_pair_applicability,
)
from world_model.dynamics.attention import TypedAttentionInteractionResidual
from world_model.dynamics.contacts import ContactPlane, SphereContactResolver
from world_model.dynamics.events import EventModel
from world_model.dynamics.graph import InteractionGraph, InteractionOutput
from world_model.dynamics.modal import ModalDynamics
from world_model.dynamics.rollout import RolloutEngine, RolloutStep
from world_model.dynamics.uncertainty import UncertaintyDynamics


def _stable_substep_count(elapsed: Tensor, max_substep: float) -> int:
    """Return a ceiling count without inventing ticks from float clock noise.

    Observation timestamps are stored in the belief dtype.  Subtracting two
    float32 frame timestamps can put an intended integral ratio such as
    ``0.05 / (1 / 120) == 6`` a few ulps above six.  A literal ``ceil`` then
    alternates between six and seven dynamics ticks even though the simulator
    advances the same 20 Hz interval with six ticks.

    Snap only ratios indistinguishable from an integer at the elapsed tensor's
    precision.  The absolute cap keeps reduced-precision dtypes conservative;
    every genuinely non-integral interval still uses the specified ceiling.
    """

    maximum_elapsed = float(elapsed.max().detach().cpu())
    if maximum_elapsed <= 0.0:
        return 0
    ratio = maximum_elapsed / max_substep
    nearest = round(ratio)
    precision = torch.finfo(elapsed.dtype).eps
    integer_tolerance = min(
        1.0e-4,
        16.0 * precision * max(1.0, abs(ratio)),
    )
    if nearest >= 1 and abs(ratio - nearest) <= integer_tolerance:
        return nearest
    return max(1, math.ceil(ratio))


@dataclass(frozen=True)
class DynamicsConfig:
    """Self-contained dynamics dimensions and numerical settings."""

    modal_count: int = 2
    modal_dim: int = 2
    residual_dynamics_dim: int = 8
    global_code_dim: int = 8
    geometry_dim: int = 8
    appearance_dim: int = 32
    parameter_memory_dim: int = 48
    max_substep: float = 1.0 / 120.0
    # ``None`` preserves the historical behavior: evaluate the learned
    # interaction stack on every analytic microstep.  A finite value holds one
    # graph/attention proposal for at most this many seconds while modal state,
    # analytic kinematics, contacts/events, and uncertainty continue to advance
    # on the stable ``max_substep`` grid.
    learned_effect_interval_seconds: float | None = None
    # Disabled is the exact historical behavior.  When enabled, only learned
    # pair/event residuals receive a smooth causal geometry/motion/uncertainty
    # envelope; analytic kinematics and contact jumps remain unmodified.
    pair_applicability_enabled: bool = False
    pair_applicability_lookahead_seconds: float = 0.05
    pair_applicability_margin_m: float = 0.05
    pair_applicability_gap_temperature_m: float = 0.025
    pair_applicability_velocity_temperature_mps: float = 0.10
    graph_hidden_dim: int = 64
    uncertainty_hidden_dim: int = 32
    interaction_radius: float = 0.5
    constant_mode_count: int = 0
    max_modal_acceleration: float = 5.0
    max_pair_force: float = 2.0
    max_node_acceleration: float = 2.0
    attention_residual_enabled: bool = False
    attention_relation_endpoint_binding_enabled: bool = False
    attention_width: int = 128
    attention_heads: int = 4
    attention_layers: int = 4
    attention_feed_forward_width: int = 512
    attention_dropout: float = 0.0
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
    # Historical event logits were hard +/- constants with learned pair
    # residuals added afterward.  The opt-in hazard path keeps hard analytic
    # resolution for jumps while exposing continuous, calibratable logits.
    smooth_event_hazard_enabled: bool = False
    event_hazard_gap_temperature_m: float = 0.02
    event_hazard_velocity_temperature_mps: float = 0.10
    event_hazard_resolved_logit_floor: float = 2.0
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
        if self.geometry_dim <= 0 or self.appearance_dim < 0 or self.parameter_memory_dim < 0:
            raise ValueError("typed attention state dimensions are invalid")
        if self.attention_width <= 0 or self.attention_heads <= 0 or self.attention_layers <= 0:
            raise ValueError("attention width, heads, and layers must be positive")
        if self.attention_width % self.attention_heads != 0:
            raise ValueError("attention width must be divisible by attention heads")
        if self.attention_feed_forward_width <= 0:
            raise ValueError("attention feed-forward width must be positive")
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError("attention dropout must lie in [0,1)")
        if not isinstance(self.attention_relation_endpoint_binding_enabled, bool):
            raise ValueError("attention relation endpoint binding flag must be boolean")
        if self.max_substep <= 0 or not math.isfinite(self.max_substep):
            raise ValueError("max_substep must be finite and positive")
        if self.learned_effect_interval_seconds is not None and (
            isinstance(self.learned_effect_interval_seconds, bool)
            or not math.isfinite(self.learned_effect_interval_seconds)
            or self.learned_effect_interval_seconds < self.max_substep
        ):
            raise ValueError(
                "learned_effect_interval_seconds must be finite and no smaller than max_substep"
            )
        if not isinstance(self.pair_applicability_enabled, bool):
            raise ValueError("pair_applicability_enabled must be boolean")
        for name, value in (
            (
                "pair_applicability_lookahead_seconds",
                self.pair_applicability_lookahead_seconds,
            ),
            ("pair_applicability_margin_m", self.pair_applicability_margin_m),
        ):
            if isinstance(value, bool) or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name, value in (
            (
                "pair_applicability_gap_temperature_m",
                self.pair_applicability_gap_temperature_m,
            ),
            (
                "pair_applicability_velocity_temperature_mps",
                self.pair_applicability_velocity_temperature_mps,
            ),
        ):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
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
        if not isinstance(self.smooth_event_hazard_enabled, bool):
            raise ValueError("smooth_event_hazard_enabled must be boolean")
        for name, value in (
            ("event_hazard_gap_temperature_m", self.event_hazard_gap_temperature_m),
            (
                "event_hazard_velocity_temperature_mps",
                self.event_hazard_velocity_temperature_mps,
            ),
            (
                "event_hazard_resolved_logit_floor",
                self.event_hazard_resolved_logit_floor,
            ),
        ):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
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
        self.pair_applicability_config = PairApplicabilityConfig(
            enabled=self.config.pair_applicability_enabled,
            lookahead_seconds=self.config.pair_applicability_lookahead_seconds,
            margin_m=self.config.pair_applicability_margin_m,
            gap_temperature_m=self.config.pair_applicability_gap_temperature_m,
            velocity_temperature_mps=(self.config.pair_applicability_velocity_temperature_mps),
            collision_speed_epsilon=self.config.pair_collision_speed_epsilon,
        )
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
        self.attention_interactions = (
            TypedAttentionInteractionResidual(
                modal_count=self.config.modal_count,
                modal_dim=self.config.modal_dim,
                geometry_dim=self.config.geometry_dim,
                appearance_dim=self.config.appearance_dim,
                residual_dynamics_dim=self.config.residual_dynamics_dim,
                parameter_memory_dim=self.config.parameter_memory_dim,
                motion_mode_dim=len(MotionMode),
                global_code_dim=self.config.global_code_dim,
                relation_endpoint_binding_enabled=(
                    self.config.attention_relation_endpoint_binding_enabled
                ),
                width=self.config.attention_width,
                heads=self.config.attention_heads,
                layers=self.config.attention_layers,
                feed_forward_width=self.config.attention_feed_forward_width,
                dropout=self.config.attention_dropout,
                max_pair_force=self.config.max_pair_force,
                max_node_acceleration=self.config.max_node_acceleration,
            )
            if self.config.attention_residual_enabled
            else None
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
            smooth_hazard_enabled=self.config.smooth_event_hazard_enabled,
            contact_logit_scale=self.config.event_hazard_gap_temperature_m,
            collision_velocity_logit_scale=(self.config.event_hazard_velocity_temperature_mps),
            resolved_event_logit_floor=(self.config.event_hazard_resolved_logit_floor),
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
            geometry_dim=int(state.geometry_dim),
            appearance_dim=int(state.appearance_dim),
            parameter_memory_dim=int(state.parameter_memory_dim),
            max_substep=float(dynamics.max_substep),
            learned_effect_interval_seconds=(
                None
                if dynamics.learned_effect_interval_seconds is None
                else float(dynamics.learned_effect_interval_seconds)
            ),
            pair_applicability_enabled=bool(dynamics.pair_applicability_enabled),
            pair_applicability_lookahead_seconds=(
                float(dynamics.pair_applicability_lookahead_seconds)
            ),
            pair_applicability_margin_m=float(dynamics.pair_applicability_margin_m),
            pair_applicability_gap_temperature_m=(
                float(dynamics.pair_applicability_gap_temperature_m)
            ),
            pair_applicability_velocity_temperature_mps=(
                float(dynamics.pair_applicability_velocity_temperature_mps)
            ),
            graph_hidden_dim=int(dynamics.hidden_dim),
            uncertainty_hidden_dim=max(16, int(dynamics.hidden_dim) // 2),
            interaction_radius=float(dynamics.interaction_radius),
            max_modal_acceleration=float(dynamics.modal_acceleration_scale),
            max_pair_force=float(dynamics.residual_acceleration_scale),
            max_node_acceleration=float(dynamics.residual_acceleration_scale),
            attention_residual_enabled=bool(dynamics.attention_residual_enabled),
            attention_relation_endpoint_binding_enabled=bool(
                dynamics.attention_relation_endpoint_binding_enabled
            ),
            attention_width=int(dynamics.attention_width),
            attention_heads=int(dynamics.attention_heads),
            attention_layers=int(dynamics.attention_layers),
            attention_feed_forward_width=int(dynamics.attention_feed_forward_width),
            attention_dropout=float(dynamics.attention_dropout),
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
            smooth_event_hazard_enabled=bool(dynamics.smooth_event_hazard_enabled),
            event_hazard_gap_temperature_m=float(dynamics.event_hazard_gap_temperature_m),
            event_hazard_velocity_temperature_mps=float(
                dynamics.event_hazard_velocity_temperature_mps
            ),
            event_hazard_resolved_logit_floor=float(dynamics.event_hazard_resolved_logit_floor),
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
            geometry_dim=belief.objects.geometry_dim,
            appearance_dim=belief.objects.appearance_dim,
            parameter_memory_dim=belief.objects.parameter_memory.shape[-1],
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

    def _evaluate_interaction(
        self,
        belief: WorldBelief,
        modal_acceleration: Tensor,
    ) -> InteractionOutput:
        """Evaluate one differentiable learned interaction proposal.

        The returned tensors are deliberately not detached.  An opt-in
        multi-rate prediction may consume the same proposal on several
        analytic microsteps, allowing endpoint losses to accumulate gradient
        into the graph/attention invocation that produced it.
        """

        interaction = self.interactions(
            belief.objects,
            belief.global_code,
            modal_acceleration=modal_acceleration,
        )
        if self.attention_interactions is not None:
            interaction = self.attention_interactions(
                belief.objects,
                belief,
                interaction,
            )
        return interaction

    def _apply_pair_applicability(
        self,
        objects: ObjectBeliefTensor,
        interaction: InteractionOutput,
    ) -> tuple[InteractionOutput, PairApplicability]:
        """Apply current causal support to a possibly held learned proposal."""

        return apply_pair_applicability(
            objects,
            interaction,
            self.pair_applicability_config,
        )

    def _learned_effect_stride(self) -> int:
        """Return the bounded number of analytic ticks per learned proposal."""

        interval = self.config.learned_effect_interval_seconds
        if interval is None:
            return 1
        # Validation guarantees ``interval >= max_substep``.  Flooring keeps
        # the actual hold duration bounded because every analytic microstep is
        # no longer than ``max_substep``.  The small tolerance only avoids an
        # off-by-one from decimal serialization of an integral ratio.
        ratio = interval / self.config.max_substep
        return max(1, int(math.floor(ratio + 1.0e-12)))

    def _substep(
        self,
        belief: WorldBelief,
        dt: Tensor,
        *,
        modal_acceleration: Tensor,
        interaction: InteractionOutput,
        applicability: PairApplicability,
        external_acceleration: Tensor | None = None,
    ) -> RolloutStep:
        objects = belief.objects
        total_residual = modal_acceleration + interaction.residual_acceleration
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
                "pair_event_logits": events.pair_event_logits,
                "boundary_event_logits": events.boundary_event_logits,
                "max_penetration": events.contacts.max_penetration,
                "mean_penetration": events.contacts.mean_penetration,
                "action_reaction_residual": (events.contacts.action_reaction_residual),
                "process_variance": uncertainty.process_variance,
                "edge_process_noise": interaction.edge_process_noise,
                "residual_acceleration": total_residual,
                "pair_applicability": applicability.pair,
                "collision_applicability": applicability.collision,
            },
            update_batch,
        )
        for name in ("pair_event_logits", "boundary_event_logits"):
            value = auxiliary[name]
            mask = update_batch
            while mask.ndim < value.ndim:
                mask = mask.unsqueeze(-1)
            auxiliary[name] = torch.where(
                mask,
                value,
                value.new_full((), -4.0),
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
            "pair_event_logits": objects.position.new_full(
                (batch, count, count, 2),
                -4.0,
            ),
            "boundary_event_logits": objects.position.new_full(
                (batch, count, len(self.events.resolver.plane_names), 2),
                -4.0,
            ),
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
            "pair_applicability": objects.position.new_zeros(batch, count, count),
            "collision_applicability": objects.position.new_zeros(batch, count, count),
            "learned_effect_evaluation_count": torch.zeros(
                batch,
                device=belief.device,
                dtype=torch.int64,
            ),
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
        substeps = _stable_substep_count(elapsed, self.config.max_substep)
        if substeps == 0:
            return self._zero_step(output)
        sub_dt = elapsed / substeps
        result: RolloutStep | None = None
        held_interaction: InteractionOutput | None = None
        learned_effect_stride = self._learned_effect_stride()
        microsteps_since_effect_evaluation = learned_effect_stride
        learned_effect_evaluations = 0
        recompute_after_collision: Tensor | None = None
        interval_collision_logits: Tensor | None = None
        interval_pair_contact: Tensor | None = None
        interval_pair_collision: Tensor | None = None
        interval_boundary_contact: Tensor | None = None
        interval_boundary_collision: Tensor | None = None
        interval_ground_contact: Tensor | None = None
        interval_ground_collision: Tensor | None = None
        interval_pair_impulse: Tensor | None = None
        interval_pair_event_logits: Tensor | None = None
        interval_boundary_event_logits: Tensor | None = None
        interval_max_penetration: Tensor | None = None
        for _ in range(substeps):
            # Stable modal state and its inexpensive readout remain current on
            # every physical tick.  Only the expensive graph/attention proposal
            # is held.  Keeping this transition outside the held proposal also
            # preserves the exact modal composition contract.
            modal_objects, modal = self.modal(output.objects, sub_dt)
            output = replace(output, objects=modal_objects)
            evaluate_interaction = (
                held_interaction is None
                or microsteps_since_effect_evaluation >= learned_effect_stride
            )
            if not evaluate_interaction:
                current_edge_mask = self.interactions.candidate_edge_mask(output.objects)
                edge_set_changed = torch.any(current_edge_mask != held_interaction.edge_mask)
                invalidated = edge_set_changed
                if recompute_after_collision is not None:
                    invalidated = torch.logical_or(
                        invalidated,
                        recompute_after_collision,
                    )
                # A changed edge set invalidates the complete force/logit/noise
                # tuple, as does the previous tick's discrete velocity jump.
                # Partially remasking a stale vector proposal would be
                # incoherent, so pay one fresh learned evaluation before this
                # physical tick. One combined host decision avoids a second
                # accelerator synchronization for collision invalidation.
                evaluate_interaction = bool(invalidated.detach().cpu().item())
            if evaluate_interaction:
                held_interaction = self._evaluate_interaction(
                    output,
                    modal.residual_acceleration,
                )
                learned_effect_evaluations += 1
                microsteps_since_effect_evaluation = 0
            assert held_interaction is not None
            # Applicability depends on current geometry, relative motion, and
            # uncertainty, so it is refreshed every physical tick even when the
            # expensive raw learned proposal is held by multi-rate execution.
            step_interaction, applicability = self._apply_pair_applicability(
                output.objects,
                held_interaction,
            )
            result = self._substep(
                output,
                sub_dt,
                modal_acceleration=modal.residual_acceleration,
                interaction=step_interaction,
                applicability=applicability,
                external_acceleration=external_acceleration,
            )
            output = result.belief
            microsteps_since_effect_evaluation += 1
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
                interval_pair_event_logits = result.auxiliary["pair_event_logits"]
                interval_boundary_event_logits = result.auxiliary["boundary_event_logits"]
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
                current_pair_event_logits = result.auxiliary["pair_event_logits"]
                interval_pair_event_logits = torch.stack(
                    (
                        current_pair_event_logits[..., 0],
                        torch.maximum(
                            interval_pair_event_logits[..., 1],
                            current_pair_event_logits[..., 1],
                        ),
                    ),
                    dim=-1,
                )
                current_boundary_event_logits = result.auxiliary["boundary_event_logits"]
                interval_boundary_event_logits = torch.stack(
                    (
                        current_boundary_event_logits[..., 0],
                        torch.maximum(
                            interval_boundary_event_logits[..., 1],
                            current_boundary_event_logits[..., 1],
                        ),
                    ),
                    dim=-1,
                )
                interval_max_penetration = torch.maximum(
                    interval_max_penetration,
                    result.auxiliary["max_penetration"],
                )
            # Contacts/events still run on every microstep. Carry a tensor-only
            # collision flag to the next tick, where it shares the one host
            # decision already required by dynamic edge-set invalidation.
            if learned_effect_stride > 1:
                recompute_after_collision = torch.logical_or(
                    result.auxiliary["pair_collision"].any(),
                    result.auxiliary["boundary_collision"].any(),
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
        assert interval_pair_event_logits is not None
        assert interval_boundary_event_logits is not None
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
                "pair_event_logits": interval_pair_event_logits,
                "boundary_event_logits": interval_boundary_event_logits,
                "max_penetration": interval_max_penetration,
                "learned_effect_evaluation_count": torch.full(
                    (belief.batch_size,),
                    learned_effect_evaluations,
                    device=belief.device,
                    dtype=torch.int64,
                ),
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
        return_auxiliary: bool = True,
        auxiliary_names: Collection[str] | None = None,
    ) -> BeliefTrajectory:
        """Predict at sorted future offsets in seconds without mutating input.

        Public rollouts retain interval auxiliaries by default. Callers that
        consume only trajectory state and event logits may disable their
        collection to avoid retaining and stacking unused tensors.
        """

        return self.rollout_engine.rollout(
            lambda current, dt: self._predict_step(current, dt),
            belief,
            query_times,
            return_events=return_events,
            return_auxiliary=return_auxiliary,
            auxiliary_names=auxiliary_names,
        )
