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
from world_model.dynamics.actions import WorldAction, WorldImpulseAction, WorldImpulseSchedule
from world_model.dynamics.analytic import AnalyticKinematics
from world_model.dynamics.applicability import (
    PairApplicability,
    PairApplicabilityConfig,
    apply_pair_applicability,
)
from world_model.dynamics.attention import TypedAttentionInteractionResidual
from world_model.dynamics.contacts import ContactPlane, SphereContactResolver
from world_model.dynamics.event_driven import integrate_event_driven_state_only
from world_model.dynamics.events import EventModel
from world_model.dynamics.graph import InteractionGraph, InteractionOutput
from world_model.dynamics.modal import ModalDynamics
from world_model.dynamics.rigid_contacts import RigidContactResolver6D
from world_model.dynamics.rollout import RolloutEngine, RolloutStep
from world_model.dynamics.uncertainty import UncertaintyDynamics

_ACTION_INTERVAL_OR_AUXILIARIES = frozenset(
    {
        "interval_pair_contact",
        "pair_collision",
        "interval_boundary_contact",
        "boundary_collision",
        "interval_ground_contact",
        "ground_collision",
    }
)
_ACTION_INTERVAL_MAX_AUXILIARIES = frozenset(
    {
        "pair_impulse",
        "pair_collision_logits",
        "max_penetration",
    }
)
_ACTION_INTERVAL_EVENT_AUXILIARIES = frozenset(
    {
        "pair_event_logits",
        "boundary_event_logits",
    }
)


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
    # Specification 1.61 fields are appended so legacy positional
    # constructors retain their exact public argument mapping.
    graph_relation_hidden_dim: int | None = None
    modal_dynamics_enabled: bool = True
    continuous_pair_force_enabled: bool = True
    node_acceleration_enabled: bool = True
    # Opt-in specification-1.61 execution path. It is used only when callers
    # explicitly request neither event nor auxiliary traces; ordinary rollout
    # and every historical checkpoint retain the fixed-microstep function.
    event_driven_state_only_enabled: bool = False
    # Give the relation edge's signed process-noise output a direct, bounded
    # gradient path into propagated variance. Disabled is historical behavior.
    relation_process_uncertainty_enabled: bool = False
    # Opt-in relation execution that evaluates the learned edge MLP only on
    # active candidate pairs while reconstructing the established dense
    # result. False preserves historical checkpoint/runtime behavior.
    packed_interactions_enabled: bool = False
    # Opt-in rigid-body impulses with observable contact points, analytic body
    # inertia, angular velocity, and frictional torque.  False is the exact
    # historical linear-contact path.
    rigid_six_dof_contacts_enabled: bool = False

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
        if self.graph_hidden_dim <= 0 or (
            self.graph_relation_hidden_dim is not None and self.graph_relation_hidden_dim <= 0
        ):
            raise ValueError("graph hidden dimensions must be positive")
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
            ("modal_dynamics_enabled", self.modal_dynamics_enabled),
            ("continuous_pair_force_enabled", self.continuous_pair_force_enabled),
            ("node_acceleration_enabled", self.node_acceleration_enabled),
            ("event_driven_state_only_enabled", self.event_driven_state_only_enabled),
            (
                "relation_process_uncertainty_enabled",
                self.relation_process_uncertainty_enabled,
            ),
            ("packed_interactions_enabled", self.packed_interactions_enabled),
            ("rigid_six_dof_contacts_enabled", self.rigid_six_dof_contacts_enabled),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be boolean")
        if self.event_driven_state_only_enabled and (
            self.modal_dynamics_enabled
            or self.continuous_pair_force_enabled
            or self.node_acceleration_enabled
            or self.attention_residual_enabled
            or self.world_bounds is None
        ):
            raise ValueError(
                "event-driven state-only dynamics requires relation-only pair "
                "impulses and explicit world bounds"
            )
        if self.event_driven_state_only_enabled and self.rigid_six_dof_contacts_enabled:
            raise ValueError("six-DoF rigid contacts require fixed-microstep execution")
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
            relation_hidden_dim=self.config.graph_relation_hidden_dim,
            interaction_radius=self.config.interaction_radius,
            max_pair_force=self.config.max_pair_force,
            max_node_acceleration=self.config.max_node_acceleration,
            continuous_pair_force_enabled=self.config.continuous_pair_force_enabled,
            node_acceleration_enabled=self.config.node_acceleration_enabled,
            bounded_event_calibration_enabled=(self.config.event_driven_state_only_enabled),
            packed_interactions_enabled=self.config.packed_interactions_enabled,
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
        resolver_type = (
            RigidContactResolver6D
            if self.config.rigid_six_dof_contacts_enabled
            else SphereContactResolver
        )
        resolver = resolver_type(
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
            graph_relation_hidden_dim=(
                None
                if getattr(dynamics, "relation_hidden_dim", None) is None
                else int(dynamics.relation_hidden_dim)
            ),
            modal_dynamics_enabled=bool(getattr(dynamics, "modal_dynamics_enabled", True)),
            continuous_pair_force_enabled=bool(
                getattr(dynamics, "continuous_pair_force_enabled", True)
            ),
            node_acceleration_enabled=bool(getattr(dynamics, "node_acceleration_enabled", True)),
            event_driven_state_only_enabled=bool(
                getattr(dynamics, "event_driven_state_only_enabled", False)
            ),
            relation_process_uncertainty_enabled=bool(
                getattr(dynamics, "relation_process_uncertainty_enabled", False)
            ),
            packed_interactions_enabled=bool(
                getattr(dynamics, "packed_interactions_enabled", False)
            ),
            rigid_six_dof_contacts_enabled=bool(
                getattr(dynamics, "rigid_six_dof_contacts_enabled", False)
            ),
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
        if not torch.logical_and(torch.isfinite(value), value >= 0).all():
            raise ValueError("dt must contain finite nonnegative seconds")
        return value

    @staticmethod
    def _optional_advance_mask(sub_dt: Tensor) -> Tensor | None:
        """Return ``None`` when every batch row advances this segment.

        The elapsed tensor has already passed the public finite/nonnegative
        guard. One segment-level host decision selects the common all-positive
        path; physical microsteps then remain tensor-only. A tensor mask is
        retained unchanged for mixed positive/zero batches.
        """

        advance = sub_dt > 0
        if advance.all().detach().cpu().item():
            return None
        return advance

    @staticmethod
    def _validate_finite_segment(step: RolloutStep) -> RolloutStep:
        """Reject non-finite prediction output with one host decision.

        Child dynamics run tensor-only on parent-validated elapsed time.  This
        boundary reduction preserves fail-fast numerical integrity without
        synchronizing an accelerator once per child and physical microstep.
        It includes every floating tensor the composite dynamics returns, so
        a masked or auxiliary-only failure cannot escape detection.
        """

        tensors: list[Tensor] = [
            step.belief.timestamp,
            step.belief.gravity,
            step.belief.global_code,
            step.belief.global_log_variance,
            step.event_logits,
        ]
        tensors.extend(
            value
            for item in fields(step.belief.objects)
            if (value := getattr(step.belief.objects, item.name)).is_floating_point()
            or value.is_complex()
        )
        tensors.extend(
            value
            for item in fields(step.belief.camera)
            if (value := getattr(step.belief.camera, item.name)).is_floating_point()
            or value.is_complex()
        )
        tensors.extend(
            value
            for value in step.auxiliary.values()
            if value.is_floating_point() or value.is_complex()
        )
        device = step.belief.device
        if any(value.device != device for value in tensors):
            raise ValueError("dynamics segment output spans multiple devices")
        # The bounded belief/auxiliary tensors are small. Fusing them before
        # the reduction avoids launching one finite-check kernel per field and
        # still makes exactly one Python/host decision for the segment.
        with torch.no_grad():
            finite = torch.isfinite(torch.cat([value.reshape(-1) for value in tensors])).all()
        if not finite.detach().cpu().item():
            raise ValueError("dynamics segment output contains NaN or Inf")
        return step

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

    def _calibrated_pair_event_logits(
        self,
        analytic_logits: Tensor,
        interaction: InteractionOutput,
    ) -> Tensor:
        """Attach the opt-in bounded relation residual to pair confidence.

        The smooth-hazard event model already includes the graph residual.
        Historical hard-event profiles did not include it in their pairwise
        auxiliary, so only the specification-1.61 bounded graph path augments
        that auxiliary here.  The hard resolver outputs are untouched.
        """

        if (
            not self.interactions.bounded_event_calibration_enabled
            or self.config.smooth_event_hazard_enabled
        ):
            return analytic_logits
        residual = torch.stack(
            (interaction.contact_logits, interaction.collision_logits),
            dim=-1,
        )
        return analytic_logits + torch.where(
            interaction.edge_mask.unsqueeze(-1),
            residual,
            torch.zeros_like(residual),
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
        advance_mask: Tensor | None,
        external_acceleration: Tensor | None = None,
    ) -> RolloutStep:
        objects = belief.objects
        total_residual = modal_acceleration + interaction.residual_acceleration
        objects = self.analytic._integrate_validated_dt(
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
        relation_process_log_scale: Tensor | None = None
        if self.config.relation_process_uncertainty_enabled:
            interaction_uncertainty = interaction.interaction_density
            relation_process_log_scale = interaction.edge_process_noise.sum(dim=-1)
        else:
            # Exact historical path: the signed edge output only perturbs the
            # learned density feature. New relation-only profiles opt into the
            # direct calibrated log-scale path above.
            interaction_uncertainty = (
                interaction.interaction_density + interaction.edge_process_noise.sum(dim=-1)
            ).clamp_min(0.0)
        uncertainty = self.uncertainty._forward_validated_dt(
            events.objects,
            dt,
            event_logits=events.event_logits,
            interaction_density=interaction_uncertainty,
            residual_acceleration=total_residual,
            process_log_scale_residual=relation_process_log_scale,
        )
        updated_objects = self._blend_objects(
            belief.objects,
            uncertainty.objects,
            advance_mask,
        )
        updated = replace(
            belief,
            timestamp=belief.timestamp + dt,
            objects=updated_objects,
        )
        pair_event_logits = self._calibrated_pair_event_logits(
            events.pair_event_logits,
            interaction,
        )
        auxiliary_values = {
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
            "pair_event_logits": pair_event_logits,
            "boundary_event_logits": events.boundary_event_logits,
            "max_penetration": events.contacts.max_penetration,
            "mean_penetration": events.contacts.mean_penetration,
            "action_reaction_residual": (events.contacts.action_reaction_residual),
            "process_variance": uncertainty.process_variance,
            "edge_process_noise": interaction.edge_process_noise,
            "residual_acceleration": total_residual,
            "pair_applicability": applicability.pair,
            "collision_applicability": applicability.collision,
        }
        if self.interactions.bounded_event_calibration_enabled:
            # This is deliberately distinct from the authoritative boolean
            # ``pair_collision`` resolver result.  Qualification and the
            # disposable screen score this calibrated confidence tensor.
            auxiliary_values["pair_collision_logits"] = pair_event_logits[..., 1]
        auxiliary = self._mask_auxiliary(
            auxiliary_values,
            advance_mask,
        )
        if advance_mask is None:
            event_logits = events.event_logits
        else:
            for name in ("pair_event_logits", "boundary_event_logits"):
                value = auxiliary[name]
                mask = advance_mask
                while mask.ndim < value.ndim:
                    mask = mask.unsqueeze(-1)
                auxiliary[name] = torch.where(
                    mask,
                    value,
                    value.new_full((), -4.0),
                )
            event_logits = torch.where(
                advance_mask[:, None, None],
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
        update_batch: Tensor | None,
    ) -> ObjectBeliefTensor:
        if update_batch is None:
            return updated
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
        update_batch: Tensor | None,
    ) -> dict[str, Tensor]:
        if update_batch is None:
            return values
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
        if self.interactions.bounded_event_calibration_enabled:
            auxiliary["pair_collision_logits"] = objects.position.new_full(
                (batch, count, count),
                -4.0,
            )
        return RolloutStep(
            belief=belief.clone(),
            event_logits=event_logits,
            auxiliary=auxiliary,
        )

    @staticmethod
    def _belief_batch_row(belief: WorldBelief, index: int) -> WorldBelief:
        """Return one batch row without changing its physical representation."""

        objects = belief.objects.replace(
            **{
                item.name: getattr(belief.objects, item.name)[index : index + 1]
                for item in fields(belief.objects)
            }
        )
        camera = belief.camera.replace(
            **{
                item.name: getattr(belief.camera, item.name)[index : index + 1]
                for item in fields(belief.camera)
            }
        )
        return belief.replace(
            timestamp=belief.timestamp[index : index + 1],
            objects=objects,
            camera=camera,
            gravity=belief.gravity[index : index + 1],
            global_code=belief.global_code[index : index + 1],
            global_log_variance=belief.global_log_variance[index : index + 1],
            next_object_id=belief.next_object_id[index : index + 1],
        )

    @staticmethod
    def _concatenate_batch_steps(steps: Sequence[RolloutStep]) -> RolloutStep:
        """Reassemble independently integrated rows in their original order."""

        if not steps:
            raise ValueError("at least one rollout step is required")
        first = steps[0]
        if any(
            step.belief.active_modalities != first.belief.active_modalities
            or step.auxiliary.keys() != first.auxiliary.keys()
            for step in steps[1:]
        ):
            raise RuntimeError("independent batch rows emitted incompatible schemas")
        objects = first.belief.objects.replace(
            **{
                item.name: torch.cat(
                    [getattr(step.belief.objects, item.name) for step in steps],
                    dim=0,
                )
                for item in fields(first.belief.objects)
            }
        )
        camera = first.belief.camera.replace(
            **{
                item.name: torch.cat(
                    [getattr(step.belief.camera, item.name) for step in steps],
                    dim=0,
                )
                for item in fields(first.belief.camera)
            }
        )
        belief = first.belief.replace(
            timestamp=torch.cat([step.belief.timestamp for step in steps], dim=0),
            objects=objects,
            camera=camera,
            gravity=torch.cat([step.belief.gravity for step in steps], dim=0),
            global_code=torch.cat([step.belief.global_code for step in steps], dim=0),
            global_log_variance=torch.cat(
                [step.belief.global_log_variance for step in steps],
                dim=0,
            ),
            next_object_id=torch.cat([step.belief.next_object_id for step in steps], dim=0),
        )
        return DynamicsModel._validate_finite_segment(
            RolloutStep(
                belief=belief,
                event_logits=torch.cat([step.event_logits for step in steps], dim=0),
                auxiliary={
                    name: torch.cat([step.auxiliary[name] for step in steps], dim=0)
                    for name in first.auxiliary
                },
            )
        )

    def _predict_step_batch_independent(
        self,
        belief: WorldBelief,
        elapsed: Tensor,
        *,
        external_acceleration: Tensor | None = None,
    ) -> RolloutStep:
        """Integrate nonuniform action-split rows with exact B1 semantics."""

        positive_elapsed = elapsed.masked_select(elapsed > 0.0)
        # Zero-duration rows are already isolated by ``advance_mask``.  When
        # every advancing row has the same duration, the shared microstep grid
        # is exactly its B1 grid and the established vectorized fast path is
        # both independent and cheaper. Split only genuinely heterogeneous
        # positive durations.
        if (
            belief.batch_size == 1
            or positive_elapsed.numel() == 0
            or torch.equal(
                positive_elapsed,
                positive_elapsed[:1].expand_as(positive_elapsed),
            )
        ):
            return self._predict_step(
                belief,
                elapsed,
                external_acceleration=external_acceleration,
            )
        return self._concatenate_batch_steps(
            tuple(
                self._predict_step(
                    self._belief_batch_row(belief, index),
                    elapsed[index : index + 1],
                    external_acceleration=(
                        None
                        if external_acceleration is None
                        else external_acceleration[index : index + 1]
                    ),
                )
                for index in range(belief.batch_size)
            )
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
            return self._validate_finite_segment(self._zero_step(output))
        sub_dt = elapsed / substeps
        advance_mask = self._optional_advance_mask(sub_dt)
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
            if self.config.modal_dynamics_enabled:
                modal_objects, modal = self.modal._forward_validated_dt(
                    output.objects,
                    sub_dt,
                )
                output = replace(output, objects=modal_objects)
                modal_acceleration = modal.residual_acceleration
            else:
                modal_acceleration = torch.zeros_like(output.objects.position)
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
                    modal_acceleration,
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
                modal_acceleration=modal_acceleration,
                interaction=step_interaction,
                applicability=applicability,
                advance_mask=advance_mask,
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
        if self.interactions.bounded_event_calibration_enabled:
            auxiliary["pair_collision_logits"] = interval_pair_event_logits[..., 1]
        return self._validate_finite_segment(
            RolloutStep(
                belief=final_belief,
                event_logits=event_logits,
                auxiliary=auxiliary,
            )
        )

    @staticmethod
    def _where_batch(mask: Tensor, when_true: Tensor, when_false: Tensor) -> Tensor:
        shaped = mask
        while shaped.ndim < when_true.ndim:
            shaped = shaped.unsqueeze(-1)
        return torch.where(shaped, when_true, when_false)

    @staticmethod
    def _with_known_action_event(event_logits: Tensor, applied: Tensor) -> Tensor:
        actuated = event_logits.new_full(event_logits.shape, -4.0)
        # ``event_logits`` describes every event that occurred in the complete
        # prediction interval, rather than only one mutually-exclusive endpoint
        # mode.  Keep collision evidence on an actuated object; the typed known-
        # action auxiliaries disambiguate a simultaneous external impulse.
        actuated[..., MotionMode.COLLISION] = event_logits[..., MotionMode.COLLISION]
        actuated[..., MotionMode.EXTERNALLY_ACTUATED] = 4.0
        return torch.where(applied.unsqueeze(-1), actuated, event_logits)

    @staticmethod
    def _known_action_auxiliary(
        belief: WorldBelief,
        action: WorldImpulseAction,
        applied: Tensor,
    ) -> dict[str, Tensor]:
        impulse = action.impulse_world.unsqueeze(1).expand(
            -1,
            belief.objects.max_objects,
            -1,
        )
        return {
            "known_action_applied": applied,
            "known_impulse_world": torch.where(
                applied.unsqueeze(-1),
                impulse,
                torch.zeros_like(impulse),
            ),
        }

    def _without_applied_action(
        self,
        step: RolloutStep,
        action: WorldImpulseAction,
    ) -> RolloutStep:
        applied = torch.zeros_like(step.belief.objects.active)
        auxiliary = dict(step.auxiliary)
        auxiliary.update(self._known_action_auxiliary(step.belief, action, applied))
        return RolloutStep(
            belief=step.belief,
            event_logits=step.event_logits,
            auxiliary=auxiliary,
        )

    def _combine_action_segments(
        self,
        before: RolloutStep,
        after: RolloutStep,
        *,
        target_timestamp: Tensor,
        after_elapsed: Tensor,
        action: WorldImpulseAction,
        applied: Tensor,
    ) -> RolloutStep:
        """Compose the two physical intervals surrounding one known impulse."""

        use_after = after_elapsed > 0.0
        event_logits = self._where_batch(
            use_after,
            after.event_logits,
            before.event_logits,
        ).clone()
        event_logits[..., MotionMode.COLLISION] = torch.maximum(
            before.event_logits[..., MotionMode.COLLISION],
            after.event_logits[..., MotionMode.COLLISION],
        )
        event_logits = self._with_known_action_event(event_logits, applied)

        if before.auxiliary.keys() != after.auxiliary.keys():
            raise RuntimeError("split dynamics segments emitted different auxiliary schemas")
        auxiliary: dict[str, Tensor] = {}
        for name, before_value in before.auxiliary.items():
            after_value = after.auxiliary[name]
            if name in _ACTION_INTERVAL_OR_AUXILIARIES:
                value = before_value | after_value
            elif name in _ACTION_INTERVAL_MAX_AUXILIARIES:
                value = torch.maximum(before_value, after_value)
            elif name in _ACTION_INTERVAL_EVENT_AUXILIARIES:
                current = self._where_batch(use_after, after_value, before_value)
                value = torch.stack(
                    (
                        current[..., 0],
                        torch.maximum(before_value[..., 1], after_value[..., 1]),
                    ),
                    dim=-1,
                )
            elif name == "learned_effect_evaluation_count":
                # A vectorized action batch may contain rows that do not own
                # the action.  Their post-action duration is exactly zero,
                # while other rows still cause the shared analytic step to
                # report its (global) learned-effect evaluation count.  Do
                # not attribute those other rows' work to an untouched
                # candidate: its public diagnostic must remain identical to
                # the ordinary no-action rollout.
                value = before_value + torch.where(
                    use_after,
                    after_value,
                    torch.zeros_like(after_value),
                )
            else:
                value = self._where_batch(use_after, after_value, before_value)
            auxiliary[name] = value
        auxiliary.update(self._known_action_auxiliary(after.belief, action, applied))
        endpoint = after.belief.replace(timestamp=target_timestamp)
        return self._validate_finite_segment(
            RolloutStep(
                belief=endpoint,
                event_logits=event_logits,
                auxiliary=auxiliary,
            )
        )

    def _combine_interval_segments(
        self,
        accumulated: RolloutStep,
        segment: RolloutStep,
        *,
        use_segment: Tensor,
    ) -> RolloutStep:
        """Accumulate physical evidence across an arbitrary causal split."""

        event_logits = self._where_batch(
            use_segment,
            segment.event_logits,
            accumulated.event_logits,
        ).clone()
        event_logits[..., MotionMode.COLLISION] = torch.maximum(
            accumulated.event_logits[..., MotionMode.COLLISION],
            segment.event_logits[..., MotionMode.COLLISION],
        )
        if accumulated.auxiliary.keys() != segment.auxiliary.keys():
            raise RuntimeError("split dynamics segments emitted different auxiliary schemas")
        auxiliary: dict[str, Tensor] = {}
        for name, previous in accumulated.auxiliary.items():
            current = segment.auxiliary[name]
            if name in _ACTION_INTERVAL_OR_AUXILIARIES:
                value = previous | current
            elif name in _ACTION_INTERVAL_MAX_AUXILIARIES:
                value = torch.maximum(previous, current)
            elif name in _ACTION_INTERVAL_EVENT_AUXILIARIES:
                endpoint = self._where_batch(use_segment, current, previous)
                value = torch.stack(
                    (
                        endpoint[..., 0],
                        torch.maximum(previous[..., 1], current[..., 1]),
                    ),
                    dim=-1,
                )
            elif name == "learned_effect_evaluation_count":
                value = previous + torch.where(
                    use_segment,
                    current,
                    torch.zeros_like(current),
                )
            else:
                value = self._where_batch(use_segment, current, previous)
            auxiliary[name] = value
        return self._validate_finite_segment(
            RolloutStep(
                belief=segment.belief,
                event_logits=event_logits,
                auxiliary=auxiliary,
            )
        )

    def _predict_step_with_validated_schedule(
        self,
        belief: WorldBelief,
        elapsed: Tensor,
        schedule: WorldImpulseSchedule,
        consumed: Tensor,
    ) -> tuple[RolloutStep, Tensor]:
        """Split one interval around every due impulse in timestamp order."""

        target_timestamp = belief.timestamp + elapsed
        current = belief
        accumulated = self._zero_step(belief)
        applied_any = torch.zeros_like(belief.objects.active)
        impulse_sum = torch.zeros_like(belief.objects.position)
        action_count = torch.zeros_like(belief.objects.object_id)
        updated_consumed = consumed.clone()

        for index, action in enumerate(schedule.actions):
            application_mask = action._application_mask_for(belief)
            owns_action = (
                application_mask
                & ~updated_consumed[:, index]
                & (current.timestamp < action.timestamp)
                & (action.timestamp <= target_timestamp)
            )
            before_elapsed = torch.where(
                owns_action,
                action.timestamp - current.timestamp,
                torch.zeros_like(elapsed),
            )
            before = self._predict_step_batch_independent(current, before_elapsed)
            accumulated = self._combine_interval_segments(
                accumulated,
                before,
                use_segment=before_elapsed > 0.0,
            )
            current = before.belief

            target_mask = current.objects.active & (
                current.objects.object_id == action.object_id.unsqueeze(-1)
            )
            physical_target = owns_action.unsqueeze(-1) & target_mask
            nonzero = torch.any(action.impulse_world != 0.0, dim=-1)
            applied = physical_target & nonzero.unsqueeze(-1)
            velocity_jump = action.impulse_world.unsqueeze(1) / current.objects.mass
            velocity = current.objects.velocity + torch.where(
                physical_target.unsqueeze(-1),
                velocity_jump,
                torch.zeros_like(velocity_jump),
            )
            current = current.replace(
                timestamp=torch.where(owns_action, action.timestamp, current.timestamp),
                objects=current.objects.replace(velocity=velocity),
            )
            accumulated = RolloutStep(
                belief=current,
                event_logits=accumulated.event_logits,
                auxiliary=accumulated.auxiliary,
            )
            applied_any = applied_any | applied
            expanded_impulse = action.impulse_world.unsqueeze(1).expand_as(impulse_sum)
            impulse_sum = impulse_sum + torch.where(
                applied.unsqueeze(-1),
                expanded_impulse,
                torch.zeros_like(expanded_impulse),
            )
            action_count = action_count + applied.to(dtype=torch.int64)
            updated_consumed[:, index] = updated_consumed[:, index] | owns_action

        after_elapsed = (target_timestamp - current.timestamp).clamp_min(0.0)
        after = self._predict_step_batch_independent(current, after_elapsed)
        accumulated = self._combine_interval_segments(
            accumulated,
            after,
            use_segment=after_elapsed > 0.0,
        )
        auxiliary = dict(accumulated.auxiliary)
        auxiliary.update(
            {
                "known_action_applied": applied_any,
                "known_impulse_world": impulse_sum,
                "known_action_count": action_count,
            }
        )
        return (
            self._validate_finite_segment(
                RolloutStep(
                    belief=accumulated.belief.replace(timestamp=target_timestamp),
                    event_logits=self._with_known_action_event(
                        accumulated.event_logits,
                        applied_any,
                    ),
                    auxiliary=auxiliary,
                )
            ),
            updated_consumed,
        )

    def _predict_step_with_validated_action(
        self,
        belief: WorldBelief,
        elapsed: Tensor,
        action: WorldImpulseAction,
        owns_action: Tensor,
        nonzero_action: Tensor,
    ) -> RolloutStep:
        """Split at an absolute action time and inject momentum on owned rows."""

        target_timestamp = belief.timestamp + elapsed
        before_elapsed = torch.where(
            owns_action,
            action.timestamp - belief.timestamp,
            elapsed,
        )
        before = self._predict_step_batch_independent(belief, before_elapsed)

        target_mask = before.belief.objects.active & (
            before.belief.objects.object_id == action.object_id.unsqueeze(-1)
        )
        # A numerically zero impulse remains on the same differentiable
        # physical path as every other typed action.  ``nonzero_action`` only
        # controls the externally-actuated event/auxiliary label; using it to
        # bypass the jump would sever the derivative at the zero-action
        # baseline used by counterfactual diagnostics.
        physical_target = owns_action.unsqueeze(-1) & target_mask
        applied = physical_target & nonzero_action.unsqueeze(-1)
        velocity_jump = action.impulse_world.unsqueeze(1) / before.belief.objects.mass
        velocity = before.belief.objects.velocity + torch.where(
            physical_target.unsqueeze(-1),
            velocity_jump,
            torch.zeros_like(velocity_jump),
        )
        split_timestamp = torch.where(
            owns_action,
            action.timestamp,
            before.belief.timestamp,
        )
        split_belief = before.belief.replace(
            timestamp=split_timestamp,
            objects=before.belief.objects.replace(velocity=velocity),
        )
        after_elapsed = torch.where(
            owns_action,
            (target_timestamp - action.timestamp).clamp_min(0.0),
            torch.zeros_like(elapsed),
        )
        after = self._predict_step_batch_independent(split_belief, after_elapsed)
        return self._combine_action_segments(
            before,
            after,
            target_timestamp=target_timestamp,
            after_elapsed=after_elapsed,
            action=action,
            applied=applied,
        )

    def _event_driven_state_step(
        self,
        belief: WorldBelief,
        elapsed: Tensor,
        *,
        collect_pair_events: bool = False,
    ) -> RolloutStep:
        """Run the opt-in continuous-time state-only path or fail safe.

        The event-driven helper returns ``None`` whenever its proof obligations
        do not hold. Falling back from the untouched source belief preserves
        the authoritative fixed-microstep semantics for arbitrary callers.
        """

        result = integrate_event_driven_state_only(
            belief.objects,
            belief.gravity,
            belief.global_code,
            elapsed,
            analytic=self.analytic,
            interactions=self.interactions,
            world_bounds=self.config.world_bounds or (),
            collision_speed_epsilon=self.config.pair_collision_speed_epsilon,
            maximum_multiplier_residual=(self.events.resolver.max_impulse_multiplier_residual),
            maximum_additive_residual=(self.events.resolver.max_impulse_additive_residual),
        )
        if result is None:
            return self._predict_step(belief, elapsed)

        relation_log_scale = (
            result.relation_process_log_scale
            if self.config.relation_process_uncertainty_enabled
            else None
        )
        # State-only consumers still receive a calibrated variance trajectory.
        # One closed-form interval update avoids the 120 Hz Python loop; the
        # relation-owned log scale is an exact identity at zero residual.
        uncertainty = self.uncertainty._forward_validated_dt(
            result.objects,
            elapsed,
            event_logits=None,
            interaction_density=result.collision_count.to(dtype=belief.dtype)
            .unsqueeze(-1)
            .expand_as(belief.objects.active),
            residual_acceleration=torch.zeros_like(result.objects.position),
            process_log_scale_residual=relation_log_scale,
        )
        endpoint = belief.replace(
            timestamp=belief.timestamp + elapsed,
            objects=uncertainty.objects,
        )
        if not collect_pair_events:
            # RolloutEngine still requires a step event tensor even when its
            # caller disables event retention. No event claim crosses that
            # public state-only result.
            return RolloutStep(
                belief=endpoint,
                event_logits=endpoint.objects.motion_mode_logits,
                auxiliary={},
            )
        # The compact pair-event summary is useful to physical-objective
        # consumers and costs one relation evaluation per interval rather than
        # one per 120 Hz microstep.  The hard analytic collision remains the
        # authoritative forward signal; the learned collision output is only
        # a bounded calibration residual with a live gradient.
        relation = self.interactions(belief.objects, belief.global_code)
        count = belief.objects.max_objects
        identity = torch.eye(count, dtype=torch.bool, device=belief.device).unsqueeze(0)
        valid_pair = (
            belief.objects.active[:, :, None] & belief.objects.active[:, None, :] & ~identity
        )
        endpoint_delta = (
            endpoint.objects.position[:, None, :, :] - endpoint.objects.position[:, :, None, :]
        )
        endpoint_distance = torch.linalg.vector_norm(endpoint_delta, dim=-1)
        endpoint_radius = endpoint.objects.radius[..., 0]
        pair_contact = valid_pair & (
            endpoint_distance
            <= endpoint_radius[:, :, None]
            + endpoint_radius[:, None, :]
            + self.config.contact_margin
        )
        contact_logits = torch.where(
            pair_contact,
            endpoint_distance.new_full((), 4.0),
            endpoint_distance.new_full((), -4.0),
        ) + torch.where(
            relation.edge_mask,
            relation.contact_logits,
            torch.zeros_like(relation.contact_logits),
        )
        collision_logits = torch.where(
            result.pair_collision,
            endpoint_distance.new_full((), 6.0),
            endpoint_distance.new_full((), -4.0),
        ) + torch.where(
            relation.edge_mask,
            relation.collision_logits,
            torch.zeros_like(relation.collision_logits),
        )
        pair_event_logits = torch.stack((contact_logits, collision_logits), dim=-1)
        pair_event_logits = torch.where(
            valid_pair.unsqueeze(-1),
            pair_event_logits,
            pair_event_logits.new_full((), -4.0),
        )
        return RolloutStep(
            belief=endpoint,
            event_logits=endpoint.objects.motion_mode_logits,
            auxiliary={
                "pair_collision": result.pair_collision,
                "pair_event_logits": pair_event_logits,
                "pair_collision_logits": pair_event_logits[..., 1],
            },
        )

    def _event_driven_state_step_with_validated_action(
        self,
        belief: WorldBelief,
        elapsed: Tensor,
        action: WorldImpulseAction,
        owns_action: Tensor,
        *,
        collect_pair_events: bool = False,
    ) -> RolloutStep:
        """Split one state-only segment at its absolute public action time."""

        target_timestamp = belief.timestamp + elapsed
        before_elapsed = torch.where(
            owns_action,
            action.timestamp - belief.timestamp,
            elapsed,
        )
        before = self._event_driven_state_step(
            belief,
            before_elapsed,
            collect_pair_events=collect_pair_events,
        )
        target_mask = before.belief.objects.active & (
            before.belief.objects.object_id == action.object_id.unsqueeze(-1)
        )
        physical_target = owns_action.unsqueeze(-1) & target_mask
        velocity_jump = action.impulse_world.unsqueeze(1) / before.belief.objects.mass
        velocity = before.belief.objects.velocity + torch.where(
            physical_target.unsqueeze(-1),
            velocity_jump,
            torch.zeros_like(velocity_jump),
        )
        split_belief = before.belief.replace(
            timestamp=torch.where(
                owns_action,
                action.timestamp,
                before.belief.timestamp,
            ),
            objects=before.belief.objects.replace(velocity=velocity),
        )
        after_elapsed = torch.where(
            owns_action,
            (target_timestamp - action.timestamp).clamp_min(0.0),
            torch.zeros_like(elapsed),
        )
        after = self._event_driven_state_step(
            split_belief,
            after_elapsed,
            collect_pair_events=collect_pair_events,
        )
        if not collect_pair_events:
            return RolloutStep(
                belief=after.belief.replace(timestamp=target_timestamp),
                event_logits=after.event_logits,
                auxiliary={},
            )
        before_pair_logits = before.auxiliary["pair_event_logits"]
        after_pair_logits = after.auxiliary["pair_event_logits"]
        pair_event_logits = torch.stack(
            (
                after_pair_logits[..., 0],
                torch.maximum(before_pair_logits[..., 1], after_pair_logits[..., 1]),
            ),
            dim=-1,
        )
        return RolloutStep(
            belief=after.belief.replace(timestamp=target_timestamp),
            event_logits=after.event_logits,
            auxiliary={
                "pair_collision": (
                    before.auxiliary["pair_collision"] | after.auxiliary["pair_collision"]
                ),
                "pair_event_logits": pair_event_logits,
                "pair_collision_logits": pair_event_logits[..., 1],
            },
        )

    def _predict_event_driven_schedule_segment(
        self,
        belief: WorldBelief,
        elapsed: Tensor,
        schedule: WorldImpulseSchedule,
        consumed: Tensor,
        *,
        collect_pair_events: bool,
    ) -> tuple[RolloutStep, Tensor]:
        """Apply all due schedule entries on the fast state-only path."""

        target_timestamp = belief.timestamp + elapsed
        current = belief
        updated_consumed = consumed.clone()
        applied_any = torch.zeros_like(belief.objects.active)
        impulse_sum = torch.zeros_like(belief.objects.position)
        action_count = torch.zeros_like(belief.objects.object_id)
        pair_collision = torch.zeros(
            belief.batch_size,
            belief.objects.max_objects,
            belief.objects.max_objects,
            dtype=torch.bool,
            device=belief.device,
        )
        pair_event_logits = belief.objects.position.new_full(
            (
                belief.batch_size,
                belief.objects.max_objects,
                belief.objects.max_objects,
                2,
            ),
            -4.0,
        )
        latest_event_logits = belief.objects.motion_mode_logits.clone()
        latest_event_logits[..., MotionMode.COLLISION] = -4.0

        def integrate(duration: Tensor) -> None:
            nonlocal current, pair_collision, pair_event_logits, latest_event_logits
            segment = self._event_driven_state_step(
                current,
                duration,
                collect_pair_events=collect_pair_events,
            )
            used = duration > 0.0
            current = segment.belief
            latest_event_logits = self._where_batch(
                used,
                segment.event_logits,
                latest_event_logits,
            )
            if collect_pair_events:
                current_pair_logits = segment.auxiliary["pair_event_logits"]
                endpoint = self._where_batch(used, current_pair_logits, pair_event_logits)
                pair_event_logits = torch.stack(
                    (
                        endpoint[..., 0],
                        torch.maximum(
                            pair_event_logits[..., 1],
                            current_pair_logits[..., 1],
                        ),
                    ),
                    dim=-1,
                )
                pair_collision = pair_collision | segment.auxiliary["pair_collision"]

        for index, action in enumerate(schedule.actions):
            application_mask = action._application_mask_for(belief)
            owns_action = (
                application_mask
                & ~updated_consumed[:, index]
                & (current.timestamp < action.timestamp)
                & (action.timestamp <= target_timestamp)
            )
            before_elapsed = torch.where(
                owns_action,
                action.timestamp - current.timestamp,
                torch.zeros_like(elapsed),
            )
            integrate(before_elapsed)
            target_mask = current.objects.active & (
                current.objects.object_id == action.object_id.unsqueeze(-1)
            )
            physical_target = owns_action.unsqueeze(-1) & target_mask
            nonzero = torch.any(action.impulse_world != 0.0, dim=-1)
            applied = physical_target & nonzero.unsqueeze(-1)
            velocity_jump = action.impulse_world.unsqueeze(1) / current.objects.mass
            current = current.replace(
                timestamp=torch.where(owns_action, action.timestamp, current.timestamp),
                objects=current.objects.replace(
                    velocity=current.objects.velocity
                    + torch.where(
                        physical_target.unsqueeze(-1),
                        velocity_jump,
                        torch.zeros_like(velocity_jump),
                    )
                ),
            )
            applied_any = applied_any | applied
            expanded_impulse = action.impulse_world.unsqueeze(1).expand_as(impulse_sum)
            impulse_sum = impulse_sum + torch.where(
                applied.unsqueeze(-1),
                expanded_impulse,
                torch.zeros_like(expanded_impulse),
            )
            action_count = action_count + applied.to(dtype=torch.int64)
            updated_consumed[:, index] = updated_consumed[:, index] | owns_action

        integrate((target_timestamp - current.timestamp).clamp_min(0.0))
        auxiliary: dict[str, Tensor] = {}
        if collect_pair_events:
            auxiliary = {
                "pair_collision": pair_collision,
                "pair_event_logits": pair_event_logits,
                "pair_collision_logits": pair_event_logits[..., 1],
                "known_action_applied": applied_any,
                "known_impulse_world": impulse_sum,
                "known_action_count": action_count,
            }
        return (
            RolloutStep(
                belief=current.replace(timestamp=target_timestamp),
                event_logits=self._with_known_action_event(
                    latest_event_logits,
                    applied_any,
                ),
                auxiliary=auxiliary,
            ),
            updated_consumed,
        )

    def predict_state_only_step(
        self,
        belief: WorldBelief,
        dt: float | Tensor,
        *,
        action: WorldAction | None = None,
    ) -> RolloutStep:
        """Return one fast state interval plus compact pair-event evidence.

        This explicit entry point is restricted to profiles which opted into
        the event-driven state path.  Unsupported rows still fall back from
        the untouched source belief to the authoritative fixed-microstep
        implementation.
        """

        if not self.config.event_driven_state_only_enabled:
            raise RuntimeError("state-only interval prediction is not enabled")
        self._validate_dimensions(belief)
        elapsed = self._normalise_dt(belief, dt)
        if action is None:
            return self._event_driven_state_step(
                belief,
                elapsed,
                collect_pair_events=True,
            )
        if isinstance(action, WorldImpulseSchedule):
            action.validate_for(
                belief,
                latest_timestamp=belief.timestamp + elapsed,
            )
            application = torch.stack(
                [item._application_mask_for(belief) for item in action.actions],
                dim=-1,
            )
            step, consumed = self._predict_event_driven_schedule_segment(
                belief,
                elapsed,
                action,
                ~application,
                collect_pair_events=True,
            )
            if not bool(consumed.all()):
                raise RuntimeError("validated action schedule was not consumed")
            return step
        if not isinstance(action, WorldImpulseAction):
            raise TypeError("action must be a WorldImpulseAction, WorldImpulseSchedule, or None")
        target_mask = action.validate_for(
            belief,
            latest_timestamp=belief.timestamp + elapsed,
        )
        owns_action = action._application_mask_for(belief)
        step = self._event_driven_state_step_with_validated_action(
            belief,
            elapsed,
            action,
            owns_action,
            collect_pair_events=True,
        )
        applied = target_mask & torch.any(action.impulse_world != 0.0, dim=-1).unsqueeze(-1)
        auxiliary = dict(step.auxiliary)
        auxiliary.update(self._known_action_auxiliary(step.belief, action, applied))
        return RolloutStep(
            belief=step.belief,
            event_logits=step.event_logits,
            auxiliary=auxiliary,
        )

    def _event_driven_state_rollout(
        self,
        belief: WorldBelief,
        query_times: Tensor | Sequence[float],
        action: WorldAction | None,
    ) -> BeliefTrajectory:
        """Return an event-free trajectory from the opt-in fast state path."""

        offsets = self.validate_action_rollout(belief, query_times, action)
        if action is None:
            return self.rollout_engine.rollout(
                self._event_driven_state_step,
                belief,
                offsets,
                return_events=False,
                return_auxiliary=False,
            )

        if isinstance(action, WorldImpulseSchedule):
            application = torch.stack(
                [item._application_mask_for(belief) for item in action.actions],
                dim=-1,
            )
            consumed = ~application

            def predict_schedule_segment(
                current: WorldBelief,
                segment_elapsed: Tensor,
            ) -> RolloutStep:
                nonlocal consumed
                step, consumed = self._predict_event_driven_schedule_segment(
                    current,
                    segment_elapsed,
                    action,
                    consumed,
                    collect_pair_events=False,
                )
                return step

            trajectory = self.rollout_engine.rollout(
                predict_schedule_segment,
                belief,
                offsets,
                return_events=False,
                return_auxiliary=False,
            )
            if not bool(consumed.all()):
                raise RuntimeError("validated action schedule was not consumed")
            return trajectory

        assert isinstance(action, WorldImpulseAction)
        application_mask = action._application_mask_for(belief)
        consumed = ~application_mask

        def predict_action_segment(
            current: WorldBelief,
            segment_elapsed: Tensor,
        ) -> RolloutStep:
            nonlocal consumed
            target_timestamp = current.timestamp + segment_elapsed
            owns_action = (
                application_mask
                & ~consumed
                & (current.timestamp < action.timestamp)
                & (action.timestamp <= target_timestamp)
            )
            if not bool(owns_action.any()):
                return self._event_driven_state_step(current, segment_elapsed)
            step = self._event_driven_state_step_with_validated_action(
                current,
                segment_elapsed,
                action,
                owns_action,
            )
            consumed = consumed | owns_action
            return step

        trajectory = self.rollout_engine.rollout(
            predict_action_segment,
            belief,
            offsets,
            return_events=False,
            return_auxiliary=False,
        )
        if not bool(consumed.all()):
            raise RuntimeError("validated action was not consumed by the state-only rollout")
        return trajectory

    def predict(
        self,
        belief: WorldBelief,
        dt: float | Tensor,
        *,
        action: WorldAction | None = None,
    ) -> WorldBelief:
        """Predict a new belief after elapsed seconds without mutating input."""

        return self.predict_step(belief, dt, action=action).belief

    def predict_step(
        self,
        belief: WorldBelief,
        dt: float | Tensor,
        *,
        action: WorldAction | None = None,
    ) -> RolloutStep:
        """Return the endpoint belief plus events over the elapsed interval."""

        if action is None:
            return self._predict_step(belief, dt)
        elapsed = self._normalise_dt(belief, dt)
        if isinstance(action, WorldImpulseSchedule):
            action.validate_for(
                belief,
                latest_timestamp=belief.timestamp + elapsed,
            )
            application = torch.stack(
                [item._application_mask_for(belief) for item in action.actions],
                dim=-1,
            )
            step, consumed = self._predict_step_with_validated_schedule(
                belief,
                elapsed,
                action,
                ~application,
            )
            if not bool(consumed.all().detach().cpu().item()):
                raise RuntimeError("validated action schedule was not consumed")
            return step
        if not isinstance(action, WorldImpulseAction):
            raise TypeError("action must be a WorldImpulseAction, WorldImpulseSchedule, or None")
        action.validate_for(
            belief,
            latest_timestamp=belief.timestamp + elapsed,
        )
        application_mask = action._application_mask_for(belief)
        nonzero = torch.any(action.impulse_world != 0.0, dim=-1)
        return self._predict_step_with_validated_action(
            belief,
            elapsed,
            action,
            application_mask,
            nonzero,
        )

    def validate_action_rollout(
        self,
        belief: WorldBelief,
        query_times: Tensor | Sequence[float],
        action: WorldAction | None,
    ) -> Tensor:
        """Purely normalize rollout offsets and validate one optional action."""

        offsets = self.rollout_engine._normalise_query_times(belief, query_times)
        if action is None:
            return offsets
        if not isinstance(action, (WorldImpulseAction, WorldImpulseSchedule)):
            raise TypeError("action must be a WorldImpulseAction, WorldImpulseSchedule, or None")
        if offsets.shape[1] == 0:
            raise ValueError("an action rollout requires at least one query time")
        action.validate_for(
            belief,
            latest_timestamp=belief.timestamp + offsets[:, -1],
        )
        return offsets

    def rollout(
        self,
        belief: WorldBelief,
        query_times: Tensor | Sequence[float],
        *,
        action: WorldAction | None = None,
        return_events: bool = True,
        return_auxiliary: bool = True,
        auxiliary_names: Collection[str] | None = None,
    ) -> BeliefTrajectory:
        """Predict at sorted future offsets in seconds without mutating input.

        Public rollouts retain interval auxiliaries by default. Callers that
        consume only trajectory state and event logits may disable their
        collection to avoid retaining and stacking unused tensors.
        """

        if auxiliary_names is not None and not return_auxiliary:
            raise ValueError("auxiliary_names requires return_auxiliary=True")
        if (
            self.config.event_driven_state_only_enabled
            and not return_events
            and not return_auxiliary
        ):
            return self._event_driven_state_rollout(belief, query_times, action)

        if action is None:
            return self.rollout_engine.rollout(
                lambda current, dt: self._predict_step_batch_independent(current, dt),
                belief,
                query_times,
                return_events=return_events,
                return_auxiliary=return_auxiliary,
                auxiliary_names=auxiliary_names,
            )

        offsets = self.validate_action_rollout(belief, query_times, action)
        if isinstance(action, WorldImpulseSchedule):
            application = torch.stack(
                [item._application_mask_for(belief) for item in action.actions],
                dim=-1,
            )
            consumed = ~application

            def predict_schedule_segment(
                current: WorldBelief,
                elapsed: Tensor,
            ) -> RolloutStep:
                nonlocal consumed
                result, consumed = self._predict_step_with_validated_schedule(
                    current,
                    elapsed,
                    action,
                    consumed,
                )
                return result

            trajectory = self.rollout_engine.rollout(
                predict_schedule_segment,
                belief,
                offsets,
                return_events=return_events,
                return_auxiliary=return_auxiliary,
                auxiliary_names=auxiliary_names,
            )
            if not bool(consumed.all().detach().cpu().item()):
                raise RuntimeError("validated action schedule was not consumed by the rollout")
            return trajectory

        assert isinstance(action, WorldImpulseAction)
        application_mask = action._application_mask_for(belief)
        nonzero = torch.any(action.impulse_world != 0.0, dim=-1)
        consumed = ~application_mask

        def predict_action_segment(current: WorldBelief, elapsed: Tensor) -> RolloutStep:
            nonlocal consumed
            target_timestamp = current.timestamp + elapsed
            owns_action = (
                application_mask
                & ~consumed
                & (current.timestamp < action.timestamp)
                & (action.timestamp <= target_timestamp)
            )
            if not owns_action.any().detach().cpu().item():
                return self._without_applied_action(
                    self._predict_step_batch_independent(current, elapsed),
                    action,
                )
            result = self._predict_step_with_validated_action(
                current,
                elapsed,
                action,
                owns_action,
                nonzero,
            )
            consumed = consumed | owns_action
            return result

        trajectory = self.rollout_engine.rollout(
            predict_action_segment,
            belief,
            offsets,
            return_events=return_events,
            return_auxiliary=return_auxiliary,
            auxiliary_names=auxiliary_names,
        )
        if not consumed.all().detach().cpu().item():
            raise RuntimeError("validated action was not consumed by the rollout")
        return trajectory
