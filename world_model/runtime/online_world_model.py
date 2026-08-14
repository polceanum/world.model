"""Persistent event-driven predict–observe–associate–correct runtime."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from itertools import groupby
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from world_model.abstractions import (
    AbstractionAssignment,
    PredictiveAbstractionRouter,
    PredictiveTokenBatch,
    WorldBeliefTokenizer,
)
from world_model.belief import (
    BeliefFactory,
    BeliefTrajectory,
    LifecycleConfig,
    ObjectLifecycle,
    WorldBelief,
)
from world_model.filtering import (
    BeliefUpdater,
    BeliefUpdaterConfig,
)
from world_model.fusion import (
    Associator,
    ObservationMode,
    ObservationScheduler,
    SurpriseClassifier,
)
from world_model.identification import (
    ObservabilityConfig,
    ObservabilityEstimator,
    ParameterUpdaterConfig,
    RecurrentParameterUpdater,
)
from world_model.observations import (
    MeasurementSet,
    ObservationContext,
    ObservationModule,
    ObservationPacket,
    SensorContext,
)
from world_model.observations.registry import validate_module_mapping
from world_model.runtime.diagnostics import (
    RuntimeDiagnostics,
    RuntimeStepDiagnostics,
)
from world_model.runtime.prepared import (
    PreparedPropagation,
    PreparedPropagationError,
    TensorVersionSignature,
    tensor_identity_version_signature,
)
from world_model.runtime.state import RuntimeState

if TYPE_CHECKING:
    from world_model.utils.config import OrpheusConfig


class OutOfSequenceObservationError(ValueError):
    """The milestone-one filter rejects delayed observations explicitly."""


def _packet_batch_size(packet: ObservationPacket) -> int:
    payload = packet.payload
    if isinstance(payload, Tensor):
        return int(payload.shape[0]) if payload.ndim >= 4 else 1
    if isinstance(payload, Mapping):
        for value in payload.values():
            if isinstance(value, Tensor):
                # Oracle position is [N,3] or [B,N,3].
                return int(value.shape[0]) if value.ndim >= 3 else 1
    return 1


def _move_packet(
    packet: ObservationPacket,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> ObservationPacket:
    def move(value: object) -> object:
        if isinstance(value, Tensor):
            target_dtype = dtype if value.is_floating_point() else None
            return value.to(device=device, dtype=target_dtype)
        if isinstance(value, Mapping):
            return {key: move(item) for key, item in value.items()}
        return value

    return replace(
        packet,
        payload=move(packet.payload),
        calibration=move(packet.calibration),
    )


class OnlineWorldModel(nn.Module):
    """Stateful convenience runtime built from functional PyTorch modules."""

    def __init__(
        self,
        *,
        observation_modules: Mapping[str, ObservationModule],
        dynamics: nn.Module,
        associator: Associator,
        updater: BeliefUpdater,
        lifecycle: ObjectLifecycle,
        identifier: RecurrentParameterUpdater | None,
        observability: ObservabilityEstimator,
        scheduler: ObservationScheduler,
        belief_factory: BeliefFactory,
        surprise_classifier: SurpriseClassifier | None = None,
        modality_order: Sequence[str] = ("debug_oracle", "rgb"),
        strict_timestamps: bool = True,
        allow_debug_oracle: bool = False,
        birth_confidence: float = 0.55,
        initial_velocity_variance: float = 1.0,
        gravity: Sequence[float] = (0.0, -9.81, 0.0),
    ) -> None:
        super().__init__()
        validate_module_mapping(observation_modules)
        self.observation_modules = nn.ModuleDict(dict(observation_modules))
        self.dynamics = dynamics
        self.associator = associator
        self.updater = updater
        self.lifecycle = lifecycle
        self.identifier = identifier
        self.observability = observability
        self.scheduler = scheduler
        self.belief_factory = belief_factory
        self.surprise_classifier = surprise_classifier or SurpriseClassifier()
        self.modality_order = tuple(modality_order)
        self.strict_timestamps = strict_timestamps
        self.allow_debug_oracle = allow_debug_oracle
        self.birth_confidence = birth_confidence
        self.initial_velocity_variance = initial_velocity_variance
        self.gravity = tuple(float(value) for value in gravity)
        self.state = RuntimeState()
        self.diagnostics = RuntimeDiagnostics()
        self._last_measurements: MeasurementSet | None = None
        self._prepared_propagation_owner = object()
        self.abstraction_router = PredictiveAbstractionRouter()
        self.belief_tokenizer = WorldBeliefTokenizer(self.abstraction_router)

    @classmethod
    def from_config(
        cls,
        config: OrpheusConfig,
        *,
        device: torch.device | str | None = None,
    ) -> OnlineWorldModel:
        """Build the complete RGB/oracle-debug vertical path from typed config."""

        from world_model.belief import NUM_MOTION_MODES
        from world_model.dynamics import DynamicsModel
        from world_model.observations.rgb import (
            RGBObservationConfig,
            RGBObservationModule,
        )
        from world_model.observations.state import (
            StateObservationConfig,
            StateObservationModule,
        )
        from world_model.utils.config import OrpheusConfig

        if not isinstance(config, OrpheusConfig):
            raise TypeError("OnlineWorldModel.from_config expects OrpheusConfig")
        config.validate()
        state_config = config.model.state
        factory = BeliefFactory(
            max_objects=config.model.max_objects,
            geometry_dim=state_config.geometry_dim,
            appearance_dim=state_config.appearance_dim,
            residual_dynamics_dim=state_config.residual_dynamics_dim,
            modal_count=state_config.modal_count,
            modal_dim=state_config.modal_dim,
            parameter_memory_dim=state_config.parameter_memory_dim,
            global_code_dim=state_config.global_dim,
        )
        dynamics = DynamicsModel.from_config(config)
        modules: dict[str, ObservationModule] = {}
        rgb_config = config.model.rgb
        if rgb_config.enabled:
            query_count = max(
                rgb_config.proposal_queries,
                config.model.max_objects,
            )
            modules["rgb"] = RGBObservationModule(
                RGBObservationConfig(
                    max_objects=config.model.max_objects,
                    birth_extra_queries=query_count - config.model.max_objects,
                    backbone_channels=tuple(rgb_config.backbone_channels),
                    feature_dim=rgb_config.feature_dim,
                    appearance_dim=state_config.appearance_dim,
                    global_detector_cpu_on_mps=(config.device.global_detector_cpu_on_mps),
                    roi_size=rgb_config.roi_size,
                    roi_hidden_dim=config.model.filter.hidden_dim,
                    fast_depth_residual_enabled=(rgb_config.fast_depth_residual_enabled),
                    temporal_velocity_enabled=rgb_config.temporal_velocity_enabled,
                    temporal_velocity_history_size=(rgb_config.temporal_velocity_history_size),
                    temporal_velocity_min_samples=(rgb_config.temporal_velocity_min_samples),
                    temporal_velocity_min_dt=rgb_config.temporal_velocity_min_dt,
                    temporal_velocity_variance_scale=(rgb_config.temporal_velocity_variance_scale),
                    temporal_velocity_variance_floor=(rgb_config.temporal_velocity_variance_floor),
                    temporal_velocity_variance_ceiling=(
                        rgb_config.temporal_velocity_variance_ceiling
                    ),
                    temporal_velocity_lateral_only=(rgb_config.temporal_velocity_lateral_only),
                    temporal_velocity_post_event_gravity_axis_enabled=(
                        rgb_config.temporal_velocity_post_event_gravity_axis_enabled
                    ),
                    temporal_velocity_unobserved_variance=(
                        rgb_config.temporal_velocity_unobserved_variance
                    ),
                    temporal_velocity_reset_on_collision=(
                        rgb_config.temporal_velocity_reset_on_collision
                    ),
                    temporal_velocity_max_age_steps=(rgb_config.temporal_velocity_max_age_steps),
                    temporal_velocity_post_event_max_samples=(
                        rgb_config.temporal_velocity_post_event_max_samples
                    ),
                    temporal_velocity_post_event_min_samples=(
                        rgb_config.temporal_velocity_post_event_min_samples
                    ),
                    temporal_velocity_change_point_enabled=(
                        rgb_config.temporal_velocity_change_point_enabled
                    ),
                    temporal_velocity_change_point_minimum_speed=(
                        rgb_config.temporal_velocity_change_point_minimum_speed
                    ),
                    temporal_velocity_change_point_minimum_delta=(
                        rgb_config.temporal_velocity_change_point_minimum_delta
                    ),
                    temporal_velocity_change_point_strong_delta=(
                        rgb_config.temporal_velocity_change_point_strong_delta
                    ),
                    temporal_velocity_change_point_require_contact_mode=(
                        rgb_config.temporal_velocity_change_point_require_contact_mode
                    ),
                    temporal_velocity_change_point_gate=(
                        rgb_config.temporal_velocity_change_point_gate
                    ),
                    temporal_velocity_change_point_linear_weights=(
                        rgb_config.temporal_velocity_change_point_linear_weights
                    ),
                    temporal_velocity_change_point_linear_bias=(
                        rgb_config.temporal_velocity_change_point_linear_bias
                    ),
                    temporal_velocity_change_point_mlp_hidden_weights=(
                        rgb_config.temporal_velocity_change_point_mlp_hidden_weights
                    ),
                    temporal_velocity_change_point_mlp_hidden_bias=(
                        rgb_config.temporal_velocity_change_point_mlp_hidden_bias
                    ),
                    temporal_velocity_change_point_mlp_output_weights=(
                        rgb_config.temporal_velocity_change_point_mlp_output_weights
                    ),
                    temporal_velocity_change_point_mlp_output_bias=(
                        rgb_config.temporal_velocity_change_point_mlp_output_bias
                    ),
                    temporal_velocity_change_point_probability_threshold=(
                        rgb_config.temporal_velocity_change_point_probability_threshold
                    ),
                    temporal_velocity_change_point_minimum_interval_samples=(
                        rgb_config.temporal_velocity_change_point_minimum_interval_samples
                    ),
                    temporal_velocity_outgoing_proposal_enabled=(
                        rgb_config.temporal_velocity_outgoing_proposal_enabled
                    ),
                    temporal_velocity_outgoing_proposal_hidden_weights=(
                        rgb_config.temporal_velocity_outgoing_proposal_hidden_weights
                    ),
                    temporal_velocity_outgoing_proposal_hidden_bias=(
                        rgb_config.temporal_velocity_outgoing_proposal_hidden_bias
                    ),
                    temporal_velocity_outgoing_proposal_output_weights=(
                        rgb_config.temporal_velocity_outgoing_proposal_output_weights
                    ),
                    temporal_velocity_outgoing_proposal_output_bias=(
                        rgb_config.temporal_velocity_outgoing_proposal_output_bias
                    ),
                    temporal_velocity_outgoing_proposal_variance=(
                        rgb_config.temporal_velocity_outgoing_proposal_variance
                    ),
                    temporal_velocity_outgoing_proposal_maximum_delta=(
                        rgb_config.temporal_velocity_outgoing_proposal_maximum_delta
                    ),
                    temporal_velocity_lateral_intervention_enabled=(
                        rgb_config.temporal_velocity_lateral_intervention_enabled
                    ),
                    temporal_velocity_lateral_intervention_hidden_weights=(
                        rgb_config.temporal_velocity_lateral_intervention_hidden_weights
                    ),
                    temporal_velocity_lateral_intervention_hidden_bias=(
                        rgb_config.temporal_velocity_lateral_intervention_hidden_bias
                    ),
                    temporal_velocity_lateral_intervention_output_weights=(
                        rgb_config.temporal_velocity_lateral_intervention_output_weights
                    ),
                    temporal_velocity_lateral_intervention_output_bias=(
                        rgb_config.temporal_velocity_lateral_intervention_output_bias
                    ),
                    temporal_velocity_lateral_intervention_variance_floor=(
                        rgb_config.temporal_velocity_lateral_intervention_variance_floor
                    ),
                    temporal_velocity_lateral_intervention_variance_ceiling=(
                        rgb_config.temporal_velocity_lateral_intervention_variance_ceiling
                    ),
                    temporal_velocity_lateral_intervention_gain_power=(
                        rgb_config.temporal_velocity_lateral_intervention_gain_power
                    ),
                    temporal_velocity_lateral_intervention_maximum_delta=(
                        rgb_config.temporal_velocity_lateral_intervention_maximum_delta
                    ),
                    temporal_velocity_gravity_intervention_enabled=(
                        rgb_config.temporal_velocity_gravity_intervention_enabled
                    ),
                    temporal_velocity_gravity_intervention_hidden_weights=(
                        rgb_config.temporal_velocity_gravity_intervention_hidden_weights
                    ),
                    temporal_velocity_gravity_intervention_hidden_bias=(
                        rgb_config.temporal_velocity_gravity_intervention_hidden_bias
                    ),
                    temporal_velocity_gravity_intervention_output_weights=(
                        rgb_config.temporal_velocity_gravity_intervention_output_weights
                    ),
                    temporal_velocity_gravity_intervention_output_bias=(
                        rgb_config.temporal_velocity_gravity_intervention_output_bias
                    ),
                    temporal_velocity_gravity_intervention_variance_floor=(
                        rgb_config.temporal_velocity_gravity_intervention_variance_floor
                    ),
                    temporal_velocity_gravity_intervention_variance_ceiling=(
                        rgb_config.temporal_velocity_gravity_intervention_variance_ceiling
                    ),
                    temporal_velocity_gravity_intervention_gain_power=(
                        rgb_config.temporal_velocity_gravity_intervention_gain_power
                    ),
                    temporal_velocity_gravity_intervention_maximum_delta=(
                        rgb_config.temporal_velocity_gravity_intervention_maximum_delta
                    ),
                    temporal_velocity_measurement_position_blend=(
                        rgb_config.temporal_velocity_measurement_position_blend
                    ),
                    temporal_velocity_position_innovation_coupling=(
                        rgb_config.temporal_velocity_position_innovation_coupling
                    ),
                    temporal_position_enabled=rgb_config.temporal_position_enabled,
                    temporal_position_min_samples=rgb_config.temporal_position_min_samples,
                    temporal_position_robust_threshold=(
                        rgb_config.temporal_position_robust_threshold
                    ),
                    temporal_position_variance_scale=(rgb_config.temporal_position_variance_scale),
                    temporal_position_variance_floor=(rgb_config.temporal_position_variance_floor),
                    temporal_position_variance_ceiling=(
                        rgb_config.temporal_position_variance_ceiling
                    ),
                    temporal_position_depth_only=rgb_config.temporal_position_depth_only,
                    structured_disc_center_enabled=(rgb_config.structured_disc_center_enabled),
                    structured_disc_threshold=rgb_config.structured_disc_threshold,
                    structured_disc_min_pixels=rgb_config.structured_disc_min_pixels,
                    structured_disc_max_assignment_distance=(
                        rgb_config.structured_disc_max_assignment_distance
                    ),
                    structured_disc_center_std_pixels=(
                        rgb_config.structured_disc_center_std_pixels
                    ),
                    structured_disc_fast_depth_enabled=(
                        rgb_config.structured_disc_fast_depth_enabled
                    ),
                    structured_disc_depth_relative_std=(
                        rgb_config.structured_disc_depth_relative_std
                    ),
                    structured_disc_depth_outlier_relative_threshold=(
                        rgb_config.structured_disc_depth_outlier_relative_threshold
                    ),
                    structured_disc_depth_outlier_variance_scale=(
                        rgb_config.structured_disc_depth_outlier_variance_scale
                    ),
                    structured_disc_position_confidence=(
                        rgb_config.structured_disc_position_confidence
                    ),
                    roi_uncertainty_scale=rgb_config.roi_uncertainty_scale,
                    default_world_radius=(sum(config.simulator.radius_range) / 2.0),
                    proposal_threshold=rgb_config.existence_threshold,
                    measurement_log_variance_min=(rgb_config.measurement_log_variance_min),
                    measurement_log_variance_max=(rgb_config.measurement_log_variance_max),
                )
            )
        debug_allowed = (
            config.runtime.enable_debug_oracle or config.runtime.modality == "debug_oracle"
        )
        if debug_allowed:
            modules["debug_oracle"] = StateObservationModule(
                StateObservationConfig(appearance_dim=state_config.appearance_dim)
            )
        if config.runtime.modality not in modules:
            raise ValueError(
                f"configured runtime modality {config.runtime.modality!r} is not enabled"
            )
        association_config = config.model.association
        associator = Associator(
            geometry_weight=association_config.geometry_weight,
            appearance_weight=association_config.appearance_weight,
            existence_weight=association_config.existence_weight,
            mahalanobis_gate=association_config.mahalanobis_gate,
            maximum_cost=association_config.maximum_cost,
            ambiguity_margin=association_config.ambiguity_margin,
            minimum_measurement_confidence=(association_config.minimum_measurement_confidence),
        )
        filter_config = config.model.filter
        updater = BeliefUpdater(
            fast_state_dim=factory.fast_state_dim,
            num_motion_modes=NUM_MOTION_MODES,
            hidden_dim=filter_config.hidden_dim,
            config=BeliefUpdaterConfig(
                robust_clip_norm=filter_config.robust_clip,
                minimum_log_variance=filter_config.min_log_variance,
                maximum_log_variance=filter_config.max_log_variance,
                learned_residual_scale=filter_config.learned_residual_scale,
                innovation_anchored_correction=(filter_config.innovation_anchored_correction),
                missed_fast_variance_increment=(filter_config.missed_variance_growth),
                observed_confidence_threshold=(association_config.minimum_measurement_confidence),
            ),
        )
        lifecycle_config = config.model.lifecycle
        lifecycle = ObjectLifecycle(
            LifecycleConfig(
                max_missed_steps=lifecycle_config.max_missed_steps,
                max_occluded_steps=lifecycle_config.max_occluded_steps,
                missed_existence_delta=-lifecycle_config.existence_decay,
                occluded_existence_delta=(-lifecycle_config.occlusion_existence_decay),
                initial_radius=factory.initial_radius,
                initial_mass=factory.initial_mass,
                initial_restitution=factory.initial_restitution,
                initial_drag=factory.initial_drag,
                initial_friction=factory.initial_friction,
                initial_log_variance=factory.initial_log_variance,
                birth_confirmations=lifecycle_config.birth_confirmations,
                birth_confirmation_distance_m=(lifecycle_config.birth_confirmation_distance_m),
            )
        )
        identification_config = config.model.identification
        identifier = (
            RecurrentParameterUpdater(
                ParameterUpdaterConfig(
                    hidden_dim=identification_config.hidden_dim,
                    slow_learning_rate=(identification_config.slow_learning_rate),
                )
            )
            if identification_config.enabled
            else None
        )
        if (
            identifier is not None
            and state_config.parameter_memory_dim != identification_config.hidden_dim
        ):
            raise ValueError("state.parameter_memory_dim must equal identification.hidden_dim")
        scheduler = ObservationScheduler(
            global_every_steps=rgb_config.global_every_steps,
            uncertainty_threshold=rgb_config.global_uncertainty_threshold,
            surprise_threshold=rgb_config.surprise_threshold,
        )
        model = cls(
            observation_modules=modules,
            dynamics=dynamics,
            associator=associator,
            updater=updater,
            lifecycle=lifecycle,
            identifier=identifier,
            observability=ObservabilityEstimator(
                ObservabilityConfig(minimum_drag_speed=(identification_config.drag_speed_threshold))
            ),
            scheduler=scheduler,
            belief_factory=factory,
            modality_order=config.runtime.modality_order,
            strict_timestamps=config.runtime.strict_timestamps,
            allow_debug_oracle=debug_allowed,
            birth_confidence=lifecycle_config.birth_confidence,
            gravity=config.simulator.gravity,
        )
        if device is not None:
            model = model.to(device)
        return model

    @property
    def belief(self) -> WorldBelief | None:
        return self.state.belief

    @property
    def caches(self) -> dict[str, object]:
        return self.state.caches

    @property
    def last_measurements(self) -> MeasurementSet | None:
        """Detached measurements from the most recent scheduled observation."""

        return self._last_measurements

    def predictive_abstractions(self) -> AbstractionAssignment:
        """Return the current derived executable abstraction per entity.

        ``WorldBelief`` remains authoritative; this method never caches a
        second copy of physical state.
        """

        if self.state.belief is None:
            raise RuntimeError("OnlineWorldModel must ingest an observation first")
        return self.abstraction_router.route(self.state.belief)

    def predictive_tokens(self) -> PredictiveTokenBatch:
        """Return a reversible LLM-style typed token view of the current belief."""

        if self.state.belief is None:
            raise RuntimeError("OnlineWorldModel must ingest an observation first")
        assignment = self.predictive_abstractions()
        return self.belief_tokenizer.encode(self.state.belief, assignment)

    def reset(self, batch_size: int = 1) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.state = RuntimeState(batch_size=batch_size)
        self.scheduler.reset()
        self.diagnostics.reset()
        self._last_measurements = None
        if self.identifier is not None:
            self.identifier.last_diagnostics = None

    def _model_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        parameter = next(self.parameters(), None)
        if parameter is None:
            return torch.device("cpu"), torch.float32
        return parameter.device, parameter.dtype

    def _dynamics_tensor_signature(self) -> TensorVersionSignature:
        return tensor_identity_version_signature(
            {
                "parameters": dict(self.dynamics.named_parameters()),
                "buffers": dict(self.dynamics.named_buffers()),
            }
        )

    @staticmethod
    def _requested_timestamp(
        current: WorldBelief,
        timestamp: float | Tensor,
    ) -> Tensor:
        requested = torch.as_tensor(
            timestamp,
            device=current.device,
            dtype=current.dtype,
        )
        if requested.ndim == 0:
            requested = requested.expand_as(current.timestamp).clone()
        if requested.shape != current.timestamp.shape:
            raise ValueError(
                "observation timestamp must be scalar or match belief batch "
                f"{tuple(current.timestamp.shape)}"
            )
        if not bool(torch.isfinite(requested).all()):
            raise ValueError("observation timestamp must be finite")
        if torch.any(requested < current.timestamp):
            raise OutOfSequenceObservationError(
                f"observation timestamp {float(requested.min()):.9g} precedes current "
                f"belief timestamp {float(current.timestamp.min()):.9g}"
            )
        return requested

    @staticmethod
    def _interval_collision_mask(
        current: WorldBelief,
        auxiliary: Mapping[str, Tensor],
    ) -> Tensor | None:
        pair_collision = auxiliary.get("pair_collision")
        boundary_collision = auxiliary.get("boundary_collision")
        ground_collision = auxiliary.get("ground_collision")
        if pair_collision is None or (boundary_collision is None and ground_collision is None):
            return None
        expected_pair = (
            current.batch_size,
            current.objects.max_objects,
            current.objects.max_objects,
        )
        if pair_collision.shape != expected_pair or pair_collision.dtype is not torch.bool:
            raise ValueError("dynamics pair_collision must be boolean [B,N,N]")
        if boundary_collision is not None:
            if (
                boundary_collision.ndim != 3
                or boundary_collision.shape[:2] != current.objects.active.shape
                or boundary_collision.dtype is not torch.bool
            ):
                raise ValueError("dynamics boundary_collision must be boolean [B,N,P]")
            boundary_collision_mask = boundary_collision.any(dim=-1)
        else:
            assert ground_collision is not None
            if (
                ground_collision.shape != current.objects.active.shape
                or ground_collision.dtype is not torch.bool
            ):
                raise ValueError("dynamics ground_collision must be boolean [B,N]")
            boundary_collision_mask = ground_collision
        return pair_collision.any(dim=-1) | boundary_collision_mask

    def prepare_propagation(
        self,
        target_timestamp: float | Tensor,
    ) -> PreparedPropagation:
        """Predict once from the current runtime revision for one future ingest."""

        current = self.state.belief
        if current is None:
            raise RuntimeError("cannot prepare propagation before belief initialization")
        requested = self._requested_timestamp(current, target_timestamp)
        if not torch.equal(requested, requested[:1].expand_as(requested)):
            raise PreparedPropagationError(
                "prepared propagation target timestamp must be uniform across batch"
            )
        delta_time = requested - current.timestamp
        source_tensor_signature = tensor_identity_version_signature(current)
        predict_step = getattr(self.dynamics, "predict_step", None)
        if not callable(predict_step):
            raise TypeError("prepared propagation requires dynamics.predict_step")
        propagation = predict_step(current, delta_time)
        if tensor_identity_version_signature(current) != source_tensor_signature:
            raise PreparedPropagationError(
                "dynamics.predict_step mutated the prepared propagation source belief"
            )
        prior = getattr(propagation, "belief", None)
        event_logits = getattr(propagation, "event_logits", None)
        raw_auxiliary = getattr(propagation, "auxiliary", None)
        if not isinstance(prior, WorldBelief):
            raise TypeError("dynamics.predict_step must return a WorldBelief endpoint")
        if event_logits is not None and not isinstance(event_logits, Tensor):
            raise TypeError("dynamics.predict_step event_logits must be a tensor or null")
        if not isinstance(raw_auxiliary, Mapping) or any(
            not isinstance(name, str) or not isinstance(value, Tensor)
            for name, value in raw_auxiliary.items()
        ):
            raise TypeError("dynamics.predict_step auxiliary must map strings to tensors")
        auxiliary = dict(raw_auxiliary)
        if (
            prior.batch_size != current.batch_size
            or prior.device != current.device
            or prior.dtype != current.dtype
        ):
            raise ValueError("prepared prior must preserve source batch, device, and dtype")
        if not torch.equal(prior.timestamp, requested):
            raise ValueError("prepared prior timestamp must equal the requested target")
        interval_collision_mask = self._interval_collision_mask(current, auxiliary)
        model_device, model_dtype = self._model_device_dtype()
        if model_device != current.device or model_dtype != current.dtype:
            raise ValueError("runtime belief and model execution device/dtype do not match")
        source_timestamp = current.timestamp.detach().clone()
        target_timestamp = requested.detach().clone()
        result_tensor_signature = tensor_identity_version_signature(
            {
                "prior": prior,
                "source_timestamp": source_timestamp,
                "target_timestamp": target_timestamp,
                "delta_time": delta_time,
                "event_logits": event_logits,
                "auxiliary": auxiliary,
                "interval_collision_mask": interval_collision_mask,
            }
        )
        dynamics_tensor_signature = self._dynamics_tensor_signature()
        return PreparedPropagation(
            source=current,
            prior=prior,
            source_revision=self.state.ingest_count,
            source_timestamp=source_timestamp,
            target_timestamp=target_timestamp,
            delta_time=delta_time,
            source_device=current.device,
            source_dtype=current.dtype,
            source_batch_size=current.batch_size,
            event_logits=event_logits,
            auxiliary=auxiliary,
            interval_collision_mask=interval_collision_mask,
            source_tensor_signature=source_tensor_signature,
            result_tensor_signature=result_tensor_signature,
            dynamics_tensor_signature=dynamics_tensor_signature,
            dynamics_training=self.dynamics.training,
            _owner_token=self._prepared_propagation_owner,
        )

    def _validate_prepared_propagation(
        self,
        prepared: PreparedPropagation,
        *,
        current: WorldBelief,
        requested: Tensor,
        packet_batch_size: int,
    ) -> None:
        if not isinstance(prepared, PreparedPropagation):
            raise TypeError("prepared must be a PreparedPropagation")
        if prepared.consumed:
            raise PreparedPropagationError("prepared propagation has already been consumed")
        if prepared._owner_token is not self._prepared_propagation_owner:
            raise PreparedPropagationError("prepared propagation belongs to another runtime")
        if prepared.source_revision != self.state.ingest_count:
            raise PreparedPropagationError(
                "prepared propagation source revision is stale "
                f"({prepared.source_revision} != {self.state.ingest_count})"
            )
        if prepared.source is not current:
            raise PreparedPropagationError("prepared propagation source belief is stale")
        if prepared.dynamics_training != self.dynamics.training:
            raise PreparedPropagationError(
                "prepared propagation dynamics training/evaluation mode has changed"
            )
        if self._dynamics_tensor_signature() != prepared.dynamics_tensor_signature:
            raise PreparedPropagationError(
                "prepared propagation dynamics parameters or buffers have changed"
            )
        if (
            prepared.source_batch_size != current.batch_size
            or packet_batch_size != current.batch_size
            or self.state.batch_size != current.batch_size
        ):
            raise PreparedPropagationError(
                "prepared propagation batch does not match runtime and observation"
            )
        model_device, model_dtype = self._model_device_dtype()
        if (
            prepared.source_device != current.device
            or prepared.source_device != model_device
            or prepared.source_dtype != current.dtype
            or prepared.source_dtype != model_dtype
        ):
            raise PreparedPropagationError(
                "prepared propagation source device/dtype no longer matches runtime"
            )
        if not torch.equal(current.timestamp, prepared.source_timestamp):
            raise PreparedPropagationError("prepared propagation source timestamp has changed")
        if tensor_identity_version_signature(current) != prepared.source_tensor_signature:
            raise PreparedPropagationError(
                "prepared propagation source belief tensors have changed"
            )
        if not torch.equal(requested, prepared.target_timestamp):
            raise PreparedPropagationError(
                "prepared propagation target timestamp does not match observation"
            )
        expected_delta = requested - current.timestamp
        if (
            prepared.delta_time.shape != expected_delta.shape
            or prepared.delta_time.device != expected_delta.device
            or prepared.delta_time.dtype != expected_delta.dtype
            or not torch.equal(prepared.delta_time, expected_delta)
        ):
            raise PreparedPropagationError(
                "prepared propagation temporal delta does not match source and target"
            )
        if (
            prepared.prior.batch_size != current.batch_size
            or prepared.prior.device != current.device
            or prepared.prior.dtype != current.dtype
            or not torch.equal(prepared.prior.timestamp, requested)
        ):
            raise PreparedPropagationError(
                "prepared propagation prior no longer matches target execution context"
            )
        result_tensor_signature = tensor_identity_version_signature(
            {
                "prior": prepared.prior,
                "source_timestamp": prepared.source_timestamp,
                "target_timestamp": prepared.target_timestamp,
                "delta_time": prepared.delta_time,
                "event_logits": prepared.event_logits,
                "auxiliary": prepared.auxiliary,
                "interval_collision_mask": prepared.interval_collision_mask,
            }
        )
        if result_tensor_signature != prepared.result_tensor_signature:
            raise PreparedPropagationError("prepared propagation result tensors have changed")
        recomputed_collision_mask = self._interval_collision_mask(
            current,
            prepared.auxiliary,
        )
        if (recomputed_collision_mask is None) != (prepared.interval_collision_mask is None) or (
            recomputed_collision_mask is not None
            and prepared.interval_collision_mask is not None
            and not torch.equal(
                recomputed_collision_mask,
                prepared.interval_collision_mask,
            )
        ):
            raise PreparedPropagationError(
                "prepared propagation interval collision evidence has changed"
            )

    def _initial_belief(self, packet: ObservationPacket) -> WorldBelief:
        device, dtype = self._model_device_dtype()
        batch_size = _packet_batch_size(packet)
        belief = self.belief_factory.create(
            batch_size,
            timestamp=packet.timestamp,
            device=device,
            dtype=dtype,
            gravity=self.gravity,
            active_modalities=tuple(self.observation_modules.keys()),
        )
        return self._update_known_sensor_state(belief, packet)

    @staticmethod
    def _image_size(packet: ObservationPacket) -> tuple[int, int] | None:
        metadata_size = packet.metadata.get("image_size")
        if isinstance(metadata_size, (tuple, list)) and len(metadata_size) == 2:
            return int(metadata_size[0]), int(metadata_size[1])
        payload = packet.payload
        if isinstance(payload, Tensor) and payload.ndim >= 3:
            return int(payload.shape[-2]), int(payload.shape[-1])
        return None

    @staticmethod
    def _update_known_sensor_state(
        belief: WorldBelief,
        packet: ObservationPacket,
    ) -> WorldBelief:
        camera = belief.camera
        updates: dict[str, Tensor] = {}
        for key in ("world_from_camera", "intrinsics"):
            value = packet.calibration.get(key)
            if not isinstance(value, Tensor):
                continue
            tensor = value.to(device=belief.device, dtype=belief.dtype)
            expected_tail = (4, 4) if key == "world_from_camera" else (3, 3)
            if tensor.shape == expected_tail:
                tensor = tensor.unsqueeze(0).expand(belief.batch_size, -1, -1).clone()
            if tensor.shape != (belief.batch_size, *expected_tail):
                raise ValueError(f"calibration {key} has incompatible batch shape")
            updates[key] = tensor
        if not updates:
            return belief
        return belief.replace(camera=camera.replace(**updates))

    def _order_group(
        self,
        packets: Sequence[ObservationPacket],
    ) -> list[ObservationPacket]:
        order = {modality: index for index, modality in enumerate(self.modality_order)}
        return sorted(
            packets,
            key=lambda packet: (
                order.get(packet.modality, len(order)),
                packet.modality,
                packet.sensor_id,
            ),
        )

    @staticmethod
    def _cache_matches_belief(
        cache: object,
        belief: WorldBelief,
    ) -> bool:
        """Check optional object-indexed cache identity against persistent slots."""

        cached_object_ids = getattr(cache, "object_ids", None)
        if cached_object_ids is None:
            return True
        if not isinstance(cached_object_ids, Tensor):
            return False
        belief_object_ids = belief.objects.object_id
        return cached_object_ids.shape == belief_object_ids.shape and torch.equal(
            cached_object_ids.to(device=belief_object_ids.device),
            belief_object_ids,
        )

    def _ingest_one(
        self,
        packet: ObservationPacket,
        posterior: WorldBelief,
        prediction_dt: Tensor,
        prior_interval_collision_mask: Tensor | None = None,
    ) -> WorldBelief:
        started = time.perf_counter()
        if self.identifier is not None:
            # Diagnostics describe this packet only.  A propagation-only
            # scheduler decision must not expose the preceding update again.
            self.identifier.last_diagnostics = None
        if packet.modality == "debug_oracle" and not self.allow_debug_oracle:
            raise ValueError(
                "debug_oracle input is disabled; RGB runtime cannot consume "
                "privileged simulator state"
            )
        try:
            module = self.observation_modules[packet.modality]
        except KeyError as exc:
            available = ", ".join(self.observation_modules.keys())
            raise KeyError(
                f"unsupported modality {packet.modality!r}; enabled: {available}"
            ) from exc
        module.validate_packet(packet)
        posterior = self._update_known_sensor_state(posterior, packet)
        sensor_context = SensorContext(
            sensor_id=packet.sensor_id,
            timestamp=packet.timestamp,
            calibration=packet.calibration,
            frame_id=packet.frame_id,
            image_size=self._image_size(packet),
            metadata=packet.metadata,
        )
        predicted = module.project(posterior, sensor_context)
        predicted.validate()
        if self.strict_timestamps and not torch.equal(
            predicted.timestamp,
            posterior.timestamp,
        ):
            raise ValueError(
                "predicted measurement timestamp must equal the propagated belief timestamp"
            )
        scheduler_context = self.diagnostics.scheduler_context(packet.sensor_id)
        mode = self.scheduler.choose(
            packet=packet,
            belief=posterior,
            predicted=predicted,
            diagnostics=scheduler_context,
        )
        observation_context = ObservationContext(
            timestamp=packet.timestamp,
            calibration=packet.calibration,
            frame_id=packet.frame_id,
            max_objects=self.belief_factory.max_objects,
            device=posterior.device,
            dtype=posterior.dtype,
            training=self.training,
            predicted_regions=predicted.rois,
            metadata=packet.metadata,
        )
        cache = self.state.caches.get(packet.sensor_id)
        if cache is not None and not self._cache_matches_belief(cache, posterior):
            self.state.caches.pop(packet.sensor_id, None)
            cache = None
        if mode in {ObservationMode.GLOBAL_DISCOVERY, ObservationMode.RECOVERY}:
            # A global pass invalidates object-indexed ROI features even when it
            # was cadence-triggered rather than surprise-triggered.
            self.state.caches.pop(packet.sensor_id, None)
            measurements = module.initialise_measurements([packet], observation_context)
        elif mode == ObservationMode.FAST_ROI:
            measurements, new_cache = module.encode_measurements(
                [packet], posterior, predicted, cache
            )
            self.state.caches[packet.sensor_id] = new_cache
        else:
            return posterior
        measurements.validate()
        expected_measurement_timestamp = posterior.timestamp.new_full(
            posterior.timestamp.shape,
            packet.timestamp,
        )
        if self.strict_timestamps and not torch.equal(
            measurements.timestamp,
            expected_measurement_timestamp,
        ):
            raise ValueError("measurement timestamp must equal its observation packet timestamp")
        if prior_interval_collision_mask is not None:
            expected = posterior.objects.active.shape
            if (
                prior_interval_collision_mask.shape != expected
                or prior_interval_collision_mask.dtype is not torch.bool
            ):
                raise ValueError("prior interval collision mask must be boolean belief-slot [B,N]")
            measurements = replace(
                measurements,
                auxiliary={
                    **measurements.auxiliary,
                    "prior_interval_collision_mask": (prior_interval_collision_mask),
                },
            )
        active_before = int(posterior.objects.active.sum().detach().cpu())
        association = self.associator.match(posterior, measurements, predicted)
        innovation = module.innovation(measurements, predicted, association)
        surprise = self.surprise_classifier(innovation, association)
        posterior = self.updater.correct(
            prior=posterior,
            measured=measurements,
            predicted=predicted,
            association=association,
            innovation=innovation,
            dt=prediction_dt,
            cause=surprise,
        )
        velocity_evidence, temporal_history = module.update_temporal_history(
            posterior=posterior,
            measured=measurements,
            association=association,
            history=self.state.temporal_histories.get(packet.sensor_id),
        )
        if temporal_history is None:
            self.state.temporal_histories.pop(packet.sensor_id, None)
        else:
            self.state.temporal_histories[packet.sensor_id] = temporal_history
        if velocity_evidence is not None:
            posterior = self.updater.correct_direct_velocity(
                posterior,
                velocity_evidence,
            )
        self._last_measurements = measurements.detach()
        observed_mask = (
            self.updater.last_diagnostics.observed_mask
            if self.updater.last_diagnostics is not None
            else torch.zeros_like(posterior.objects.active)
        )
        predicted_occluded = predicted.auxiliary.get("unobservable_mask")
        if predicted_occluded is None:
            predicted_occluded = predicted.auxiliary.get("fully_occluded_mask")
        if predicted_occluded is None:
            predicted_occluded = torch.zeros_like(observed_mask)
        posterior = self.lifecycle.update_visibility(
            posterior,
            observed_mask,
            occluded_mask=predicted_occluded,
        )
        tentative_birth_candidates = 0
        confirmed_births = 0
        if mode in {ObservationMode.GLOBAL_DISCOVERY, ObservationMode.RECOVERY}:
            tentative_key = (packet.modality, packet.sensor_id)
            confirmed_measurements, tentative_state = self.lifecycle.confirm_tentative_births(
                measurements,
                association.unmatched_measurements,
                self.state.tentative_births.get(tentative_key),
                confidence_threshold=self.birth_confidence,
            )
            tentative_birth_candidates = int(tentative_state.active.sum().detach().cpu())
            if tentative_birth_candidates:
                self.state.tentative_births[tentative_key] = tentative_state
            else:
                self.state.tentative_births.pop(tentative_key, None)
            active_before_birth = int(posterior.objects.active.sum().detach().cpu())
            posterior = self.lifecycle.birth_from_measurements(
                posterior,
                measurements,
                confirmed_measurements,
                confidence_threshold=self.birth_confidence,
                initial_velocity_variance=self.initial_velocity_variance,
            )
            confirmed_births = max(
                0,
                int(posterior.objects.active.sum().detach().cpu()) - active_before_birth,
            )
        observable = self.observability(
            posterior,
            innovation,
            association,
            surprise,
            interval_collision_mask=prior_interval_collision_mask,
        )
        if self.identifier is not None:
            posterior = self.identifier.update(
                posterior,
                innovation,
                association,
                observable,
            )
        cached_after_update = self.state.caches.get(packet.sensor_id)
        if cached_after_update is not None and not self._cache_matches_belief(
            cached_after_update,
            posterior,
        ):
            self.state.caches.pop(packet.sensor_id, None)
        aggregate_surprise = float(surprise.aggregate_surprise.mean().detach().cpu())
        unmatched_count = int(association.unmatched_measurements.sum().detach().cpu())
        failures = int(active_before > 0 and int(association.pair_mask.sum().detach().cpu()) == 0)
        self.scheduler.record(
            packet.sensor_id,
            mode,
            surprise=aggregate_surprise,
            association_failures=failures,
        )
        correction_norm = (
            float(self.updater.last_diagnostics.correction_norm.mean().detach().cpu())
            if self.updater.last_diagnostics is not None
            and self.updater.last_diagnostics.correction_norm.numel() > 0
            else 0.0
        )
        self.diagnostics.record(
            RuntimeStepDiagnostics(
                timestamp=packet.timestamp,
                modality=packet.modality,
                sensor_id=packet.sensor_id,
                observation_mode=mode.value,
                active_objects_before=active_before,
                active_objects_after=int(posterior.objects.active.sum().detach().cpu()),
                matched_pairs=int(association.pair_mask.sum().detach().cpu()),
                unmatched_measurements=unmatched_count,
                tentative_birth_candidates=tentative_birth_candidates,
                confirmed_births=confirmed_births,
                ambiguous_pairs=int(association.ambiguous.sum().detach().cpu()),
                aggregate_surprise=aggregate_surprise,
                correction_norm=correction_norm,
                elapsed_milliseconds=(time.perf_counter() - started) * 1000.0,
            )
        )
        return posterior

    def ingest(
        self,
        packets: ObservationPacket | Sequence[ObservationPacket],
        *,
        prepared: PreparedPropagation | None = None,
    ) -> WorldBelief:
        packet_list = [packets] if isinstance(packets, ObservationPacket) else list(packets)
        if not packet_list:
            raise ValueError("ingest requires at least one observation packet")
        if prepared is not None and len({packet.timestamp for packet in packet_list}) != 1:
            raise PreparedPropagationError(
                "prepared propagation requires one observation timestamp group"
            )
        device, dtype = self._model_device_dtype()
        packet_list = [_move_packet(packet, device=device, dtype=dtype) for packet in packet_list]
        packet_list.sort(key=lambda packet: packet.timestamp)
        for timestamp, iterator in groupby(packet_list, key=lambda packet: packet.timestamp):
            group = list(iterator)
            if self.state.belief is None:
                if prepared is not None:
                    raise PreparedPropagationError(
                        "cannot consume prepared propagation before belief initialization"
                    )
                self.state.batch_size = _packet_batch_size(group[0])
                self.state.belief = self._initial_belief(group[0])
            current = self.state.belief
            requested = self._requested_timestamp(current, timestamp)
            if prepared is not None:
                packet_batch_sizes = {_packet_batch_size(packet) for packet in group}
                if len(packet_batch_sizes) != 1:
                    raise PreparedPropagationError(
                        "prepared propagation observation modalities disagree on batch size"
                    )
                self._validate_prepared_propagation(
                    prepared,
                    current=current,
                    requested=requested,
                    packet_batch_size=packet_batch_sizes.pop(),
                )
                prepared._consume()
                prior = prepared.prior
                dt = prepared.delta_time
                prior_interval_collision_mask = prepared.interval_collision_mask
            else:
                dt = requested - current.timestamp
                predict_step = getattr(self.dynamics, "predict_step", None)
                prior_interval_collision_mask: Tensor | None = None
                if callable(predict_step):
                    propagation = predict_step(current, dt)
                    prior = propagation.belief
                    auxiliary = getattr(propagation, "auxiliary", None)
                    if not isinstance(auxiliary, Mapping):
                        raise TypeError("dynamics.predict_step auxiliary must be a mapping")
                    prior_interval_collision_mask = self._interval_collision_mask(
                        current,
                        auxiliary,
                    )
                else:
                    prior = self.dynamics.predict(current, dt)
            posterior = prior
            for packet_index, packet in enumerate(self._order_group(group)):
                # Elapsed-time evidence belongs to the first assimilation only.
                # Later modalities at the same timestamp correct that posterior
                # without applying temporal position→velocity coupling twice.
                packet_dt = dt if packet_index == 0 else torch.zeros_like(dt)
                posterior = self._ingest_one(
                    packet,
                    posterior,
                    packet_dt,
                    prior_interval_collision_mask,
                )
            self.state.belief = posterior.with_timestamp(timestamp)
            self.state.ingest_count += 1
        assert self.state.belief is not None
        return self.state.belief

    def initialize(
        self,
        packets: Sequence[ObservationPacket],
    ) -> WorldBelief:
        if not packets:
            raise ValueError("initialize requires at least one packet")
        self.reset(batch_size=_packet_batch_size(packets[0]))
        return self.ingest(packets)

    def predict(
        self,
        query_times: Sequence[float] | Tensor,
    ) -> BeliefTrajectory:
        if self.state.belief is None:
            raise RuntimeError("OnlineWorldModel must ingest an observation first")
        times = torch.as_tensor(
            query_times,
            device=self.state.belief.device,
            dtype=self.state.belief.dtype,
        )
        if times.ndim not in {1, 2}:
            raise ValueError("query_times must be [T] or [B,T]")
        if not torch.isfinite(times).all() or torch.any(times < 0):
            raise ValueError("query_times must be finite nonnegative offsets")
        return self.dynamics.rollout(self.state.belief, times)

    def predict_hypotheses(
        self,
        hypothesis_pool: object,
        query_times: Sequence[float] | Tensor,
    ) -> list[BeliefTrajectory]:
        """Run a candidate pool from the current persistent belief.

        The pool is intentionally injected rather than stored as runtime
        truth: model hypotheses are replaceable, while ``self.state.belief``
        remains the sole source of truth for future cycles.
        """

        if self.state.belief is None:
            raise RuntimeError("OnlineWorldModel must ingest an observation first")
        rollout = getattr(hypothesis_pool, "rollout", None)
        if not callable(rollout):
            raise TypeError("hypothesis_pool must expose rollout(belief, query_times)")
        trajectories = rollout(self.state.belief, query_times)
        if not isinstance(trajectories, list) or not all(
            isinstance(item, BeliefTrajectory) for item in trajectories
        ):
            raise TypeError("hypothesis_pool.rollout must return list[BeliefTrajectory]")
        return trajectories

    def assimilate_hypotheses(
        self,
        hypothesis_pool: object,
        target_positions: Tensor,
        target_mask: Tensor,
        trajectories: Sequence[BeliefTrajectory],
        *,
        target_collision: Tensor | None = None,
        position_weight: float = 1.0,
        lifecycle_weight: float = 0.0,
        event_weight: float = 0.0,
        position_gate_ratio: float = 0.0,
        axis_gate_ratio: float = 0.0,
        axis_weights: Sequence[float] | Tensor | None = None,
        uncertainty_aware: bool = True,
        evidence_decay_override: float | None = None,
    ) -> object:
        """Assimilate delayed rollout evidence without mutating the belief."""

        if self.state.belief is None:
            raise RuntimeError("OnlineWorldModel must ingest an observation first")
        assimilate = getattr(hypothesis_pool, "assimilate", None)
        if not callable(assimilate):
            raise TypeError("hypothesis_pool must expose assimilate(...)")
        return assimilate(
            self.state.belief,
            target_positions,
            target_mask,
            trajectories=trajectories,
            target_collision=target_collision,
            position_weight=position_weight,
            lifecycle_weight=lifecycle_weight,
            event_weight=event_weight,
            position_gate_ratio=position_gate_ratio,
            axis_gate_ratio=axis_gate_ratio,
            axis_weights=axis_weights,
            uncertainty_aware=uncertainty_aware,
            evidence_decay_override=evidence_decay_override,
        )

    def step(
        self,
        packet: ObservationPacket,
        *,
        prediction_horizon: float | None = None,
    ) -> WorldBelief | tuple[WorldBelief, BeliefTrajectory]:
        belief = self.ingest(packet)
        if prediction_horizon is None:
            return belief
        future = self.predict([prediction_horizon])
        return belief, future

    def detach_state(self) -> None:
        self.state = self.state.detach()
