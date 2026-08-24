"""Strict, dependency-light YAML configuration for Project Orpheus."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from numbers import Real
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
    # Branch-heavy sequential filtering can be slower on MPS than CPU even
    # when batched CNN pretraining benefits from MPS. ``same`` preserves the
    # historical single-device behavior.
    closed_loop_preference: str = "same"
    # PyTorch 2.10 MPS may emit data-dependent NaN matrix gradients in the
    # small proposal transformer. Keep CNNs on MPS but execute that block on
    # CPU through differentiable device copies.
    global_detector_cpu_on_mps: bool = True
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
    ensured_pair_lateral_offset_range: tuple[float, float] = (0.0, 0.0)
    ensured_pair_vertical_speed_range: tuple[float, float] | None = None
    ensured_pair_event_frame_range: tuple[int, int] | None = None
    ensured_pair_floor_clearance_frames: int = 2
    ensured_pair_floor_clearance_margin_m: float = 0.03
    ensured_pair_scene_resample_attempts: int = 32
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
    # Opt-in dense local-maximum proposal detector. Disabled preserves the
    # historical query detector module/state and exact runtime behavior.
    dense_global_detector_enabled: bool = False
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
    temporal_velocity_independent_raw_history_enabled: bool = False
    temporal_velocity_continuous_gravity_axis_enabled: bool = False
    temporal_velocity_post_event_gravity_axis_enabled: bool = False
    temporal_velocity_unobserved_variance: float = 1.0e4
    temporal_velocity_reset_on_collision: bool = False
    temporal_velocity_max_age_steps: int | None = None
    temporal_velocity_post_event_max_samples: int | None = None
    temporal_velocity_post_event_min_samples: int = 2
    temporal_velocity_change_point_enabled: bool = False
    temporal_velocity_change_point_minimum_speed: float = 0.25
    temporal_velocity_change_point_minimum_delta: float = 0.75
    temporal_velocity_change_point_strong_delta: float = 2.0
    temporal_velocity_change_point_require_contact_mode: bool = True
    temporal_velocity_change_point_gate: str = "heuristic"
    temporal_velocity_change_point_linear_weights: tuple[float, ...] = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    temporal_velocity_change_point_linear_bias: float = -8.0
    temporal_velocity_change_point_mlp_hidden_weights: tuple[float, ...] = ()
    temporal_velocity_change_point_mlp_hidden_bias: tuple[float, ...] = ()
    temporal_velocity_change_point_mlp_output_weights: tuple[float, ...] = ()
    temporal_velocity_change_point_mlp_output_bias: float = 0.0
    temporal_velocity_change_point_probability_threshold: float = 0.5
    temporal_velocity_change_point_minimum_interval_samples: int = 6
    temporal_velocity_outgoing_proposal_enabled: bool = False
    temporal_velocity_outgoing_proposal_hidden_weights: tuple[float, ...] = ()
    temporal_velocity_outgoing_proposal_hidden_bias: tuple[float, ...] = ()
    temporal_velocity_outgoing_proposal_output_weights: tuple[float, ...] = ()
    temporal_velocity_outgoing_proposal_output_bias: float = 0.0
    temporal_velocity_outgoing_proposal_variance: float = 1.0
    temporal_velocity_outgoing_proposal_maximum_delta: float = 5.0
    temporal_velocity_lateral_intervention_enabled: bool = False
    temporal_velocity_lateral_intervention_hidden_weights: tuple[float, ...] = ()
    temporal_velocity_lateral_intervention_hidden_bias: tuple[float, ...] = ()
    temporal_velocity_lateral_intervention_output_weights: tuple[float, ...] = ()
    temporal_velocity_lateral_intervention_output_bias: tuple[float, float] = (0.0, 0.0)
    temporal_velocity_lateral_intervention_variance_floor: float = 0.04
    temporal_velocity_lateral_intervention_variance_ceiling: float = 25.0
    temporal_velocity_lateral_intervention_gain_power: float = 2.0
    temporal_velocity_lateral_intervention_maximum_delta: float = 5.0
    temporal_velocity_gravity_intervention_enabled: bool = False
    temporal_velocity_gravity_intervention_hidden_weights: tuple[float, ...] = ()
    temporal_velocity_gravity_intervention_hidden_bias: tuple[float, ...] = ()
    temporal_velocity_gravity_intervention_output_weights: tuple[float, ...] = ()
    temporal_velocity_gravity_intervention_output_bias: tuple[float, float] = (0.0, 0.0)
    temporal_velocity_gravity_intervention_variance_floor: float = 0.04
    temporal_velocity_gravity_intervention_variance_ceiling: float = 25.0
    temporal_velocity_gravity_intervention_gain_power: float = 2.0
    temporal_velocity_gravity_intervention_maximum_delta: float = 5.0
    temporal_velocity_measurement_position_blend: float = 0.0
    temporal_velocity_position_innovation_coupling: bool = False
    temporal_position_enabled: bool = False
    temporal_position_min_samples: int = 3
    temporal_position_robust_threshold: float = 2.5
    temporal_position_variance_scale: float = 4.0
    temporal_position_variance_floor: float = 0.01
    temporal_position_variance_ceiling: float | None = None
    temporal_position_depth_only: bool = True
    structured_disc_center_enabled: bool = False
    structured_disc_threshold: float = 0.04
    structured_disc_min_pixels: int = 4
    structured_disc_max_assignment_distance: float = 0.75
    structured_disc_center_std_pixels: float = 0.75
    structured_disc_fast_depth_enabled: bool = False
    structured_disc_photometric_fast_depth_enabled: bool = False
    structured_disc_photometric_maximum_fit_rms: float = 0.035
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
    # Opt-in multi-rate execution. ``None`` retains one graph/attention
    # evaluation per analytic microstep for exact historical compatibility.
    learned_effect_interval_seconds: float | None = None
    # Smooth, parameter-free support for learned pair/event residuals. Disabled
    # preserves the historical broad candidate-mask behavior exactly.
    pair_applicability_enabled: bool = False
    pair_applicability_lookahead_seconds: float = 0.05
    pair_applicability_margin_m: float = 0.05
    pair_applicability_gap_temperature_m: float = 0.025
    pair_applicability_velocity_temperature_mps: float = 0.10
    hidden_dim: int = 96
    interaction_radius: float = 1.0
    process_noise_position: float = 1e-4
    process_noise_velocity: float = 2e-3
    modal_acceleration_scale: float = 0.25
    residual_acceleration_scale: float = 0.5
    attention_residual_enabled: bool = False
    # Historical attention checkpoints used relation geometry without explicit
    # entity-endpoint incidence. Keep that exact function unless a new
    # protocol opts into symmetric endpoint binding.
    attention_relation_endpoint_binding_enabled: bool = False
    attention_width: int = 128
    attention_heads: int = 4
    attention_layers: int = 4
    attention_feed_forward_width: int = 512
    attention_dropout: float = 0.0
    contact_margin: float = 0.0
    boundary_contact_tolerance: float = 1.0e-4
    penetration_slop: float = 1e-4
    max_penetration_correction: float = 0.08
    contact_confidence_sigma: float = 0.0
    pair_collision_speed_epsilon: float = 1.0e-7
    boundary_collision_speed_epsilon: float = 0.1
    # False is the exact historical hard-logit event semantic.  New protocols
    # may opt into continuous analytic hazards plus learned pair calibration;
    # physical contact jumps remain hard and fail-safe in both modes.
    smooth_event_hazard_enabled: bool = False
    event_hazard_gap_temperature_m: float = 0.02
    event_hazard_velocity_temperature_mps: float = 0.10
    event_hazard_resolved_logit_floor: float = 2.0
    sleep_speed: float = 0.05
    allow_large_substep: bool = False


@dataclass(frozen=True)
class FilterConfig:
    hidden_dim: int = 128
    robust_clip: float = 5.0
    min_log_variance: float = -12.0
    max_log_variance: float = 8.0
    learned_residual_scale: float = 0.15
    # False preserves checkpoints trained before specification 1.19. New
    # protocols should opt into evidence-anchored, component-masked updates.
    innovation_anchored_correction: bool = False
    # False preserves the exact pre-provenance learned correction. New
    # protocols may bind learned position and position-derived velocity
    # mean/variance updates to independently observed world axes.
    learned_correction_independent_axis_support: bool = False
    missed_variance_growth: float = 0.08


@dataclass(frozen=True)
class AssociationConfig:
    geometry_weight: float = 1.0
    appearance_weight: float = 0.25
    existence_weight: float = 0.05
    mahalanobis_gate: float = 16.0
    maximum_cost: float = 25.0
    ambiguity_margin: float = 0.02
    minimum_measurement_confidence: float = 0.45


@dataclass(frozen=True)
class LifecycleConfig:
    birth_confidence: float = 0.55
    birth_confirmations: int = 1
    birth_confirmation_distance_m: float = 0.5
    max_missed_steps: int = 12
    max_occluded_steps: int = 60
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
    # Opt-in, RGB-measurement-only delayed rollout selection.  This remains
    # separate from WorldBelief: candidates are planning proposals and their
    # evidence is never an alternate state estimate.
    hypothesis_pool_enabled: bool = False
    hypothesis_evidence_horizons_seconds: tuple[float, ...] = (0.05,)
    hypothesis_axis_independent_axes: tuple[int, ...] = (0,)
    hypothesis_axis_prior_strength: float = 0.001
    hypothesis_evidence_decay: float = 1.0
    hypothesis_timestamp_tolerance_seconds: float = 1.0e-5
    hypothesis_local_applicability_enabled: bool = False
    hypothesis_minimum_support_count: int = 1
    hypothesis_maximum_evidence_age_seconds: float = 1.0
    hypothesis_minimum_observability: float = 0.0
    hypothesis_minimum_confidence_margin: float = 0.0
    hypothesis_velocity_evidence_weight: float = 0.0
    hypothesis_velocity_nonregression_gate_enabled: bool = False
    hypothesis_residual_correction_gain_by_axis: tuple[float, float, float] = (0.0, 0.0, 0.0)
    hypothesis_robust_influence_delta: float = 0.0
    hypothesis_composition_step_seconds: float | None = None
    hypothesis_online_acceleration_enabled: bool = False
    hypothesis_online_acceleration_minimum_support_count: int = 4
    hypothesis_online_acceleration_maximum_mps2: float = 20.0
    # Opt-in execution optimization: semigroup-safe candidates share one
    # chronological rollout across all evidence horizons. Unsafe candidates
    # retain the historical independent-horizon schedule.
    hypothesis_shared_horizon_rollout_enabled: bool = False


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 4
    # When enabled, every training batch contains equal support from every
    # declared simulator scenario. This is an optimization-protocol choice,
    # not a validation balancing shortcut.
    scenario_balanced_batches: bool = False
    steps: int = 1000
    learning_rate: float = 3e-4
    closed_loop_learning_rate_scale: float = 0.1
    # Keep the historical constant-rate trajectory unless a new protocol opts
    # into a deterministic causal-update schedule.  The cosine horizon is an
    # absolute update count rather than ``training.steps`` so a convergence
    # extension cannot silently reshape an already-running schedule.
    closed_loop_learning_rate_schedule: str = "constant"
    closed_loop_learning_rate_warmup_steps: int = 0
    # Number of cosine-decay updates after warmup, not an endpoint inferred
    # from the mutable overall training budget.
    closed_loop_learning_rate_cosine_decay_steps: int | None = None
    closed_loop_learning_rate_minimum_scale: float = 0.1
    closed_loop_global_trainable_steps: int = 50
    closed_loop_trainable_scope: str = "all"
    # Optional causal-update boundary for a declared two-scope curriculum.
    # The primary scope applies before this many completed causal updates;
    # the late scope applies from the boundary onward.
    closed_loop_late_trainable_scope: str | None = None
    closed_loop_scope_transition_steps: int | None = None
    # Optional exact per-scope overrides for the event objective.  An empty
    # mapping preserves the historical ``loss_weights.event`` behavior.  This
    # lets staged causal curricula suppress event supervision until a scope
    # with an explicit event/relation owner is trainable.
    closed_loop_event_loss_weights: dict[str, float] = field(default_factory=dict)
    # Optional parameter-specific node-event routing into the learned state
    # heads of the node-collision scope.  The canonical forward objective and
    # relation-row weight above remain unchanged; an empty mapping is exact
    # specification-1.60 behavior.
    closed_loop_state_event_loss_weights: dict[str, float] = field(default_factory=dict)
    # A measurement-only checkpoint may score well on the few proposals that
    # survive lifecycle gating while destroying the persistent runtime's
    # training support.  At the stage boundary, require both absolute and
    # reference-relative current/future coverage before using that candidate
    # as the mutable causal starting point.
    handoff_minimum_target_coverage: float = 0.05
    handoff_minimum_forecast_coverage: float = 0.05
    handoff_minimum_reference_coverage_ratio: float = 0.50
    # Unsupported causal windows are data draws, not optimiser updates.
    # Resample deterministically up to this bound and then fail loudly.
    maximum_no_gradient_batches_per_update: int = 32
    minimum_effective_gradient_norm: float = 1.0e-12
    weight_decay: float = 1e-4
    tbptt_steps: int = 24
    grad_clip_norm: float = 1.0
    # Recursive multi-horizon losses can amplify the same learned interaction
    # residual across many substeps. Bound that subsystem before the global
    # clip so one edge-Jacobian spike cannot suppress unrelated gradients.
    interaction_grad_clip_norm: float = 1.0
    # Optional fail-fast gate after all typed-output and decoder-row caps. A
    # smaller retained fraction means local isolation failed, so applying the
    # normalized update would starve the complete interaction model.
    minimum_interaction_gradient_retention: float | None = None
    # Recursive node-acceleration residuals can accumulate across rollout
    # substeps even after every invocation is locally bounded. Bound the
    # complete x/y/z decoder group before the interaction-wide cap.
    attention_node_grad_clip_norm: float | None = None
    # Rare collision-event batches can put a large direct gradient on the
    # typed attention collision-logit decoder row.  Optionally bound that row
    # before the complete interaction group so event imbalance cannot reduce
    # the effective update of unrelated force, uncertainty, and token paths.
    attention_collision_grad_clip_norm: float | None = None
    # Recursive force residuals can accumulate through many rollout substeps
    # and make the typed normal/tangent decoder rows dominate the complete
    # attention gradient. Optionally bound those two semantic rows jointly
    # before the interaction cap, without changing their forward outputs.
    attention_force_grad_clip_norm: float | None = None
    # Recursive collision impulses use distinct multiplier/additive outputs.
    # Bound those rows separately so one jump cannot starve force/event paths.
    attention_impulse_grad_clip_norm: float | None = None
    # Parameter-row caps run after gradients have already traversed the shared
    # attention stack. These optional per-invocation output-gradient caps act
    # at the typed decoder boundary so rare recursive node/event/force/impulse
    # signals cannot first dominate every shared token projection and block.
    attention_node_output_grad_clip_norm: float | None = None
    attention_collision_output_grad_clip_norm: float | None = None
    attention_force_output_grad_clip_norm: float | None = None
    attention_impulse_output_grad_clip_norm: float | None = None
    # RGB discovery and the shared ROI backbone can likewise dominate the
    # whole-model norm during causal adaptation. Bound the complete, disjoint
    # RGB observation module before the global clip.
    closed_loop_perception_grad_clip_norm: float = 1.0
    checkpoint_every: int = 100
    eval_every: int = 100
    log_every: int = 10
    train_episodes: int = 256
    validation_episodes: int = 16
    num_workers: int = 0
    fixed_dataset: bool = False
    rgb_pretrain_steps: int = 100
    # ``all`` preserves historical paired global/fast RGB pretraining.
    # ``global_detector`` freezes the shared backbone and every non-global
    # owner so a balanced discovery curriculum cannot drift the protected
    # persistent state, ROI, identifier, or dynamics paths.
    rgb_pretrain_trainable_scope: str = "all"
    fast_roi_pretrain_weight: float = 1.0
    measurement_validation_frames: int = 8
    perturbation_probability: float = 0.25
    perturbation_position_std: float = 0.12
    perturbation_velocity_std: float = 0.20
    collision_window_probability: float = 0.50
    long_horizon_window_probability: float = 0.50
    # Keep axis-specific rollout objectives on the same fixed global horizon
    # denominator as the aggregate position objective.  Disabling this is a
    # legacy-compatibility escape hatch for an already-running campaign.
    normalize_rollout_axes_over_configured_horizons: bool = True
    # When collision and maximum-horizon sampling are both requested, retain a
    # maximum-horizon-capable window.  If one window can cover both constraints
    # it does; otherwise the long-horizon request wins over a late collision.
    joint_collision_long_horizon_sampling: bool = True
    # Point forecasts before this per-track age are treated as cold-start
    # distributional supervision rather than deterministic trajectory targets.
    minimum_rollout_age_steps: int = 0
    # ``None`` preserves the historical behavior of scoring every eligible
    # frame in a TBPTT window. Long-running profiles may bound the expensive
    # recursive rollouts while still ingesting and supervising every frame.
    rollout_anchors_per_window: int | None = None
    # Historical causal training builds one detached prior rollout at every
    # scored post-initial anchor so it can optimize future correction
    # improvement.  This is an explicit objective/compute protocol: disabling
    # it retains current-correction and posterior-rollout supervision while
    # omitting the extra prior rollout and every future-correction loss term.
    closed_loop_prior_future_correction_enabled: bool = True
    # False is the exact historical object/coordinate-pooled objective. New
    # protocols may normalize supported physical losses within each batch row
    # before averaging rows so object-rich scenarios cannot dominate.
    closed_loop_batch_macro_physical_losses_enabled: bool = False
    # False preserves vector-norm correction non-regression. New protocols may
    # require non-regression independently on each supported coordinate.
    closed_loop_axiswise_correction_hinge_enabled: bool = False
    # Keep the forward causal graph unchanged while assigning backward
    # ownership by subsystem: direct RGB objectives train perception, and
    # physical state/rollout objectives train the recurrent filter and slow
    # identifier. False preserves the historical fully coupled backward.
    closed_loop_modular_gradient_ownership_enabled: bool = False
    # Training-only relaxation of hard Hungarian association. ``None`` keeps
    # the exact legacy runtime/training graph. A positive temperature exposes
    # the same gated cost matrix to differentiable physical supervision while
    # deployment continues to use deterministic discrete assignments.
    closed_loop_soft_association_temperature: float | None = None
    # Forward-exact, training-only gradient carrier from a gated soft analytic
    # correction into the hard posterior. Integer identity/lifecycle and all
    # deployed values remain hard; recursive rollout gradients follow the
    # equation-based soft posterior surrogate across assignment boundaries.
    closed_loop_soft_posterior_straight_through_enabled: bool = False
    # Optional group-robust physical objective for one-row-per-scenario
    # balanced batches. Each axis/horizon cell reduces the highest-loss
    # supported fraction of scenario rows instead of allowing easy regimes to
    # cancel a hard-regime regression. None is the exact legacy reduction.
    closed_loop_scenario_tail_fraction: float | None = None
    # Optional forward-exact robustification for physical Gaussian-NLL
    # optimization. Reported NLL remains the exact proper-score value; only
    # standardized-error backward influence above this cap is logarithmic.
    closed_loop_uncertainty_standardized_error_gradient_cap: float | None = None
    # Optional frozen-reference non-regression objective.  A positive weight
    # compares each scenario row, physical axis, and current/forecast horizon
    # against the immutable step-zero rollout reference on the exact same
    # causal batch/window/perturbation stream.  Zero is exact legacy behavior.
    closed_loop_protected_reference_nonregression_weight: float = 0.0
    # Trend validation still ingests and scores every current frame, but may
    # use a deterministic spread of forecast anchors. Full promotion
    # evaluation remains a separate, larger manifest.
    validation_rollout_anchors_per_episode: int | None = None
    # Validation may batch posterior rollout anchors from one already-ingested
    # episode. One preserves the historical serial execution exactly.
    validation_rollout_anchor_batch_size: int = 1
    # A nonzero RMSE denominator alone can make a scenario look supported from
    # one lucky tracked object. Require explicit label-only causal opportunity,
    # matched point support, and multiple independently generated episodes
    # before a scenario slice may authorize checkpoint promotion.
    validation_minimum_predictable_target_count_per_scenario_horizon: int = 1
    validation_minimum_matched_target_count_per_scenario_horizon: int = 1
    validation_minimum_supported_episodes_per_scenario: int = 1
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
            "rollout_nll": 0.02,
            "event": 0.2,
            "parameter": 0.1,
            "existence": 0.2,
            "uncertainty": 0.05,
            "correction": 0.02,
            "soft_association_state": 0.0,
            "soft_association_velocity": 0.0,
            "soft_association_exclusivity": 0.0,
            "rgb_reprojection": 0.0,
            "soft_shadow_physical": 0.0,
        }
    )


@dataclass(frozen=True)
class EvaluationConfig:
    horizons_seconds: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0, 2.0)
    episodes: int = 16
    rgb_only: bool = True
    # Recovery is an independent checkpoint-loaded prefix replay. The primary
    # evaluator trajectory is always intervention-free.
    recovery_probe_enabled: bool = False
    perturbation_position_std: float = 0.15
    perturbation_velocity_std: float = 0.25
    confidence_level: float = 0.90
    benchmark_warmup: int = 2
    # Optional coordinate-wise hypothesis composition. Joint selection remains
    # the safe default; coordinates listed here use delayed per-axis evidence.
    hypothesis_axis_independent: bool = False
    hypothesis_axis_independent_axes: tuple[int, ...] = (0, 1)
    hypothesis_axis_prior_strength: float = 0.0


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
        if any(
            axis not in (0, 1, 2) for axis in self.evaluation.hypothesis_axis_independent_axes
        ) or len(set(self.evaluation.hypothesis_axis_independent_axes)) != len(
            self.evaluation.hypothesis_axis_independent_axes
        ):
            raise ValueError(
                "evaluation.hypothesis_axis_independent_axes must contain unique axes 0, 1, or 2"
            )
        if (
            self.evaluation.hypothesis_axis_independent
            and not self.evaluation.hypothesis_axis_independent_axes
        ):
            raise ValueError("axis-independent evaluation requires at least one axis")
        if not math.isfinite(self.evaluation.hypothesis_axis_prior_strength) or not (
            0.0 <= self.evaluation.hypothesis_axis_prior_strength <= 1.0
        ):
            raise ValueError("evaluation.hypothesis_axis_prior_strength must lie in [0,1]")
        runtime = self.runtime
        if any(axis not in (0, 1, 2) for axis in runtime.hypothesis_axis_independent_axes) or len(
            set(runtime.hypothesis_axis_independent_axes)
        ) != len(runtime.hypothesis_axis_independent_axes):
            raise ValueError(
                "runtime.hypothesis_axis_independent_axes must contain unique axes 0, 1, or 2"
            )
        if runtime.hypothesis_pool_enabled and not runtime.hypothesis_axis_independent_axes:
            raise ValueError("enabled runtime hypothesis selection requires at least one axis")
        if not runtime.hypothesis_evidence_horizons_seconds or any(
            not math.isfinite(horizon) or horizon <= 0.0
            for horizon in runtime.hypothesis_evidence_horizons_seconds
        ):
            raise ValueError(
                "runtime.hypothesis_evidence_horizons_seconds must be finite and positive"
            )
        if not math.isfinite(runtime.hypothesis_evidence_decay) or not (
            0.0 < runtime.hypothesis_evidence_decay <= 1.0
        ):
            raise ValueError("runtime.hypothesis_evidence_decay must lie in (0,1]")
        if not math.isfinite(runtime.hypothesis_axis_prior_strength) or not (
            0.0 <= runtime.hypothesis_axis_prior_strength <= 1.0
        ):
            raise ValueError("runtime.hypothesis_axis_prior_strength must lie in [0,1]")
        if not math.isfinite(runtime.hypothesis_timestamp_tolerance_seconds) or (
            runtime.hypothesis_timestamp_tolerance_seconds < 0.0
        ):
            raise ValueError(
                "runtime.hypothesis_timestamp_tolerance_seconds must be finite and nonnegative"
            )
        if not isinstance(runtime.hypothesis_local_applicability_enabled, bool):
            raise ValueError("runtime.hypothesis_local_applicability_enabled must be boolean")
        if not isinstance(runtime.hypothesis_velocity_nonregression_gate_enabled, bool):
            raise ValueError(
                "runtime.hypothesis_velocity_nonregression_gate_enabled must be boolean"
            )
        if not isinstance(runtime.hypothesis_online_acceleration_enabled, bool):
            raise ValueError("runtime.hypothesis_online_acceleration_enabled must be boolean")
        if not isinstance(runtime.hypothesis_shared_horizon_rollout_enabled, bool):
            raise ValueError("runtime.hypothesis_shared_horizon_rollout_enabled must be boolean")
        if (
            not isinstance(runtime.hypothesis_online_acceleration_minimum_support_count, int)
            or isinstance(runtime.hypothesis_online_acceleration_minimum_support_count, bool)
            or runtime.hypothesis_online_acceleration_minimum_support_count <= 0
        ):
            raise ValueError(
                "runtime.hypothesis_online_acceleration_minimum_support_count "
                "must be a positive integer"
            )
        if (
            isinstance(runtime.hypothesis_online_acceleration_maximum_mps2, bool)
            or not isinstance(runtime.hypothesis_online_acceleration_maximum_mps2, Real)
            or not math.isfinite(runtime.hypothesis_online_acceleration_maximum_mps2)
            or runtime.hypothesis_online_acceleration_maximum_mps2 <= 0.0
        ):
            raise ValueError(
                "runtime.hypothesis_online_acceleration_maximum_mps2 must be finite and positive"
            )
        if runtime.hypothesis_online_acceleration_enabled:
            if not runtime.hypothesis_local_applicability_enabled:
                raise ValueError("online acceleration requires local hypothesis applicability")
            if not model.rgb.temporal_velocity_enabled:
                raise ValueError("online acceleration requires causal RGB temporal velocity")
        gains = runtime.hypothesis_residual_correction_gain_by_axis
        if not isinstance(gains, (tuple, list)) or len(gains) != 3:
            raise ValueError(
                "runtime.hypothesis_residual_correction_gain_by_axis must contain three values"
            )
        for value in gains:
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value < 0.0
                or value > 1.0
            ):
                raise ValueError(
                    "runtime.hypothesis_residual_correction_gain_by_axis values must lie in [0,1]"
                )
        residual_axes = {axis for axis, value in enumerate(gains) if value > 0.0}
        if residual_axes and not runtime.hypothesis_local_applicability_enabled:
            raise ValueError(
                "runtime.hypothesis_residual_correction_gain_by_axis requires local applicability"
            )
        if not residual_axes.issubset(set(runtime.hypothesis_axis_independent_axes)):
            raise ValueError(
                "runtime.hypothesis_residual_correction_gain_by_axis axes must be independently configured"
            )
        if (
            not isinstance(runtime.hypothesis_minimum_support_count, int)
            or isinstance(runtime.hypothesis_minimum_support_count, bool)
            or runtime.hypothesis_minimum_support_count <= 0
        ):
            raise ValueError("runtime.hypothesis_minimum_support_count must be a positive integer")
        for name, value in (
            (
                "hypothesis_maximum_evidence_age_seconds",
                runtime.hypothesis_maximum_evidence_age_seconds,
            ),
            ("hypothesis_minimum_observability", runtime.hypothesis_minimum_observability),
            (
                "hypothesis_minimum_confidence_margin",
                runtime.hypothesis_minimum_confidence_margin,
            ),
            (
                "hypothesis_velocity_evidence_weight",
                runtime.hypothesis_velocity_evidence_weight,
            ),
            ("hypothesis_robust_influence_delta", runtime.hypothesis_robust_influence_delta),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(f"runtime.{name} must be finite and nonnegative")
        if runtime.hypothesis_minimum_observability > 1.0:
            raise ValueError("runtime.hypothesis_minimum_observability must lie in [0,1]")
        if runtime.hypothesis_minimum_confidence_margin > 1.0:
            raise ValueError("runtime.hypothesis_minimum_confidence_margin must lie in [0,1]")
        if runtime.hypothesis_composition_step_seconds is not None:
            step = runtime.hypothesis_composition_step_seconds
            if (
                isinstance(step, bool)
                or not isinstance(step, Real)
                or not math.isfinite(step)
                or step <= 0.0
            ):
                raise ValueError(
                    "runtime.hypothesis_composition_step_seconds must be null or finite and positive"
                )
            if not runtime.hypothesis_local_applicability_enabled:
                raise ValueError("runtime hypothesis composition requires local applicability")
            if not any(
                math.isclose(step, horizon, rel_tol=0.0, abs_tol=1.0e-9)
                for horizon in runtime.hypothesis_evidence_horizons_seconds
            ):
                raise ValueError(
                    "runtime hypothesis composition step must match a supported evidence horizon"
                )
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
        if model.lifecycle.max_occluded_steps < model.lifecycle.max_missed_steps:
            raise ValueError(
                "model.lifecycle.max_occluded_steps must be no smaller than "
                "model.lifecycle.max_missed_steps"
            )
        if not 0.0 <= model.association.minimum_measurement_confidence <= 1.0:
            raise ValueError("model.association.minimum_measurement_confidence must lie in [0,1]")
        if (
            not math.isfinite(model.association.maximum_cost)
            or model.association.maximum_cost <= 0.0
        ):
            raise ValueError("model.association.maximum_cost must be finite and positive")
        for name, value in (
            (
                "pair_collision_speed_epsilon",
                model.dynamics.pair_collision_speed_epsilon,
            ),
            (
                "boundary_collision_speed_epsilon",
                model.dynamics.boundary_collision_speed_epsilon,
            ),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"model.dynamics.{name} must be finite and nonnegative")
        if not isinstance(model.dynamics.smooth_event_hazard_enabled, bool):
            raise ValueError("model.dynamics.smooth_event_hazard_enabled must be boolean")
        for name, value in (
            (
                "event_hazard_gap_temperature_m",
                model.dynamics.event_hazard_gap_temperature_m,
            ),
            (
                "event_hazard_velocity_temperature_mps",
                model.dynamics.event_hazard_velocity_temperature_mps,
            ),
            (
                "event_hazard_resolved_logit_floor",
                model.dynamics.event_hazard_resolved_logit_floor,
            ),
        ):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"model.dynamics.{name} must be finite and positive")
        if (
            isinstance(model.rgb.global_every_steps, bool)
            or not isinstance(model.rgb.global_every_steps, int)
            or model.rgb.global_every_steps <= 0
        ):
            raise ValueError("model.rgb.global_every_steps must be a positive integer")
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
        if model.rgb.temporal_velocity_continuous_gravity_axis_enabled and (
            not model.rgb.temporal_velocity_enabled
        ):
            raise ValueError(
                "model.rgb continuous gravity-axis velocity requires temporal velocity"
            )
        if model.rgb.temporal_velocity_independent_raw_history_enabled and not (
            model.rgb.temporal_velocity_enabled or model.rgb.temporal_position_enabled
        ):
            raise ValueError(
                "model.rgb independent raw RGB history requires temporal velocity "
                "or temporal position"
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
        if model.rgb.temporal_velocity_change_point_gate not in {
            "heuristic",
            "linear",
            "mlp",
        }:
            raise ValueError(
                "model.rgb.temporal_velocity_change_point_gate must be heuristic, linear, or mlp"
            )
        if len(model.rgb.temporal_velocity_change_point_linear_weights) != 9 or not all(
            math.isfinite(value)
            for value in model.rgb.temporal_velocity_change_point_linear_weights
        ):
            raise ValueError(
                "model.rgb.temporal_velocity_change_point_linear_weights must "
                "contain nine finite values"
            )
        if not math.isfinite(model.rgb.temporal_velocity_change_point_linear_bias):
            raise ValueError("model.rgb.temporal_velocity_change_point_linear_bias must be finite")
        mlp_hidden = len(model.rgb.temporal_velocity_change_point_mlp_hidden_bias)
        if model.rgb.temporal_velocity_change_point_gate == "mlp" and (
            mlp_hidden <= 0
            or len(model.rgb.temporal_velocity_change_point_mlp_hidden_weights) != 9 * mlp_hidden
            or len(model.rgb.temporal_velocity_change_point_mlp_output_weights) != mlp_hidden
        ):
            raise ValueError("model.rgb change-point MLP coefficient dimensions are inconsistent")
        if not all(
            math.isfinite(value)
            for values in (
                model.rgb.temporal_velocity_change_point_mlp_hidden_weights,
                model.rgb.temporal_velocity_change_point_mlp_hidden_bias,
                model.rgb.temporal_velocity_change_point_mlp_output_weights,
            )
            for value in values
        ) or not math.isfinite(model.rgb.temporal_velocity_change_point_mlp_output_bias):
            raise ValueError("model.rgb change-point MLP coefficients must be finite")
        proposal_hidden = len(model.rgb.temporal_velocity_outgoing_proposal_hidden_bias)
        if model.rgb.temporal_velocity_outgoing_proposal_enabled and (
            not model.rgb.temporal_velocity_change_point_enabled
            or model.rgb.temporal_velocity_change_point_gate not in {"linear", "mlp"}
            or proposal_hidden <= 0
            or len(model.rgb.temporal_velocity_outgoing_proposal_hidden_weights)
            != 11 * proposal_hidden
            or len(model.rgb.temporal_velocity_outgoing_proposal_output_weights) != proposal_hidden
        ):
            raise ValueError(
                "model.rgb outgoing velocity proposal requires a learned gate "
                "and consistent eleven-input MLP coefficients"
            )
        if (
            not math.isfinite(model.rgb.temporal_velocity_outgoing_proposal_variance)
            or model.rgb.temporal_velocity_outgoing_proposal_variance <= 0
        ):
            raise ValueError(
                "model.rgb.temporal_velocity_outgoing_proposal_variance must be finite and positive"
            )
        if (
            not math.isfinite(model.rgb.temporal_velocity_outgoing_proposal_maximum_delta)
            or model.rgb.temporal_velocity_outgoing_proposal_maximum_delta <= 0
        ):
            raise ValueError(
                "model.rgb.temporal_velocity_outgoing_proposal_maximum_delta "
                "must be finite and positive"
            )
        lateral_hidden = len(model.rgb.temporal_velocity_lateral_intervention_hidden_bias)
        if model.rgb.temporal_velocity_lateral_intervention_enabled and (
            not model.rgb.temporal_velocity_enabled
            or not model.rgb.temporal_velocity_lateral_only
            or lateral_hidden <= 0
            or len(model.rgb.temporal_velocity_lateral_intervention_hidden_weights)
            != 19 * lateral_hidden
            or len(model.rgb.temporal_velocity_lateral_intervention_output_weights)
            != 2 * lateral_hidden
        ):
            raise ValueError(
                "model.rgb lateral velocity intervention requires lateral temporal "
                "velocity and consistent nineteen-input, two-output MLP coefficients"
            )
        lateral_coefficients = (
            model.rgb.temporal_velocity_lateral_intervention_hidden_weights,
            model.rgb.temporal_velocity_lateral_intervention_hidden_bias,
            model.rgb.temporal_velocity_lateral_intervention_output_weights,
            model.rgb.temporal_velocity_lateral_intervention_output_bias,
        )
        if not all(math.isfinite(value) for values in lateral_coefficients for value in values):
            raise ValueError("model.rgb lateral intervention coefficients must be finite")
        if (
            not math.isfinite(model.rgb.temporal_velocity_lateral_intervention_variance_floor)
            or not math.isfinite(model.rgb.temporal_velocity_lateral_intervention_variance_ceiling)
            or not 0
            < model.rgb.temporal_velocity_lateral_intervention_variance_floor
            <= model.rgb.temporal_velocity_lateral_intervention_variance_ceiling
        ):
            raise ValueError(
                "model.rgb lateral intervention variance bounds must be finite, "
                "positive, and ordered"
            )
        if (
            not math.isfinite(model.rgb.temporal_velocity_lateral_intervention_gain_power)
            or model.rgb.temporal_velocity_lateral_intervention_gain_power < 1
        ):
            raise ValueError(
                "model.rgb lateral intervention gain power must be finite and at least one"
            )
        if (
            not math.isfinite(model.rgb.temporal_velocity_lateral_intervention_maximum_delta)
            or model.rgb.temporal_velocity_lateral_intervention_maximum_delta <= 0
        ):
            raise ValueError(
                "model.rgb lateral intervention maximum delta must be finite and positive"
            )
        gravity_hidden = len(model.rgb.temporal_velocity_gravity_intervention_hidden_bias)
        if model.rgb.temporal_velocity_gravity_intervention_enabled and (
            not model.rgb.temporal_velocity_enabled
            or not model.rgb.temporal_velocity_lateral_only
            or gravity_hidden <= 0
            or len(model.rgb.temporal_velocity_gravity_intervention_hidden_weights)
            != 21 * gravity_hidden
            or len(model.rgb.temporal_velocity_gravity_intervention_output_weights)
            != 2 * gravity_hidden
        ):
            raise ValueError(
                "model.rgb gravity velocity intervention requires lateral-only temporal "
                "velocity and consistent twenty-one-input, two-output MLP coefficients"
            )
        gravity_coefficients = (
            model.rgb.temporal_velocity_gravity_intervention_hidden_weights,
            model.rgb.temporal_velocity_gravity_intervention_hidden_bias,
            model.rgb.temporal_velocity_gravity_intervention_output_weights,
            model.rgb.temporal_velocity_gravity_intervention_output_bias,
        )
        if not all(math.isfinite(value) for values in gravity_coefficients for value in values):
            raise ValueError("model.rgb gravity intervention coefficients must be finite")
        if (
            not math.isfinite(model.rgb.temporal_velocity_gravity_intervention_variance_floor)
            or not math.isfinite(model.rgb.temporal_velocity_gravity_intervention_variance_ceiling)
            or not 0
            < model.rgb.temporal_velocity_gravity_intervention_variance_floor
            <= model.rgb.temporal_velocity_gravity_intervention_variance_ceiling
        ):
            raise ValueError(
                "model.rgb gravity intervention variance bounds must be finite, "
                "positive, and ordered"
            )
        if (
            not math.isfinite(model.rgb.temporal_velocity_gravity_intervention_gain_power)
            or model.rgb.temporal_velocity_gravity_intervention_gain_power < 1
        ):
            raise ValueError(
                "model.rgb gravity intervention gain power must be finite and at least one"
            )
        if (
            not math.isfinite(model.rgb.temporal_velocity_gravity_intervention_maximum_delta)
            or model.rgb.temporal_velocity_gravity_intervention_maximum_delta <= 0
        ):
            raise ValueError(
                "model.rgb gravity intervention maximum delta must be finite and positive"
            )
        if (
            not math.isfinite(model.rgb.temporal_velocity_change_point_probability_threshold)
            or not 0.0 < model.rgb.temporal_velocity_change_point_probability_threshold < 1.0
        ):
            raise ValueError(
                "model.rgb.temporal_velocity_change_point_probability_threshold must lie in (0, 1)"
            )
        if model.rgb.temporal_velocity_change_point_minimum_interval_samples < 3:
            raise ValueError(
                "model.rgb.temporal_velocity_change_point_minimum_interval_samples "
                "must be at least three"
            )
        if (
            not math.isfinite(model.rgb.temporal_velocity_measurement_position_blend)
            or not 0.0 <= model.rgb.temporal_velocity_measurement_position_blend <= 1.0
        ):
            raise ValueError(
                "model.rgb.temporal_velocity_measurement_position_blend must lie in [0, 1]"
            )
        if (
            not 2
            <= model.rgb.temporal_position_min_samples
            <= (model.rgb.temporal_velocity_history_size)
        ):
            raise ValueError(
                "model.rgb.temporal_position_min_samples must lie between two "
                "and temporal_velocity_history_size"
            )
        if (
            not math.isfinite(model.rgb.temporal_position_robust_threshold)
            or model.rgb.temporal_position_robust_threshold <= 0
        ):
            raise ValueError(
                "model.rgb.temporal_position_robust_threshold must be finite and positive"
            )
        if (
            not math.isfinite(model.rgb.temporal_position_variance_scale)
            or model.rgb.temporal_position_variance_scale < 1
        ):
            raise ValueError(
                "model.rgb.temporal_position_variance_scale must be finite and at least one"
            )
        if (
            not math.isfinite(model.rgb.temporal_position_variance_floor)
            or model.rgb.temporal_position_variance_floor <= 0
        ):
            raise ValueError(
                "model.rgb.temporal_position_variance_floor must be finite and positive"
            )
        position_variance_ceiling = model.rgb.temporal_position_variance_ceiling
        if position_variance_ceiling is not None and (
            not math.isfinite(position_variance_ceiling)
            or position_variance_ceiling < model.rgb.temporal_position_variance_floor
        ):
            raise ValueError(
                "model.rgb.temporal_position_variance_ceiling must be finite "
                "and no smaller than temporal_position_variance_floor"
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
        if not isinstance(model.rgb.structured_disc_photometric_fast_depth_enabled, bool):
            raise ValueError(
                "model.rgb.structured_disc_photometric_fast_depth_enabled must be boolean"
            )
        if model.rgb.structured_disc_photometric_fast_depth_enabled and not (
            model.rgb.structured_disc_center_enabled
            and model.rgb.structured_disc_fast_depth_enabled
        ):
            raise ValueError(
                "model.rgb.structured_disc_photometric_fast_depth_enabled requires "
                "structured centre and fast depth"
            )
        if (
            isinstance(model.rgb.structured_disc_photometric_maximum_fit_rms, bool)
            or not isinstance(model.rgb.structured_disc_photometric_maximum_fit_rms, Real)
            or not math.isfinite(model.rgb.structured_disc_photometric_maximum_fit_rms)
            or model.rgb.structured_disc_photometric_maximum_fit_rms <= 0.0
        ):
            raise ValueError(
                "model.rgb.structured_disc_photometric_maximum_fit_rms must be finite and positive"
            )
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
        if not isinstance(model.rgb.dense_global_detector_enabled, bool):
            raise ValueError("model.rgb.dense_global_detector_enabled must be boolean")
        if not (
            0.0 <= model.lifecycle.occlusion_existence_decay <= model.lifecycle.existence_decay
        ):
            raise ValueError(
                "model.lifecycle occlusion_existence_decay must lie between "
                "zero and existence_decay"
            )
        if (
            isinstance(model.lifecycle.birth_confirmations, bool)
            or not isinstance(model.lifecycle.birth_confirmations, int)
            or model.lifecycle.birth_confirmations < 1
        ):
            raise ValueError("model.lifecycle.birth_confirmations must be a positive integer")
        if (
            not math.isfinite(model.lifecycle.birth_confirmation_distance_m)
            or model.lifecycle.birth_confirmation_distance_m <= 0.0
        ):
            raise ValueError(
                "model.lifecycle.birth_confirmation_distance_m must be finite and positive"
            )
        if not math.isfinite(model.dynamics.max_substep) or model.dynamics.max_substep <= 0:
            raise ValueError("model.dynamics.max_substep must be finite and positive")
        if model.dynamics.learned_effect_interval_seconds is not None and (
            isinstance(model.dynamics.learned_effect_interval_seconds, bool)
            or not math.isfinite(model.dynamics.learned_effect_interval_seconds)
            or model.dynamics.learned_effect_interval_seconds < model.dynamics.max_substep
        ):
            raise ValueError(
                "model.dynamics.learned_effect_interval_seconds must be finite and "
                "no smaller than max_substep"
            )
        if not isinstance(model.dynamics.pair_applicability_enabled, bool):
            raise ValueError("model.dynamics.pair_applicability_enabled must be boolean")
        for name, value in (
            (
                "pair_applicability_lookahead_seconds",
                model.dynamics.pair_applicability_lookahead_seconds,
            ),
            (
                "pair_applicability_margin_m",
                model.dynamics.pair_applicability_margin_m,
            ),
        ):
            if isinstance(value, bool) or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"model.dynamics.{name} must be finite and nonnegative")
        for name, value in (
            (
                "pair_applicability_gap_temperature_m",
                model.dynamics.pair_applicability_gap_temperature_m,
            ),
            (
                "pair_applicability_velocity_temperature_mps",
                model.dynamics.pair_applicability_velocity_temperature_mps,
            ),
        ):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"model.dynamics.{name} must be finite and positive")
        if not isinstance(model.dynamics.attention_residual_enabled, bool):
            raise ValueError("model.dynamics.attention_residual_enabled must be boolean")
        if not isinstance(
            model.dynamics.attention_relation_endpoint_binding_enabled,
            bool,
        ):
            raise ValueError(
                "model.dynamics.attention_relation_endpoint_binding_enabled must be boolean"
            )
        for name in ("attention_width", "attention_heads", "attention_layers"):
            value = getattr(model.dynamics, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"model.dynamics.{name} must be a positive integer")
        if model.dynamics.attention_width % model.dynamics.attention_heads != 0:
            raise ValueError("model.dynamics.attention_width must be divisible by attention_heads")
        if (
            isinstance(model.dynamics.attention_feed_forward_width, bool)
            or not isinstance(model.dynamics.attention_feed_forward_width, int)
            or model.dynamics.attention_feed_forward_width <= 0
        ):
            raise ValueError(
                "model.dynamics.attention_feed_forward_width must be a positive integer"
            )
        if (
            not math.isfinite(model.dynamics.attention_dropout)
            or not 0.0 <= model.dynamics.attention_dropout < 1.0
        ):
            raise ValueError("model.dynamics.attention_dropout must lie in [0,1)")
        for name, value in (
            ("contact_margin", model.dynamics.contact_margin),
            (
                "boundary_contact_tolerance",
                model.dynamics.boundary_contact_tolerance,
            ),
            ("contact_confidence_sigma", model.dynamics.contact_confidence_sigma),
        ):
            if value < 0 or not math.isfinite(value):
                raise ValueError(f"model.dynamics.{name} must be finite and nonnegative")
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
            (
                "ensured_pair_lateral_offset",
                simulator.ensured_pair_lateral_offset_range,
            ),
            ("external_impulse", simulator.external_impulse_range),
        ):
            if len(bounds) != 2 or bounds[0] > bounds[1]:
                raise ValueError(f"invalid simulator {name}_range")
        vertical_speed_range = simulator.ensured_pair_vertical_speed_range
        if vertical_speed_range is not None and (
            len(vertical_speed_range) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, Real)
                for value in vertical_speed_range
            )
            or not all(math.isfinite(float(value)) for value in vertical_speed_range)
            or vertical_speed_range[0] < 0.0
            or vertical_speed_range[1] < vertical_speed_range[0]
        ):
            raise ValueError(
                "simulator.ensured_pair_vertical_speed_range must be null or a finite "
                "nonnegative increasing pair"
            )
        if not (0 <= simulator.restitution_range[0] <= simulator.restitution_range[1] <= 1):
            raise ValueError("restitution_range must lie in [0, 1]")
        if not (0 <= simulator.friction_range[0] <= simulator.friction_range[1] <= 1):
            raise ValueError("friction_range must lie in [0, 1]")
        if simulator.ensure_collision and simulator.max_objects >= 2:
            event_frame_range = simulator.ensured_pair_event_frame_range
            if event_frame_range is not None and (
                len(event_frame_range) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in event_frame_range
                )
                or event_frame_range[0] < 1
                or event_frame_range[1] < event_frame_range[0]
                or event_frame_range[1] + simulator.ensured_pair_floor_clearance_frames - 1
                >= simulator.sequence_frames
            ):
                raise ValueError(
                    "simulator.ensured_pair_event_frame_range must be null or an "
                    "increasing positive integer pair that leaves the configured "
                    "floor-clearance frames inside the episode"
                )
            if (
                isinstance(simulator.ensured_pair_floor_clearance_frames, bool)
                or not isinstance(simulator.ensured_pair_floor_clearance_frames, int)
                or simulator.ensured_pair_floor_clearance_frames < 1
            ):
                raise ValueError(
                    "simulator.ensured_pair_floor_clearance_frames must be a positive integer"
                )
            if simulator.ensured_pair_floor_clearance_frames + 1 >= simulator.sequence_frames:
                raise ValueError(
                    "simulator.ensured_pair_floor_clearance_frames must leave room "
                    "for a pair event and later floor event"
                )
            if (
                not math.isfinite(simulator.ensured_pair_floor_clearance_margin_m)
                or simulator.ensured_pair_floor_clearance_margin_m <= 0
            ):
                raise ValueError(
                    "simulator.ensured_pair_floor_clearance_margin_m must be finite and positive"
                )
            if simulator.ensured_pair_lateral_offset_range[1] >= 2.0 * simulator.radius_range[0]:
                raise ValueError(
                    "simulator ensured pair lateral offset must remain below the "
                    "minimum combined sphere radius"
                )
            if (
                isinstance(simulator.ensured_pair_scene_resample_attempts, bool)
                or not isinstance(simulator.ensured_pair_scene_resample_attempts, int)
                or simulator.ensured_pair_scene_resample_attempts < 1
            ):
                raise ValueError(
                    "simulator.ensured_pair_scene_resample_attempts must be a positive integer"
                )
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
        if not isinstance(self.device.global_detector_cpu_on_mps, bool):
            raise ValueError("device.global_detector_cpu_on_mps must be boolean")
        for field_name in (
            "innovation_anchored_correction",
            "learned_correction_independent_axis_support",
        ):
            if not isinstance(getattr(self.model.filter, field_name), bool):
                raise ValueError(f"model.filter.{field_name} must be boolean")
        if self.device.closed_loop_preference not in {
            "same",
            "auto",
            "cpu",
            "mps",
            "cuda",
        }:
            raise ValueError(
                f"Unsupported closed-loop device preference {self.device.closed_loop_preference!r}"
            )
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
        quantized_horizons = [
            max(1, int(round(float(horizon) * simulator.frame_rate)))
            for horizon in self.evaluation.horizons_seconds
        ]
        if len(set(quantized_horizons)) != len(quantized_horizons):
            raise ValueError(
                "evaluation horizons must map to unique observation-frame offsets "
                "at simulator.frame_rate"
            )
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
        if not self.training.loss_weights or any(
            not math.isfinite(weight) or weight < 0
            for weight in self.training.loss_weights.values()
        ):
            raise ValueError("training.loss_weights must be nonempty, finite, and nonnegative")
        episode_duration = (simulator.sequence_frames - 1) / simulator.frame_rate
        if max(self.evaluation.horizons_seconds) > episode_duration + 1e-9:
            raise ValueError(
                "evaluation horizon exceeds generated episode duration "
                f"({max(self.evaluation.horizons_seconds):.3f}s > {episode_duration:.3f}s)"
            )
        if self.training.batch_size <= 0 or self.training.steps <= 0:
            raise ValueError("training batch_size and steps must be positive")
        if (
            isinstance(self.training.rgb_pretrain_steps, bool)
            or not isinstance(self.training.rgb_pretrain_steps, int)
            or self.training.rgb_pretrain_steps < 0
        ):
            raise ValueError("training.rgb_pretrain_steps must be a nonnegative integer")
        if self.training.rgb_pretrain_trainable_scope not in {
            "all",
            "global_detector",
            "dense_global_detector",
        }:
            raise ValueError(
                "training.rgb_pretrain_trainable_scope must be 'all', "
                "'global_detector', or 'dense_global_detector'"
            )
        if (
            self.training.rgb_pretrain_trainable_scope == "dense_global_detector"
            and not model.rgb.dense_global_detector_enabled
        ):
            raise ValueError(
                "training.rgb_pretrain_trainable_scope=dense_global_detector requires "
                "model.rgb.dense_global_detector_enabled=true"
            )
        if (
            self.training.rgb_pretrain_steps > 0
            and self.training.rgb_pretrain_trainable_scope
            in {"global_detector", "dense_global_detector"}
            and float(self.training.loss_weights.get("measurement", 0.0)) <= 0.0
        ):
            raise ValueError(
                "training.loss_weights.measurement must be positive when detector-only "
                "rgb_pretrain_steps are enabled"
            )
        if self.training.train_episodes <= 0 or self.training.validation_episodes <= 0:
            raise ValueError("training train_episodes and validation_episodes must be positive")
        if len(set(simulator.scenario_mixture)) != len(simulator.scenario_mixture):
            raise ValueError(
                "simulator.scenario_mixture must contain unique scenario names "
                "for deterministic balanced validation"
            )
        if self.training.validation_episodes < len(simulator.scenario_mixture):
            raise ValueError(
                "training.validation_episodes must cover every simulator scenario at least once"
            )
        validation_support_integer_fields = (
            "validation_minimum_predictable_target_count_per_scenario_horizon",
            "validation_minimum_matched_target_count_per_scenario_horizon",
            "validation_minimum_supported_episodes_per_scenario",
        )
        for name in validation_support_integer_fields:
            value = getattr(self.training, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"training.{name} must be a positive integer")
        minimum_scenario_episode_count = self.training.validation_episodes // len(
            simulator.scenario_mixture
        )
        if (
            self.training.validation_minimum_supported_episodes_per_scenario
            > minimum_scenario_episode_count
        ):
            raise ValueError(
                "training.validation_minimum_supported_episodes_per_scenario "
                "exceeds the guaranteed balanced validation episodes per scenario"
            )
        if self.training.learning_rate <= 0:
            raise ValueError("training.learning_rate must be positive")
        if not isinstance(self.training.scenario_balanced_batches, bool):
            raise ValueError("training.scenario_balanced_batches must be a boolean")
        if self.training.scenario_balanced_batches:
            scenario_count = len(simulator.scenario_mixture)
            if self.training.batch_size % scenario_count != 0:
                raise ValueError(
                    "training.batch_size must be a multiple of the scenario count "
                    "when scenario_balanced_batches is enabled"
                )
            if self.training.train_episodes % self.training.batch_size != 0:
                raise ValueError(
                    "training.train_episodes must be divisible by training.batch_size "
                    "when scenario_balanced_batches is enabled"
                )
        if not 0 < self.training.closed_loop_learning_rate_scale <= 1:
            raise ValueError("training.closed_loop_learning_rate_scale must lie in (0, 1]")
        schedule = self.training.closed_loop_learning_rate_schedule
        if schedule not in {"constant", "warmup_cosine"}:
            raise ValueError(
                "training.closed_loop_learning_rate_schedule must be 'constant' or 'warmup_cosine'"
            )
        warmup_steps = self.training.closed_loop_learning_rate_warmup_steps
        decay_steps = self.training.closed_loop_learning_rate_cosine_decay_steps
        minimum_scale = self.training.closed_loop_learning_rate_minimum_scale
        if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int) or warmup_steps < 0:
            raise ValueError(
                "training.closed_loop_learning_rate_warmup_steps must be a nonnegative integer"
            )
        if (
            isinstance(minimum_scale, bool)
            or not isinstance(minimum_scale, (int, float))
            or not 0.0 <= minimum_scale <= 1.0
        ):
            raise ValueError("training.closed_loop_learning_rate_minimum_scale must lie in [0, 1]")
        if schedule == "constant":
            if warmup_steps != 0 or decay_steps is not None:
                raise ValueError(
                    "constant closed-loop learning rate requires zero warmup and "
                    "null cosine decay steps"
                )
        else:
            if warmup_steps <= 0:
                raise ValueError(
                    "warmup_cosine closed-loop learning rate requires positive warmup steps"
                )
            if (
                isinstance(decay_steps, bool)
                or not isinstance(decay_steps, int)
                or decay_steps <= 0
            ):
                raise ValueError(
                    "warmup_cosine closed-loop learning rate requires positive cosine decay steps"
                )
        if self.training.closed_loop_global_trainable_steps < 0:
            raise ValueError("training.closed_loop_global_trainable_steps must be nonnegative")
        valid_closed_loop_scopes = {
            "all",
            "attention",
            "attention_relation",
            "attention_node_x",
            "attention_node_y",
            "attention_node_z",
            "dynamics",
            "updater",
            "updater_state_heads",
            "updater_state_heads_xy",
            "updater_state_heads_xy_collision",
            "updater_state_heads_xy_collision_node",
            "updater_mean",
            "updater_mean_y",
            "fast_roi",
            "state_dynamics",
            "state_roi",
            "state_relation_roi",
            "state_dynamics_fast_roi",
            "state_dynamics_roi",
            "differentiable_state_estimator",
        }
        if self.training.closed_loop_trainable_scope not in valid_closed_loop_scopes:
            raise ValueError(
                "training.closed_loop_trainable_scope must be "
                "'all', 'attention', 'attention_relation', 'attention_node_x', "
                "'attention_node_y', 'attention_node_z', 'dynamics', "
                "'updater', 'updater_state_heads', 'updater_state_heads_xy', "
                "'updater_state_heads_xy_collision', 'updater_mean', "
                "'updater_state_heads_xy_collision_node', "
                "'updater_mean_y', 'fast_roi', "
                "'state_dynamics', 'state_roi', 'state_relation_roi', "
                "'state_dynamics_fast_roi', 'state_dynamics_roi', or "
                "'differentiable_state_estimator'"
            )
        late_scope = self.training.closed_loop_late_trainable_scope
        transition_steps = self.training.closed_loop_scope_transition_steps
        configured_scopes = {self.training.closed_loop_trainable_scope, late_scope}
        soft_temperature = self.training.closed_loop_soft_association_temperature
        soft_posterior_straight_through = (
            self.training.closed_loop_soft_posterior_straight_through_enabled
        )
        if not isinstance(soft_posterior_straight_through, bool):
            raise ValueError(
                "training.closed_loop_soft_posterior_straight_through_enabled must be boolean"
            )
        if soft_temperature is not None and (
            isinstance(soft_temperature, bool)
            or not isinstance(soft_temperature, (int, float))
            or not math.isfinite(soft_temperature)
            or soft_temperature <= 0.0
        ):
            raise ValueError(
                "training.closed_loop_soft_association_temperature must be null "
                "or finite and positive"
            )
        soft_weight_names = (
            "soft_association_state",
            "soft_association_velocity",
            "soft_association_exclusivity",
            "soft_shadow_physical",
        )
        soft_weights = [
            float(self.training.loss_weights.get(name, 0.0)) for name in soft_weight_names
        ]
        if any(weight > 0.0 for weight in soft_weights):
            if soft_temperature is None:
                raise ValueError(
                    "positive soft-association loss weights require "
                    "training.closed_loop_soft_association_temperature"
                )
            if "differentiable_state_estimator" not in configured_scopes:
                raise ValueError(
                    "soft-association objectives require the "
                    "differentiable_state_estimator trainable scope"
                )
        if soft_posterior_straight_through:
            if soft_temperature is None:
                raise ValueError(
                    "soft posterior straight-through requires "
                    "training.closed_loop_soft_association_temperature"
                )
            if "differentiable_state_estimator" not in configured_scopes:
                raise ValueError(
                    "soft posterior straight-through requires the "
                    "differentiable_state_estimator trainable scope"
                )
        if float(self.training.loss_weights.get("rgb_reprojection", 0.0)) > 0.0:
            if "differentiable_state_estimator" not in configured_scopes:
                raise ValueError(
                    "RGB reprojection requires the differentiable_state_estimator trainable scope"
                )
            if self.runtime.modality != "rgb":
                raise ValueError("RGB reprojection requires runtime.modality=rgb")
        if (late_scope is None) != (transition_steps is None):
            raise ValueError(
                "training.closed_loop_late_trainable_scope and "
                "closed_loop_scope_transition_steps must be configured together"
            )
        if late_scope is not None and late_scope not in valid_closed_loop_scopes:
            raise ValueError("training.closed_loop_late_trainable_scope is invalid")
        event_loss_weights = self.training.closed_loop_event_loss_weights
        if any(
            not isinstance(scope, str) or scope not in valid_closed_loop_scopes
            for scope in event_loss_weights
        ) or any(
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or weight < 0.0
            for weight in event_loss_weights.values()
        ):
            raise ValueError(
                "training.closed_loop_event_loss_weights must map valid scopes "
                "to finite nonnegative weights"
            )
        state_event_loss_weights = self.training.closed_loop_state_event_loss_weights
        state_event_scope = "updater_state_heads_xy_collision_node"
        if any(scope != state_event_scope for scope in state_event_loss_weights) or any(
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or weight < 0.0
            for weight in state_event_loss_weights.values()
        ):
            raise ValueError(
                "training.closed_loop_state_event_loss_weights must map only "
                "updater_state_heads_xy_collision_node to a finite nonnegative weight"
            )
        if state_event_loss_weights.get(state_event_scope, 0.0) > 0.0 and (
            state_event_scope not in configured_scopes
            or event_loss_weights.get(state_event_scope, 0.0) <= 0.0
        ):
            raise ValueError(
                "positive state-event routing requires the configured node-collision "
                "scope and its positive direct event-loss override"
            )
        if (
            "state_relation_roi" in configured_scopes
            and not self.model.dynamics.attention_residual_enabled
        ):
            raise ValueError(
                "state_relation_roi scope requires model.dynamics.attention_residual_enabled=true"
            )
        direct_collision_scopes = {
            "updater_state_heads_xy_collision",
            "updater_state_heads_xy_collision_node",
        }
        if (
            configured_scopes & direct_collision_scopes
            and not self.model.dynamics.attention_residual_enabled
        ):
            raise ValueError(
                "direct collision-owner scope requires "
                "model.dynamics.attention_residual_enabled=true"
            )
        for scope in configured_scopes & direct_collision_scopes:
            if event_loss_weights.get(scope, 0.0) <= 0.0:
                raise ValueError(
                    f"{scope} scope requires a positive exact closed_loop event-loss override"
                )
        if transition_steps is not None and (
            isinstance(transition_steps, bool)
            or not isinstance(transition_steps, int)
            or transition_steps <= 0
        ):
            raise ValueError(
                "training.closed_loop_scope_transition_steps must be a positive integer"
            )
        for name, value in (
            (
                "handoff_minimum_target_coverage",
                self.training.handoff_minimum_target_coverage,
            ),
            (
                "handoff_minimum_forecast_coverage",
                self.training.handoff_minimum_forecast_coverage,
            ),
            (
                "handoff_minimum_reference_coverage_ratio",
                self.training.handoff_minimum_reference_coverage_ratio,
            ),
        ):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"training.{name} must lie in [0, 1]")
        if (
            isinstance(self.training.maximum_no_gradient_batches_per_update, bool)
            or not isinstance(self.training.maximum_no_gradient_batches_per_update, int)
            or self.training.maximum_no_gradient_batches_per_update < 0
        ):
            raise ValueError(
                "training.maximum_no_gradient_batches_per_update must be a nonnegative integer"
            )
        if (
            not math.isfinite(self.training.minimum_effective_gradient_norm)
            or self.training.minimum_effective_gradient_norm < 0
        ):
            raise ValueError(
                "training.minimum_effective_gradient_norm must be finite and nonnegative"
            )
        if self.training.tbptt_steps <= 0:
            raise ValueError("training.tbptt_steps must be positive")
        if not math.isfinite(self.training.grad_clip_norm) or self.training.grad_clip_norm <= 0:
            raise ValueError("training.grad_clip_norm must be finite and positive")
        if (
            not math.isfinite(self.training.interaction_grad_clip_norm)
            or self.training.interaction_grad_clip_norm <= 0
        ):
            raise ValueError("training.interaction_grad_clip_norm must be finite and positive")
        minimum_interaction_retention = self.training.minimum_interaction_gradient_retention
        if minimum_interaction_retention is not None and (
            isinstance(minimum_interaction_retention, bool)
            or not isinstance(minimum_interaction_retention, (int, float))
            or not math.isfinite(minimum_interaction_retention)
            or minimum_interaction_retention <= 0
            or minimum_interaction_retention > 1
        ):
            raise ValueError(
                "training.minimum_interaction_gradient_retention must be in (0, 1] when configured"
            )
        for config_field in (
            "attention_node_grad_clip_norm",
            "attention_collision_grad_clip_norm",
            "attention_force_grad_clip_norm",
            "attention_impulse_grad_clip_norm",
        ):
            value = getattr(self.training, config_field)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(
                    f"training.{config_field} must be finite and positive when configured"
                )
        for config_field in (
            "attention_node_output_grad_clip_norm",
            "attention_collision_output_grad_clip_norm",
            "attention_force_output_grad_clip_norm",
            "attention_impulse_output_grad_clip_norm",
        ):
            value = getattr(self.training, config_field)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(
                    f"training.{config_field} must be finite and positive when configured"
                )
        if (
            not math.isfinite(self.training.closed_loop_perception_grad_clip_norm)
            or self.training.closed_loop_perception_grad_clip_norm <= 0
        ):
            raise ValueError(
                "training.closed_loop_perception_grad_clip_norm must be finite and positive"
            )
        if self.training.measurement_validation_frames <= 0:
            raise ValueError("training.measurement_validation_frames must be positive")
        if (
            not math.isfinite(self.training.fast_roi_pretrain_weight)
            or self.training.fast_roi_pretrain_weight <= 0
        ):
            raise ValueError("training.fast_roi_pretrain_weight must be finite and positive")
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
        if not isinstance(
            self.training.normalize_rollout_axes_over_configured_horizons,
            bool,
        ):
            raise ValueError(
                "training.normalize_rollout_axes_over_configured_horizons must be boolean"
            )
        if not isinstance(self.training.joint_collision_long_horizon_sampling, bool):
            raise ValueError("training.joint_collision_long_horizon_sampling must be boolean")
        if self.training.minimum_rollout_age_steps < 0:
            raise ValueError("training.minimum_rollout_age_steps must be nonnegative")
        maximum_rollout_offset = max(quantized_horizons)
        if (
            self.training.minimum_rollout_age_steps + maximum_rollout_offset
            >= simulator.sequence_frames
        ):
            raise ValueError(
                "simulator.sequence_frames must exceed "
                "training.minimum_rollout_age_steps plus the maximum forecast offset"
            )
        if (
            self.training.rollout_anchors_per_window is not None
            and self.training.rollout_anchors_per_window <= 0
        ):
            raise ValueError("training.rollout_anchors_per_window must be positive or null")
        if not isinstance(
            self.training.closed_loop_prior_future_correction_enabled,
            bool,
        ):
            raise ValueError("training.closed_loop_prior_future_correction_enabled must be boolean")
        for field_name in (
            "closed_loop_batch_macro_physical_losses_enabled",
            "closed_loop_axiswise_correction_hinge_enabled",
            "closed_loop_modular_gradient_ownership_enabled",
        ):
            if not isinstance(getattr(self.training, field_name), bool):
                raise ValueError(f"training.{field_name} must be boolean")
        if (
            self.training.closed_loop_modular_gradient_ownership_enabled
            and "differentiable_state_estimator" not in configured_scopes
        ):
            raise ValueError(
                "modular gradient ownership requires the "
                "differentiable_state_estimator trainable scope"
            )
        scenario_tail_fraction = self.training.closed_loop_scenario_tail_fraction
        if scenario_tail_fraction is not None:
            if (
                isinstance(scenario_tail_fraction, bool)
                or not isinstance(scenario_tail_fraction, (int, float))
                or not math.isfinite(float(scenario_tail_fraction))
                or not 0.0 < float(scenario_tail_fraction) <= 1.0
            ):
                raise ValueError(
                    "training.closed_loop_scenario_tail_fraction must be null or a "
                    "finite number in (0, 1]"
                )
            if not self.training.scenario_balanced_batches:
                raise ValueError(
                    "training.closed_loop_scenario_tail_fraction requires "
                    "scenario_balanced_batches=true"
                )
            if self.training.batch_size != len(self.simulator.scenario_mixture):
                raise ValueError(
                    "training.closed_loop_scenario_tail_fraction requires exactly one "
                    "episode per declared scenario in each batch"
                )
            if not self.training.closed_loop_batch_macro_physical_losses_enabled:
                raise ValueError(
                    "training.closed_loop_scenario_tail_fraction requires "
                    "closed_loop_batch_macro_physical_losses_enabled=true"
                )
            if not self.training.closed_loop_axiswise_correction_hinge_enabled:
                raise ValueError(
                    "training.closed_loop_scenario_tail_fraction requires "
                    "closed_loop_axiswise_correction_hinge_enabled=true"
                )
        uncertainty_gradient_cap = (
            self.training.closed_loop_uncertainty_standardized_error_gradient_cap
        )
        if uncertainty_gradient_cap is not None and (
            isinstance(uncertainty_gradient_cap, bool)
            or not isinstance(uncertainty_gradient_cap, (int, float))
            or not math.isfinite(float(uncertainty_gradient_cap))
            or float(uncertainty_gradient_cap) <= 0.0
        ):
            raise ValueError(
                "training.closed_loop_uncertainty_standardized_error_gradient_cap "
                "must be null or a finite positive number"
            )
        protected_reference_weight = (
            self.training.closed_loop_protected_reference_nonregression_weight
        )
        if (
            isinstance(protected_reference_weight, bool)
            or not isinstance(protected_reference_weight, (int, float))
            or not math.isfinite(float(protected_reference_weight))
            or float(protected_reference_weight) < 0.0
        ):
            raise ValueError(
                "training.closed_loop_protected_reference_nonregression_weight "
                "must be a finite nonnegative number"
            )
        if protected_reference_weight > 0.0:
            if self.training.rgb_pretrain_steps != 0:
                raise ValueError("protected-reference non-regression requires rgb_pretrain_steps=0")
            if not self.training.scenario_balanced_batches:
                raise ValueError(
                    "protected-reference non-regression requires scenario_balanced_batches=true"
                )
            if self.training.batch_size != len(self.simulator.scenario_mixture):
                raise ValueError(
                    "protected-reference non-regression requires exactly one episode "
                    "per declared scenario in each batch"
                )
            if not self.training.closed_loop_batch_macro_physical_losses_enabled:
                raise ValueError(
                    "protected-reference non-regression requires "
                    "closed_loop_batch_macro_physical_losses_enabled=true"
                )
            if not self.training.closed_loop_axiswise_correction_hinge_enabled:
                raise ValueError(
                    "protected-reference non-regression requires "
                    "closed_loop_axiswise_correction_hinge_enabled=true"
                )
            if self.model.dynamics.attention_dropout != 0.0:
                raise ValueError(
                    "protected-reference non-regression requires attention_dropout=0 "
                    "so candidate/reference replay is deterministic"
                )
        if (
            self.training.validation_rollout_anchors_per_episode is not None
            and self.training.validation_rollout_anchors_per_episode <= 0
        ):
            raise ValueError(
                "training.validation_rollout_anchors_per_episode must be positive or null"
            )
        if (
            isinstance(self.training.validation_rollout_anchor_batch_size, bool)
            or not isinstance(self.training.validation_rollout_anchor_batch_size, int)
            or self.training.validation_rollout_anchor_batch_size <= 0
        ):
            raise ValueError(
                "training.validation_rollout_anchor_batch_size must be a positive integer"
            )
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
