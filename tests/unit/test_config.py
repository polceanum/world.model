from dataclasses import replace
from pathlib import Path

import pytest
import torch

from world_model.datasets.splits import SPLIT_SEED_RANGES
from world_model.dynamics import DynamicsModel
from world_model.observations.rgb.structured_centres import structured_disc_centres
from world_model.runtime import OnlineWorldModel
from world_model.simulator.physics import PhysicsConfig
from world_model.utils.config import OrpheusConfig, load_config, save_resolved_config

CONFIG_DIR = Path(__file__).parents[2] / "configs"


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.yaml")))
def test_profiles_resolve_and_validate(path: Path) -> None:
    config = load_config(path)
    assert config.model.max_objects >= config.simulator.max_objects
    assert config.runtime.modality == "rgb"
    assert config.evaluation.rgb_only
    test_lower, test_upper = SPLIT_SEED_RANGES["test"]
    assert test_lower <= config.demo.seed <= test_upper
    assert config.simulator.split_validation_start == SPLIT_SEED_RANGES["validation"][0]
    assert config.simulator.split_test_start == test_lower
    assert config.simulator.split_ood_start == SPLIT_SEED_RANGES["ood"][0]


def test_soft_association_requires_positive_temperature_and_differentiable_scope() -> None:
    base = load_config(CONFIG_DIR / "tiny_overfit.yaml")
    weights = {
        **base.training.loss_weights,
        "soft_association_state": 1.0,
        "soft_association_exclusivity": 0.05,
    }
    enabled = replace(
        base,
        training=replace(
            base.training,
            closed_loop_trainable_scope="differentiable_state_estimator",
            closed_loop_soft_association_temperature=0.5,
            loss_weights=weights,
        ),
    )
    enabled.validate()

    with pytest.raises(ValueError, match="require.*temperature"):
        replace(
            enabled,
            training=replace(
                enabled.training,
                closed_loop_soft_association_temperature=None,
            ),
        ).validate()
    with pytest.raises(ValueError, match="differentiable_state_estimator"):
        replace(
            enabled,
            training=replace(enabled.training, closed_loop_trainable_scope="updater"),
        ).validate()


@pytest.mark.parametrize("value", [True, 0.0, -0.5, float("nan"), float("inf"), "0.5"])
def test_soft_association_temperature_is_strictly_positive_finite_real(value: object) -> None:
    base = load_config(CONFIG_DIR / "tiny_overfit.yaml")
    with pytest.raises(ValueError, match="soft_association_temperature"):
        replace(
            base,
            training=replace(
                base.training,
                closed_loop_soft_association_temperature=value,  # type: ignore[arg-type]
            ),
        ).validate()


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_soft_posterior_straight_through_requires_strict_boolean(value: object) -> None:
    base = OrpheusConfig()
    with pytest.raises(ValueError, match="soft_posterior_straight_through"):
        replace(
            base,
            training=replace(
                base.training,
                closed_loop_soft_posterior_straight_through_enabled=value,  # type: ignore[arg-type]
            ),
        ).validate()


def test_soft_posterior_straight_through_requires_temperature_and_scope() -> None:
    base = OrpheusConfig()
    enabled = replace(
        base.training,
        closed_loop_soft_posterior_straight_through_enabled=True,
    )
    with pytest.raises(ValueError, match="requires.*temperature"):
        replace(base, training=enabled).validate()
    with pytest.raises(ValueError, match="requires.*trainable scope"):
        replace(
            base,
            training=replace(
                enabled,
                closed_loop_soft_association_temperature=0.5,
            ),
        ).validate()

    replace(
        base,
        training=replace(
            enabled,
            closed_loop_soft_association_temperature=0.5,
            closed_loop_trainable_scope="differentiable_state_estimator",
        ),
    ).validate()


def test_rgb_reprojection_requires_differentiable_rgb_state_estimator() -> None:
    base = OrpheusConfig()
    weights = {**base.training.loss_weights, "rgb_reprojection": 0.25}
    with pytest.raises(ValueError, match="differentiable_state_estimator"):
        replace(
            base,
            training=replace(base.training, loss_weights=weights),
        ).validate()

    replace(
        base,
        training=replace(
            base.training,
            closed_loop_trainable_scope="differentiable_state_estimator",
            loss_weights=weights,
        ),
    ).validate()


def test_sustained_v3_analytic_contacts_match_reference_solver_thresholds() -> None:
    config = load_config(CONFIG_DIR / "sustained_accuracy_mps_v3.yaml")
    dynamics = config.model.dynamics
    physics = PhysicsConfig()

    assert dynamics.contact_margin == 0.0
    assert dynamics.boundary_contact_tolerance == pytest.approx(1.0e-4)
    assert dynamics.penetration_slop == pytest.approx(physics.penetration_slop)
    assert dynamics.max_penetration_correction == pytest.approx(physics.max_position_correction)
    assert dynamics.contact_confidence_sigma == 0.0
    assert dynamics.pair_collision_speed_epsilon == pytest.approx(1.0e-7)


def test_innovation_anchored_correction_is_explicit_protocol_semantics() -> None:
    legacy = load_config(CONFIG_DIR / "toy_smoke.yaml")
    corrected = load_config(CONFIG_DIR / "sustained_accuracy_balanced_mps.yaml")

    assert not legacy.model.filter.innovation_anchored_correction
    assert corrected.model.filter.innovation_anchored_correction


def test_learned_correction_axis_support_is_explicit_legacy_false_semantics() -> None:
    legacy = load_config(CONFIG_DIR / "toy_smoke.yaml")
    corrected = load_config(
        CONFIG_DIR / "toy_smoke.yaml",
        overrides=["model.filter.learned_correction_independent_axis_support=true"],
    )

    assert not legacy.model.filter.learned_correction_independent_axis_support
    assert corrected.model.filter.learned_correction_independent_axis_support
    assert OnlineWorldModel.from_config(
        corrected
    ).updater.config.learned_correction_independent_axis_support


@pytest.mark.parametrize("value", ["1", "0", "null", "not-a-boolean"])
def test_learned_correction_axis_support_requires_boolean(value: str) -> None:
    with pytest.raises(
        ValueError,
        match="model.filter.learned_correction_independent_axis_support must be boolean",
    ):
        load_config(
            CONFIG_DIR / "toy_smoke.yaml",
            overrides=[f"model.filter.learned_correction_independent_axis_support={value}"],
        )


def test_dotted_override_is_typed(tmp_path: Path) -> None:
    config = load_config(
        CONFIG_DIR / "toy_smoke.yaml",
        overrides=["training.steps=3", "simulator.image_size=[32, 40]"],
    )
    assert config.training.steps == 3
    assert config.simulator.image_size == (32, 40)


def test_primary_evaluation_is_clean_and_recovery_is_explicit() -> None:
    clean = load_config(CONFIG_DIR / "default.yaml")
    recovery = load_config(
        CONFIG_DIR / "default.yaml",
        overrides=["evaluation.recovery_probe_enabled=true"],
    )

    assert not clean.evaluation.recovery_probe_enabled
    assert recovery.evaluation.recovery_probe_enabled


def test_ensured_pair_scene_resampling_is_explicit_and_validated() -> None:
    config = load_config(CONFIG_DIR / "default.yaml")

    assert config.simulator.ensured_pair_scene_resample_attempts == 32
    assert config.simulator.ensured_pair_event_frame_range is None
    assert config.simulator.ensured_pair_vertical_speed_range is None
    with pytest.raises(
        ValueError,
        match="simulator.ensured_pair_scene_resample_attempts must be a positive integer",
    ):
        load_config(
            CONFIG_DIR / "default.yaml",
            overrides=["simulator.ensured_pair_scene_resample_attempts=0"],
        )


@pytest.mark.parametrize(
    "value",
    ("[0,20]", "[20,72]", "[20,19]", "[true,20]", "[20.0,24]"),
)
def test_ensured_pair_event_frame_range_is_strict_and_bounded(value: str) -> None:
    with pytest.raises(
        ValueError,
        match="simulator.ensured_pair_event_frame_range must be null or an increasing",
    ):
        load_config(
            CONFIG_DIR / "default.yaml",
            overrides=[
                f"simulator.ensured_pair_event_frame_range={value}",
            ],
        )

    configured = load_config(
        CONFIG_DIR / "default.yaml",
        overrides=[
            "simulator.ensured_pair_event_frame_range=[20,30]",
        ],
    )
    assert configured.simulator.ensured_pair_event_frame_range == (20, 30)


@pytest.mark.parametrize("value", ("[-1,2]", "[2,1]", "[true,2]", "[nan,2]"))
def test_ensured_pair_vertical_speed_range_is_strict(value: str) -> None:
    with pytest.raises(
        ValueError,
        match="simulator.ensured_pair_vertical_speed_range must be null or a finite",
    ):
        load_config(
            CONFIG_DIR / "default.yaml",
            overrides=[f"simulator.ensured_pair_vertical_speed_range={value}"],
        )

    configured = load_config(
        CONFIG_DIR / "default.yaml",
        overrides=["simulator.ensured_pair_vertical_speed_range=[4.7,5.1]"],
    )
    assert configured.simulator.ensured_pair_vertical_speed_range == (4.7, 5.1)


def test_legacy_contaminating_evaluation_mode_is_rejected() -> None:
    with pytest.raises(KeyError, match="apply_perturbations"):
        load_config(
            CONFIG_DIR / "default.yaml",
            overrides=["evaluation.apply_perturbations=true"],
        )


def test_axis_composition_is_configured_only_for_attention_pilot() -> None:
    default = load_config(CONFIG_DIR / "toy_smoke.yaml")
    attention = load_config(CONFIG_DIR / "attention_pilot_mps.yaml")
    assert not default.evaluation.hypothesis_axis_independent
    assert attention.evaluation.hypothesis_axis_independent
    assert attention.evaluation.hypothesis_axis_independent_axes == (0,)
    assert not attention.runtime.hypothesis_pool_enabled
    assert attention.runtime.hypothesis_evidence_horizons_seconds == (0.05,)
    assert attention.runtime.hypothesis_axis_independent_axes == (0,)
    assert not attention.runtime.hypothesis_local_applicability_enabled


@pytest.mark.parametrize(
    "override",
    [
        "runtime.hypothesis_local_applicability_enabled=1",
        "runtime.hypothesis_minimum_support_count=0",
        "runtime.hypothesis_minimum_support_count=true",
        "runtime.hypothesis_minimum_support_count=1.5",
        "runtime.hypothesis_maximum_evidence_age_seconds=-1",
        "runtime.hypothesis_maximum_evidence_age_seconds=true",
        "runtime.hypothesis_maximum_evidence_age_seconds=not-a-number",
        "runtime.hypothesis_minimum_observability=1.1",
        "runtime.hypothesis_minimum_observability=false",
        "runtime.hypothesis_minimum_confidence_margin=1.1",
        "runtime.hypothesis_minimum_confidence_margin=true",
        "runtime.hypothesis_velocity_evidence_weight=-1",
        "runtime.hypothesis_velocity_evidence_weight=true",
        "runtime.hypothesis_velocity_evidence_weight=not-a-number",
        "runtime.hypothesis_velocity_nonregression_gate_enabled=1",
        "runtime.hypothesis_residual_correction_gain_by_axis=[0.1,0.2]",
        "runtime.hypothesis_residual_correction_gain_by_axis=[0.1,true,0.0]",
        "runtime.hypothesis_residual_correction_gain_by_axis=[0.1,1.1,0.0]",
        "runtime.hypothesis_robust_influence_delta=-1",
        "runtime.hypothesis_robust_influence_delta=true",
        "runtime.hypothesis_composition_step_seconds=-1",
        "runtime.hypothesis_composition_step_seconds=0.05",
        "runtime.hypothesis_online_acceleration_enabled=1",
        "runtime.hypothesis_online_acceleration_minimum_support_count=0",
        "runtime.hypothesis_online_acceleration_minimum_support_count=true",
        "runtime.hypothesis_online_acceleration_maximum_mps2=0",
        "runtime.hypothesis_online_acceleration_maximum_mps2=true",
    ],
)
def test_runtime_hypothesis_applicability_controls_are_strict(override: str) -> None:
    with pytest.raises(ValueError):
        load_config(CONFIG_DIR / "toy_smoke.yaml", overrides=[override])


def test_runtime_hypothesis_composition_step_must_have_matching_local_evidence() -> None:
    config = load_config(
        CONFIG_DIR / "toy_smoke.yaml",
        overrides=[
            "runtime.hypothesis_local_applicability_enabled=true",
            "runtime.hypothesis_composition_step_seconds=0.05",
        ],
    )
    assert config.runtime.hypothesis_composition_step_seconds == pytest.approx(0.05)


def test_runtime_hypothesis_residual_correction_requires_matching_local_axes() -> None:
    config = load_config(
        CONFIG_DIR / "toy_smoke.yaml",
        overrides=[
            "runtime.hypothesis_local_applicability_enabled=true",
            "runtime.hypothesis_axis_independent_axes=[0,1]",
            "runtime.hypothesis_residual_correction_gain_by_axis=[0.25,0.5,0.0]",
        ],
    )
    assert config.runtime.hypothesis_residual_correction_gain_by_axis == (0.25, 0.5, 0.0)

    with pytest.raises(ValueError, match="independently configured"):
        load_config(
            CONFIG_DIR / "toy_smoke.yaml",
            overrides=[
                "runtime.hypothesis_local_applicability_enabled=true",
                "runtime.hypothesis_axis_independent_axes=[0]",
                "runtime.hypothesis_residual_correction_gain_by_axis=[0.25,0.5,0.0]",
            ],
        )


def test_runtime_online_acceleration_requires_rgb_velocity_and_local_applicability() -> None:
    with pytest.raises(ValueError, match="local hypothesis applicability"):
        load_config(
            CONFIG_DIR / "toy_smoke.yaml",
            overrides=[
                "model.rgb.temporal_velocity_enabled=true",
                "runtime.hypothesis_online_acceleration_enabled=true",
            ],
        )
    with pytest.raises(ValueError, match="causal RGB temporal velocity"):
        load_config(
            CONFIG_DIR / "toy_smoke.yaml",
            overrides=[
                "runtime.hypothesis_local_applicability_enabled=true",
                "runtime.hypothesis_online_acceleration_enabled=true",
            ],
        )
    config = load_config(
        CONFIG_DIR / "toy_smoke.yaml",
        overrides=[
            "model.rgb.temporal_velocity_enabled=true",
            "runtime.hypothesis_local_applicability_enabled=true",
            "runtime.hypothesis_online_acceleration_enabled=true",
            "runtime.hypothesis_online_acceleration_minimum_support_count=3",
            "runtime.hypothesis_online_acceleration_maximum_mps2=12.0",
        ],
    )
    assert config.runtime.hypothesis_online_acceleration_enabled
    assert config.runtime.hypothesis_online_acceleration_minimum_support_count == 3
    assert config.runtime.hypothesis_online_acceleration_maximum_mps2 == pytest.approx(12.0)
    with pytest.raises(ValueError, match="match a supported evidence horizon"):
        load_config(
            CONFIG_DIR / "toy_smoke.yaml",
            overrides=[
                "runtime.hypothesis_local_applicability_enabled=true",
                "runtime.hypothesis_composition_step_seconds=0.04",
            ],
        )
    with pytest.raises(ValueError, match="null or finite and positive"):
        load_config(
            CONFIG_DIR / "toy_smoke.yaml",
            overrides=[
                "runtime.hypothesis_local_applicability_enabled=true",
                "runtime.hypothesis_composition_step_seconds=true",
            ],
        )


def test_simulator_scenario_mixture_is_typed_and_validated() -> None:
    config = load_config(
        CONFIG_DIR / "toy_smoke.yaml",
        overrides=[
            "simulator.scenario_mixture=[elastic_pairs,damped_contacts,impulse_perturbation]",
            "simulator.initial_speed_range=[0.2,1.8]",
            "simulator.external_impulse_range=[0.1,0.9]",
            "training.validation_episodes=3",
        ],
    )
    assert config.simulator.scenario_mixture == (
        "elastic_pairs",
        "damped_contacts",
        "impulse_perturbation",
    )
    assert config.simulator.initial_speed_range == (0.2, 1.8)
    assert config.simulator.external_impulse_range == (0.1, 0.9)

    with pytest.raises(ValueError, match="unsupported simulator scenarios"):
        load_config(
            CONFIG_DIR / "toy_smoke.yaml",
            overrides=["simulator.scenario_mixture=[unknown]"],
        )

    with pytest.raises(ValueError, match="validation_episodes must cover every"):
        load_config(
            CONFIG_DIR / "toy_smoke.yaml",
            overrides=[
                "simulator.scenario_mixture=[baseline,elastic_pairs,damped_contacts]",
                "training.validation_episodes=2",
            ],
        )


def test_temporal_rgb_velocity_is_opt_in_and_typed() -> None:
    default = load_config(CONFIG_DIR / "toy_smoke.yaml")
    assert not default.model.rgb.temporal_velocity_enabled
    assert default.model.rgb.temporal_velocity_history_size == 3
    assert default.model.rgb.temporal_velocity_min_samples == 3
    assert default.model.rgb.temporal_velocity_variance_ceiling is None
    assert not default.model.rgb.temporal_velocity_lateral_only
    assert not default.model.rgb.temporal_velocity_independent_raw_history_enabled
    assert not default.model.rgb.temporal_velocity_continuous_gravity_axis_enabled
    assert default.model.rgb.temporal_velocity_unobserved_variance == 1.0e4
    assert not default.model.rgb.temporal_velocity_reset_on_collision
    assert default.model.rgb.temporal_velocity_max_age_steps is None
    assert default.model.rgb.temporal_velocity_measurement_position_blend == 0.0
    assert not default.model.rgb.temporal_velocity_position_innovation_coupling
    assert not default.model.rgb.temporal_position_enabled
    assert default.model.rgb.temporal_position_min_samples == 3
    assert default.model.rgb.temporal_position_robust_threshold == 2.5
    assert default.model.rgb.temporal_position_variance_scale == 4.0
    assert default.model.rgb.temporal_position_variance_floor == 0.01
    assert default.model.rgb.temporal_position_variance_ceiling is None
    assert default.model.rgb.temporal_position_depth_only
    assert default.model.rgb.structured_disc_center_enabled
    assert default.model.rgb.structured_disc_depth_outlier_relative_threshold is None

    enabled = load_config(
        CONFIG_DIR / "toy_smoke.yaml",
        overrides=[
            "model.rgb.temporal_velocity_enabled=true",
            "model.rgb.temporal_velocity_history_size=4",
            "model.rgb.temporal_velocity_min_samples=2",
            "model.rgb.temporal_velocity_variance_scale=3.0",
            "model.rgb.temporal_velocity_variance_floor=0.4",
            "model.rgb.temporal_velocity_variance_ceiling=2.0",
            "model.rgb.temporal_velocity_lateral_only=true",
            "model.rgb.temporal_velocity_independent_raw_history_enabled=true",
            "model.rgb.temporal_velocity_continuous_gravity_axis_enabled=true",
            "model.rgb.temporal_velocity_unobserved_variance=1000.0",
            "model.rgb.temporal_velocity_reset_on_collision=true",
            "model.rgb.temporal_velocity_max_age_steps=3",
            "model.rgb.temporal_velocity_measurement_position_blend=0.25",
            "model.rgb.temporal_velocity_position_innovation_coupling=true",
            "model.rgb.temporal_position_enabled=true",
            "model.rgb.temporal_position_min_samples=2",
            "model.rgb.temporal_position_robust_threshold=2.0",
            "model.rgb.temporal_position_variance_scale=3.0",
            "model.rgb.temporal_position_variance_floor=0.02",
            "model.rgb.temporal_position_variance_ceiling=0.5",
            "model.rgb.temporal_position_depth_only=false",
            "model.rgb.structured_disc_depth_outlier_relative_threshold=0.12",
            "model.rgb.structured_disc_depth_outlier_variance_scale=9.0",
        ],
    )
    assert enabled.model.rgb.temporal_velocity_enabled
    assert enabled.model.rgb.temporal_velocity_history_size == 4
    assert enabled.model.rgb.temporal_velocity_min_samples == 2
    assert enabled.model.rgb.temporal_velocity_variance_scale == 3.0
    assert enabled.model.rgb.temporal_velocity_variance_floor == 0.4
    assert enabled.model.rgb.temporal_velocity_variance_ceiling == 2.0
    assert enabled.model.rgb.temporal_velocity_lateral_only
    assert enabled.model.rgb.temporal_velocity_independent_raw_history_enabled
    assert enabled.model.rgb.temporal_velocity_continuous_gravity_axis_enabled
    assert enabled.model.rgb.temporal_velocity_unobserved_variance == 1000.0
    assert enabled.model.rgb.temporal_velocity_reset_on_collision
    assert enabled.model.rgb.temporal_velocity_max_age_steps == 3
    assert enabled.model.rgb.temporal_velocity_measurement_position_blend == 0.25
    assert enabled.model.rgb.temporal_velocity_position_innovation_coupling
    assert enabled.model.rgb.temporal_position_enabled
    assert enabled.model.rgb.temporal_position_min_samples == 2
    assert enabled.model.rgb.temporal_position_robust_threshold == 2.0
    assert enabled.model.rgb.temporal_position_variance_scale == 3.0
    assert enabled.model.rgb.temporal_position_variance_floor == 0.02
    assert enabled.model.rgb.temporal_position_variance_ceiling == 0.5
    assert not enabled.model.rgb.temporal_position_depth_only
    assert enabled.model.rgb.structured_disc_depth_outlier_relative_threshold == 0.12
    assert enabled.model.rgb.structured_disc_depth_outlier_variance_scale == 9.0

    with pytest.raises(ValueError, match="continuous gravity-axis velocity"):
        load_config(
            CONFIG_DIR / "toy_smoke.yaml",
            overrides=[
                "model.rgb.temporal_velocity_continuous_gravity_axis_enabled=true",
            ],
        )

    with pytest.raises(ValueError, match="independent raw RGB history"):
        load_config(
            CONFIG_DIR / "toy_smoke.yaml",
            overrides=[
                "model.rgb.temporal_velocity_independent_raw_history_enabled=true",
            ],
        )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("global_every_steps", 0),
        ("temporal_velocity_min_dt", 0.0),
        ("temporal_velocity_history_size", 2),
        ("temporal_velocity_min_samples", 4),
        ("temporal_velocity_variance_scale", 0.5),
        ("temporal_velocity_variance_floor", 0.0),
        ("temporal_velocity_variance_ceiling", 0.1),
        ("temporal_velocity_unobserved_variance", 0.1),
        ("temporal_velocity_max_age_steps", 1),
        ("temporal_velocity_measurement_position_blend", 1.1),
        ("structured_disc_depth_outlier_relative_threshold", 0.0),
        ("structured_disc_depth_outlier_variance_scale", 0.5),
    ],
)
def test_temporal_rgb_velocity_uncertainty_config_is_bounded(
    key: str,
    value: float | int,
) -> None:
    with pytest.raises(ValueError, match=key):
        load_config(
            CONFIG_DIR / "toy_smoke.yaml",
            overrides=[f"model.rgb.{key}={value}"],
        )


def test_unknown_key_fails(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("project:\n  mystery: true\n", encoding="utf-8")
    with pytest.raises(KeyError, match="mystery"):
        load_config(bad)


def test_oracle_cannot_hide_in_rgb_evaluation(tmp_path: Path) -> None:
    bad = tmp_path / "oracle.yaml"
    bad.write_text(
        "runtime:\n  modality: debug_oracle\n  enable_debug_oracle: true\n"
        "evaluation:\n  rgb_only: true\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="RGB-only"):
        load_config(bad)


@pytest.mark.parametrize("scale", [0.0, -0.1, 1.1])
def test_closed_loop_learning_rate_scale_is_bounded(scale: float) -> None:
    with pytest.raises(ValueError, match="closed_loop_learning_rate_scale"):
        load_config(
            CONFIG_DIR / "tiny_overfit.yaml",
            overrides=[f"training.closed_loop_learning_rate_scale={scale}"],
        )


@pytest.mark.parametrize(
    "overrides,pattern",
    [
        (["training.closed_loop_learning_rate_schedule=unknown"], "schedule"),
        (["training.closed_loop_learning_rate_warmup_steps=-1"], "warmup_steps"),
        (["training.closed_loop_learning_rate_minimum_scale=-0.1"], "minimum_scale"),
        (["training.closed_loop_learning_rate_minimum_scale=1.1"], "minimum_scale"),
        (
            [
                "training.closed_loop_learning_rate_schedule=constant",
                "training.closed_loop_learning_rate_warmup_steps=1",
            ],
            "constant",
        ),
        (
            [
                "training.closed_loop_learning_rate_schedule=warmup_cosine",
                "training.closed_loop_learning_rate_cosine_decay_steps=4",
            ],
            "warmup steps",
        ),
        (
            [
                "training.closed_loop_learning_rate_schedule=warmup_cosine",
                "training.closed_loop_learning_rate_warmup_steps=4",
                "training.closed_loop_learning_rate_cosine_decay_steps=0",
            ],
            "decay steps",
        ),
    ],
)
def test_closed_loop_learning_rate_schedule_is_validated(
    overrides: list[str],
    pattern: str,
) -> None:
    with pytest.raises(ValueError, match=pattern):
        load_config(CONFIG_DIR / "tiny_overfit.yaml", overrides=overrides)


def test_closed_loop_device_preference_is_explicit() -> None:
    with pytest.raises(ValueError, match="closed-loop device preference"):
        load_config(
            CONFIG_DIR / "tiny_overfit.yaml",
            overrides=["device.closed_loop_preference=tpu"],
        )


def test_global_detector_cpu_on_mps_requires_boolean() -> None:
    config = load_config(CONFIG_DIR / "toy_smoke.yaml")
    invalid = replace(
        config,
        device=replace(
            config.device,
            global_detector_cpu_on_mps="false",  # type: ignore[arg-type]
        ),
    )

    with pytest.raises(ValueError, match="global_detector_cpu_on_mps must be boolean"):
        invalid.validate()


def test_closed_loop_global_trainable_steps_is_nonnegative() -> None:
    with pytest.raises(ValueError, match="closed_loop_global_trainable_steps"):
        load_config(
            CONFIG_DIR / "tiny_overfit.yaml",
            overrides=["training.closed_loop_global_trainable_steps=-1"],
        )


def test_rgb_pretrain_trainable_scope_is_strict_and_legacy_all() -> None:
    legacy = load_config(CONFIG_DIR / "tiny_overfit.yaml")
    detector = load_config(
        CONFIG_DIR / "tiny_overfit.yaml",
        overrides=["training.rgb_pretrain_trainable_scope=global_detector"],
    )

    assert legacy.training.rgb_pretrain_trainable_scope == "all"
    assert detector.training.rgb_pretrain_trainable_scope == "global_detector"
    with pytest.raises(ValueError, match="rgb_pretrain_trainable_scope"):
        load_config(
            CONFIG_DIR / "tiny_overfit.yaml",
            overrides=["training.rgb_pretrain_trainable_scope=global_and_fast"],
        )


def test_rgb_pretrain_requires_a_positive_aggregate_measurement_weight() -> None:
    with pytest.raises(
        ValueError,
        match=r"loss_weights\.measurement must be positive.*rgb_pretrain_steps",
    ):
        load_config(
            CONFIG_DIR / "axis_gated_updater_repair_cpu.yaml",
            overrides=[
                "training.rgb_pretrain_steps=1",
                "training.rgb_pretrain_trainable_scope=global_detector",
            ],
        )


def test_closed_loop_trainable_scope_is_explicit() -> None:
    with pytest.raises(ValueError, match="closed_loop_trainable_scope"):
        load_config(
            CONFIG_DIR / "tiny_overfit.yaml",
            overrides=["training.closed_loop_trainable_scope=perception"],
        )

    config = load_config(
        CONFIG_DIR / "tiny_overfit.yaml",
        overrides=["training.closed_loop_trainable_scope=updater_mean_y"],
    )
    assert config.training.closed_loop_trainable_scope == "updater_mean_y"

    relation_config = load_config(
        CONFIG_DIR / "tiny_overfit.yaml",
        overrides=["training.closed_loop_trainable_scope=attention_relation"],
    )
    assert relation_config.training.closed_loop_trainable_scope == "attention_relation"

    state_updater_config = load_config(
        CONFIG_DIR / "tiny_overfit.yaml",
        overrides=["training.closed_loop_trainable_scope=updater_state_heads"],
    )
    assert state_updater_config.training.closed_loop_trainable_scope == "updater_state_heads"

    xy_state_updater_config = load_config(
        CONFIG_DIR / "tiny_overfit.yaml",
        overrides=["training.closed_loop_trainable_scope=updater_state_heads_xy"],
    )
    assert xy_state_updater_config.training.closed_loop_trainable_scope == "updater_state_heads_xy"

    collision_config = load_config(
        CONFIG_DIR / "tiny_overfit.yaml",
        overrides=[
            "model.dynamics.attention_residual_enabled=true",
            "training.closed_loop_trainable_scope=updater_state_heads_xy_collision",
            "training.closed_loop_event_loss_weights={updater_state_heads_xy_collision: 0.05}",
        ],
    )
    assert (
        collision_config.training.closed_loop_trainable_scope == "updater_state_heads_xy_collision"
    )

    node_collision_config = load_config(
        CONFIG_DIR / "tiny_overfit.yaml",
        overrides=[
            "model.dynamics.attention_residual_enabled=true",
            "training.closed_loop_trainable_scope=updater_state_heads_xy_collision_node",
            (
                "training.closed_loop_event_loss_weights="
                "{updater_state_heads_xy_collision_node: 0.0045}"
            ),
            (
                "training.closed_loop_state_event_loss_weights="
                "{updater_state_heads_xy_collision_node: 0.04}"
            ),
        ],
    )
    assert node_collision_config.training.closed_loop_trainable_scope == (
        "updater_state_heads_xy_collision_node"
    )
    assert node_collision_config.training.closed_loop_state_event_loss_weights == {
        "updater_state_heads_xy_collision_node": 0.04
    }


@pytest.mark.parametrize("value", (-1.0, float("inf"), float("nan"), True, "0.04"))
def test_state_event_routing_weight_is_strict(value: object) -> None:
    scope = "updater_state_heads_xy_collision_node"
    source = load_config(
        CONFIG_DIR / "tiny_overfit.yaml",
        overrides=[
            "model.dynamics.attention_residual_enabled=true",
            f"training.closed_loop_trainable_scope={scope}",
            f"training.closed_loop_event_loss_weights={{{scope}: 0.0045}}",
        ],
    )
    with pytest.raises(ValueError, match="closed_loop_state_event_loss_weights"):
        replace(
            source,
            training=replace(
                source.training,
                closed_loop_state_event_loss_weights={scope: value},
            ),
        ).validate()


def test_state_event_routing_requires_matching_node_collision_scope() -> None:
    with pytest.raises(ValueError, match="positive state-event routing requires"):
        load_config(
            CONFIG_DIR / "tiny_overfit.yaml",
            overrides=[
                (
                    "training.closed_loop_state_event_loss_weights="
                    "{updater_state_heads_xy_collision_node: 0.04}"
                )
            ],
        )


def test_combined_xy_collision_scope_requires_positive_event_weight() -> None:
    with pytest.raises(ValueError, match="requires a positive exact"):
        load_config(
            CONFIG_DIR / "tiny_overfit.yaml",
            overrides=[
                "model.dynamics.attention_residual_enabled=true",
                "training.closed_loop_trainable_scope=updater_state_heads_xy_collision",
                ("training.closed_loop_event_loss_weights={updater_state_heads_xy_collision: 0.0}"),
            ],
        )

    with pytest.raises(ValueError, match="requires a positive exact"):
        load_config(
            CONFIG_DIR / "tiny_overfit.yaml",
            overrides=[
                "model.dynamics.attention_residual_enabled=true",
                "training.closed_loop_trainable_scope=updater_state_heads_xy_collision_node",
                (
                    "training.closed_loop_event_loss_weights="
                    "{updater_state_heads_xy_collision_node: 0.0}"
                ),
            ],
        )


def test_updater_state_heads_scope_roundtrips_as_typed_configuration(tmp_path: Path) -> None:
    config = load_config(
        CONFIG_DIR / "tiny_overfit.yaml",
        overrides=["training.closed_loop_trainable_scope=updater_state_heads"],
    )
    resolved_path = tmp_path / "updater-state-resolved.yaml"

    save_resolved_config(config, resolved_path)
    restored = load_config(resolved_path)

    assert restored.training.closed_loop_trainable_scope == "updater_state_heads"
    assert restored.to_dict() == config.to_dict()


def test_axis_gated_updater_repair_profile_binds_the_paired_protocol() -> None:
    repaired = load_config(CONFIG_DIR / "axis_gated_updater_repair_cpu.yaml")
    control = load_config(
        CONFIG_DIR / "axis_gated_updater_repair_cpu.yaml",
        overrides=[
            "training.closed_loop_batch_macro_physical_losses_enabled=false",
            "training.closed_loop_axiswise_correction_hinge_enabled=false",
        ],
    )

    assert repaired.training.closed_loop_trainable_scope == "updater_state_heads"
    assert repaired.training.closed_loop_late_trainable_scope is None
    assert repaired.training.closed_loop_scope_transition_steps is None
    assert repaired.training.steps == 3072
    assert repaired.training.train_episodes == 3072 * 8
    assert repaired.training.closed_loop_learning_rate_warmup_steps == 256
    assert repaired.training.closed_loop_learning_rate_cosine_decay_steps == 2304
    assert repaired.training.eval_every == 512
    assert repaired.training.validation_episodes == 32
    assert repaired.training.horizon_weights == (1.0,) * 5
    assert repaired.training.closed_loop_event_loss_weights == {"updater_state_heads": 0.0}
    assert "correction" not in repaired.training.loss_weights
    assert repaired.training.loss_weights["correction_position"] == 7.0
    assert repaired.training.loss_weights["correction_velocity"] == 2.0
    assert repaired.training.loss_weights["correction_regularization"] == 0.0
    assert repaired.model.filter.learned_correction_independent_axis_support
    assert repaired.training.closed_loop_batch_macro_physical_losses_enabled
    assert repaired.training.closed_loop_axiswise_correction_hinge_enabled
    assert not control.training.closed_loop_batch_macro_physical_losses_enabled
    assert not control.training.closed_loop_axiswise_correction_hinge_enabled
    repaired_dict = repaired.to_dict()
    control_dict = control.to_dict()
    repaired_dict["training"].pop("closed_loop_batch_macro_physical_losses_enabled")
    repaired_dict["training"].pop("closed_loop_axiswise_correction_hinge_enabled")
    control_dict["training"].pop("closed_loop_batch_macro_physical_losses_enabled")
    control_dict["training"].pop("closed_loop_axiswise_correction_hinge_enabled")
    assert control_dict == repaired_dict


def test_axis_gated_updater_xy_repair_changes_only_functional_ownership() -> None:
    source = load_config(CONFIG_DIR / "axis_gated_updater_repair_cpu.yaml")
    repaired = load_config(CONFIG_DIR / "axis_gated_updater_xy_repair_cpu.yaml")

    assert repaired.training.closed_loop_trainable_scope == "updater_state_heads_xy"
    assert repaired.training.closed_loop_event_loss_weights == {"updater_state_heads_xy": 0.0}
    source_dict = source.to_dict()
    repaired_dict = repaired.to_dict()
    source_dict["project"]["name"] = repaired_dict["project"]["name"]
    source_dict["training"]["closed_loop_trainable_scope"] = "updater_state_heads_xy"
    source_dict["training"]["closed_loop_event_loss_weights"] = {"updater_state_heads_xy": 0.0}
    assert source_dict == repaired_dict


def test_scenario_tail_objective_requires_exact_balanced_macro_axiswise_protocol() -> None:
    source = load_config(CONFIG_DIR / "axis_gated_updater_xy_repair_cpu.yaml")
    repaired = replace(
        source,
        training=replace(
            source.training,
            closed_loop_scenario_tail_fraction=0.25,
        ),
    )
    repaired.validate()
    assert repaired.training.closed_loop_scenario_tail_fraction == 0.25

    invalid_training = (
        replace(repaired.training, closed_loop_scenario_tail_fraction=value)
        for value in (0.0, -0.25, 1.25, float("inf"), True, "0.25")
    )
    for training in invalid_training:
        with pytest.raises(ValueError, match="closed_loop_scenario_tail_fraction"):
            replace(repaired, training=training).validate()


def test_uncertainty_standardized_error_gradient_cap_is_strict_and_optional() -> None:
    source = load_config(CONFIG_DIR / "axis_gated_updater_xy_repair_cpu.yaml")
    robust = replace(
        source,
        training=replace(
            source.training,
            closed_loop_uncertainty_standardized_error_gradient_cap=25.0,
        ),
    )
    robust.validate()
    assert robust.training.closed_loop_uncertainty_standardized_error_gradient_cap == 25.0
    for value in (0.0, -1.0, float("inf"), float("nan"), True, "25"):
        with pytest.raises(
            ValueError,
            match="closed_loop_uncertainty_standardized_error_gradient_cap",
        ):
            replace(
                robust,
                training=replace(
                    robust.training,
                    closed_loop_uncertainty_standardized_error_gradient_cap=value,
                ),
            ).validate()


def test_protected_reference_nonregression_requires_exact_causal_protocol() -> None:
    source = load_config(CONFIG_DIR / "protected_state_event_updater_xy_repair_cpu.yaml")
    protected = replace(
        source,
        training=replace(
            source.training,
            closed_loop_scenario_tail_fraction=None,
            closed_loop_protected_reference_nonregression_weight=1.0,
        ),
    )
    protected.validate()
    assert protected.training.closed_loop_protected_reference_nonregression_weight == 1.0

    for value in (-1.0, float("inf"), float("nan"), True, "1"):
        with pytest.raises(
            ValueError,
            match="closed_loop_protected_reference_nonregression_weight",
        ):
            replace(
                protected,
                training=replace(
                    protected.training,
                    closed_loop_protected_reference_nonregression_weight=value,
                ),
            ).validate()
    invalid = (
        replace(protected.training, rgb_pretrain_steps=1),
        replace(protected.training, scenario_balanced_batches=False),
        replace(protected.training, batch_size=protected.training.batch_size * 2),
        replace(protected.training, closed_loop_batch_macro_physical_losses_enabled=False),
        replace(protected.training, closed_loop_axiswise_correction_hinge_enabled=False),
    )
    for training in invalid:
        with pytest.raises(ValueError, match="protected-reference non-regression"):
            replace(protected, training=training).validate()
    with pytest.raises(ValueError, match="attention_dropout=0"):
        replace(
            protected,
            model=replace(
                protected.model,
                dynamics=replace(protected.model.dynamics, attention_dropout=0.1),
            ),
        ).validate()


def test_protected_reference_state_event_profile_changes_only_the_guard() -> None:
    source = load_config(CONFIG_DIR / "protected_state_event_updater_xy_repair_cpu.yaml")
    repaired = load_config(
        CONFIG_DIR / "protected_reference_state_event_updater_xy_repair_cpu.yaml"
    )

    assert repaired.training.closed_loop_protected_reference_nonregression_weight == 1.0
    source_dict = source.to_dict()
    repaired_dict = repaired.to_dict()
    source_dict["project"]["name"] = repaired_dict["project"]["name"]
    source_dict["training"]["closed_loop_protected_reference_nonregression_weight"] = 1.0
    assert source_dict == repaired_dict


def test_scenario_tail_updater_profile_changes_only_tail_and_event_objectives() -> None:
    source = load_config(CONFIG_DIR / "axis_gated_updater_xy_repair_cpu.yaml")
    repaired = load_config(CONFIG_DIR / "scenario_tail_updater_xy_repair_cpu.yaml")

    assert repaired.training.closed_loop_scenario_tail_fraction == 0.25
    assert repaired.training.closed_loop_event_loss_weights == {"updater_state_heads_xy": 0.05}
    assert repaired.training.rollout_anchors_per_window == 2
    assert repaired.training.loss_weights["event"] == 0.05
    assert repaired.training.loss_weights["uncertainty"] == 0.025
    source_dict = source.to_dict()
    repaired_dict = repaired.to_dict()
    source_dict["project"]["name"] = repaired_dict["project"]["name"]
    source_dict["training"]["closed_loop_scenario_tail_fraction"] = 0.25
    source_dict["training"]["closed_loop_event_loss_weights"] = {"updater_state_heads_xy": 0.05}
    source_dict["training"]["rollout_anchors_per_window"] = 2
    source_dict["training"]["loss_weights"]["event"] = 0.05
    source_dict["training"]["loss_weights"]["uncertainty"] = 0.025
    assert source_dict == repaired_dict

    for training in (
        replace(repaired.training, scenario_balanced_batches=False),
        replace(repaired.training, batch_size=repaired.training.batch_size * 2),
        replace(
            repaired.training,
            closed_loop_batch_macro_physical_losses_enabled=False,
        ),
        replace(
            repaired.training,
            closed_loop_axiswise_correction_hinge_enabled=False,
        ),
    ):
        with pytest.raises(ValueError, match="closed_loop_scenario_tail_fraction"):
            replace(repaired, training=training).validate()


def test_robust_scenario_tail_profile_changes_only_event_and_nll_gradient_ownership() -> None:
    source = load_config(CONFIG_DIR / "scenario_tail_updater_xy_repair_cpu.yaml")
    repaired = load_config(CONFIG_DIR / "robust_scenario_tail_updater_xy_repair_cpu.yaml")

    assert repaired.training.closed_loop_scenario_tail_fraction == 0.25
    assert repaired.training.closed_loop_event_loss_weights == {"updater_state_heads_xy": 0.0}
    assert repaired.training.loss_weights["event"] == 0.0
    assert repaired.training.closed_loop_uncertainty_standardized_error_gradient_cap == 25.0
    source_dict = source.to_dict()
    repaired_dict = repaired.to_dict()
    source_dict["project"]["name"] = repaired_dict["project"]["name"]
    source_dict["training"]["closed_loop_event_loss_weights"] = {"updater_state_heads_xy": 0.0}
    source_dict["training"]["loss_weights"]["event"] = 0.0
    source_dict["training"]["closed_loop_uncertainty_standardized_error_gradient_cap"] = 25.0
    assert source_dict == repaired_dict


def test_direct_collision_owner_profile_changes_only_typed_event_ownership() -> None:
    source = load_config(CONFIG_DIR / "robust_scenario_tail_updater_xy_repair_cpu.yaml")
    repaired = load_config(CONFIG_DIR / "direct_collision_owner_updater_xy_repair_cpu.yaml")

    assert repaired.training.closed_loop_trainable_scope == ("updater_state_heads_xy_collision")
    assert repaired.training.closed_loop_event_loss_weights == {
        "updater_state_heads_xy_collision": 0.01
    }
    assert repaired.training.loss_weights["event"] == 0.01
    source_dict = source.to_dict()
    repaired_dict = repaired.to_dict()
    source_dict["project"]["name"] = repaired_dict["project"]["name"]
    source_dict["training"]["closed_loop_trainable_scope"] = "updater_state_heads_xy_collision"
    source_dict["training"]["closed_loop_event_loss_weights"] = {
        "updater_state_heads_xy_collision": 0.01
    }
    source_dict["training"]["loss_weights"]["event"] = 0.01
    assert source_dict == repaired_dict


def test_node_collision_owner_profile_changes_only_typed_event_routing() -> None:
    source = load_config(CONFIG_DIR / "direct_collision_owner_updater_xy_repair_cpu.yaml")
    repaired = load_config(CONFIG_DIR / "node_collision_owner_updater_xy_repair_cpu.yaml")

    scope = "updater_state_heads_xy_collision_node"
    assert repaired.training.closed_loop_trainable_scope == scope
    assert repaired.training.closed_loop_event_loss_weights == {scope: 0.0045}
    assert repaired.training.loss_weights["event"] == 0.0045
    source_dict = source.to_dict()
    repaired_dict = repaired.to_dict()
    source_dict["project"]["name"] = repaired_dict["project"]["name"]
    source_dict["training"]["closed_loop_trainable_scope"] = scope
    source_dict["training"]["closed_loop_event_loss_weights"] = {scope: 0.0045}
    source_dict["training"]["loss_weights"]["event"] = 0.0045
    assert source_dict == repaired_dict


def test_protected_state_event_profile_adds_only_calibrated_state_routing() -> None:
    source = load_config(CONFIG_DIR / "node_collision_owner_updater_xy_repair_cpu.yaml")
    repaired = load_config(CONFIG_DIR / "protected_state_event_updater_xy_repair_cpu.yaml")

    scope = "updater_state_heads_xy_collision_node"
    assert repaired.training.closed_loop_event_loss_weights == {scope: 0.0045}
    assert repaired.training.closed_loop_state_event_loss_weights == {scope: 0.04}
    assert repaired.training.loss_weights["event"] == 0.0045
    source_dict = source.to_dict()
    repaired_dict = repaired.to_dict()
    source_dict["project"]["name"] = repaired_dict["project"]["name"]
    source_dict["training"]["closed_loop_state_event_loss_weights"] = {scope: 0.04}
    assert source_dict == repaired_dict


def test_state_roi_scope_roundtrips_as_typed_configuration(tmp_path: Path) -> None:
    config = load_config(
        CONFIG_DIR / "tiny_overfit.yaml",
        overrides=[
            "training.closed_loop_trainable_scope=state_roi",
            "training.closed_loop_late_trainable_scope=state_dynamics_roi",
            "training.closed_loop_scope_transition_steps=512",
        ],
    )
    resolved_path = tmp_path / "state-roi-resolved.yaml"

    save_resolved_config(config, resolved_path)
    restored = load_config(resolved_path)

    assert restored.training.closed_loop_trainable_scope == "state_roi"
    assert restored.training.closed_loop_late_trainable_scope == "state_dynamics_roi"
    assert restored.training.closed_loop_scope_transition_steps == 512
    assert restored.to_dict() == config.to_dict()


def test_state_relation_roi_scope_is_typed_and_requires_attention(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="state_relation_roi.*requires.*attention"):
        load_config(
            CONFIG_DIR / "tiny_overfit.yaml",
            overrides=[
                "training.closed_loop_trainable_scope=state_roi",
                "training.closed_loop_late_trainable_scope=state_relation_roi",
                "training.closed_loop_scope_transition_steps=512",
            ],
        )

    config = load_config(
        CONFIG_DIR / "tiny_overfit.yaml",
        overrides=[
            "model.dynamics.attention_residual_enabled=true",
            "training.closed_loop_trainable_scope=state_roi",
            "training.closed_loop_late_trainable_scope=state_relation_roi",
            "training.closed_loop_scope_transition_steps=512",
        ],
    )
    resolved_path = tmp_path / "state-relation-roi-resolved.yaml"

    save_resolved_config(config, resolved_path)
    restored = load_config(resolved_path)

    assert restored.training.closed_loop_trainable_scope == "state_roi"
    assert restored.training.closed_loop_late_trainable_scope == "state_relation_roi"
    assert restored.training.closed_loop_scope_transition_steps == 512
    assert restored.to_dict() == config.to_dict()


def test_scope_owned_event_weights_roundtrip_as_typed_configuration(tmp_path: Path) -> None:
    config = load_config(
        CONFIG_DIR / "tiny_overfit.yaml",
        overrides=[
            "training.closed_loop_event_loss_weights={state_roi: 0.0, state_relation_roi: 0.05}",
        ],
    )
    resolved_path = tmp_path / "scope-event-weights-resolved.yaml"

    save_resolved_config(config, resolved_path)
    restored = load_config(resolved_path)

    assert restored.training.closed_loop_event_loss_weights == {
        "state_roi": 0.0,
        "state_relation_roi": 0.05,
    }
    assert restored.to_dict() == config.to_dict()


def test_prior_future_correction_rollout_is_legacy_true_and_roundtrips(
    tmp_path: Path,
) -> None:
    legacy = load_config(CONFIG_DIR / "default.yaml")
    disabled = load_config(
        CONFIG_DIR / "tiny_overfit.yaml",
        overrides=["training.closed_loop_prior_future_correction_enabled=false"],
    )
    resolved_path = tmp_path / "no-prior-correction-rollout.yaml"

    save_resolved_config(disabled, resolved_path)
    restored = load_config(resolved_path)

    assert legacy.training.closed_loop_prior_future_correction_enabled
    assert not restored.training.closed_loop_prior_future_correction_enabled
    assert restored.to_dict() == disabled.to_dict()


def test_closed_loop_physical_objective_repairs_are_legacy_false_and_roundtrip(
    tmp_path: Path,
) -> None:
    legacy = load_config(CONFIG_DIR / "tiny_overfit.yaml")
    repaired = load_config(
        CONFIG_DIR / "tiny_overfit.yaml",
        overrides=[
            "training.closed_loop_batch_macro_physical_losses_enabled=true",
            "training.closed_loop_axiswise_correction_hinge_enabled=true",
        ],
    )
    resolved_path = tmp_path / "macro-axiswise-objective.yaml"

    save_resolved_config(repaired, resolved_path)
    restored = load_config(resolved_path)

    assert not legacy.training.closed_loop_batch_macro_physical_losses_enabled
    assert not legacy.training.closed_loop_axiswise_correction_hinge_enabled
    assert restored.training.closed_loop_batch_macro_physical_losses_enabled
    assert restored.training.closed_loop_axiswise_correction_hinge_enabled
    assert restored.to_dict() == repaired.to_dict()


@pytest.mark.parametrize(
    "field_name",
    [
        "closed_loop_batch_macro_physical_losses_enabled",
        "closed_loop_axiswise_correction_hinge_enabled",
    ],
)
@pytest.mark.parametrize("value", ["0", "1", "null", "not-a-boolean"])
def test_closed_loop_physical_objective_repairs_require_boolean(
    field_name: str,
    value: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=f"{field_name}|boolean"):
        load_config(
            CONFIG_DIR / "tiny_overfit.yaml",
            overrides=[f"training.{field_name}={value}"],
        )


@pytest.mark.parametrize("value", ["0", "1", "null", "not-a-boolean"])
def test_prior_future_correction_rollout_requires_boolean(value: str) -> None:
    with pytest.raises(
        (TypeError, ValueError),
        match="closed_loop_prior_future_correction_enabled|boolean",
    ):
        load_config(
            CONFIG_DIR / "tiny_overfit.yaml",
            overrides=[f"training.closed_loop_prior_future_correction_enabled={value}"],
        )


@pytest.mark.parametrize(
    "value",
    ["-1.0", ".nan", "true", "not-a-number"],
)
def test_scope_owned_event_weights_must_be_finite_and_nonnegative(value: str) -> None:
    with pytest.raises(ValueError, match="closed_loop_event_loss_weights"):
        load_config(
            CONFIG_DIR / "tiny_overfit.yaml",
            overrides=[
                f"training.closed_loop_event_loss_weights={{state_roi: {value}}}",
            ],
        )


def test_scope_owned_event_weights_reject_unknown_scope() -> None:
    with pytest.raises(ValueError, match="closed_loop_event_loss_weights"):
        load_config(
            CONFIG_DIR / "tiny_overfit.yaml",
            overrides=["training.closed_loop_event_loss_weights={perception: 0.0}"],
        )


def test_attention_relation_endpoint_binding_is_explicit_opt_in(tmp_path: Path) -> None:
    legacy = load_config(CONFIG_DIR / "default.yaml")
    enabled = load_config(
        CONFIG_DIR / "default.yaml",
        overrides=[
            "model.dynamics.attention_residual_enabled=true",
            "model.dynamics.attention_relation_endpoint_binding_enabled=true",
        ],
    )
    resolved_path = tmp_path / "endpoint-binding-resolved.yaml"

    save_resolved_config(enabled, resolved_path)
    restored = load_config(resolved_path)

    assert not legacy.model.dynamics.attention_relation_endpoint_binding_enabled
    assert enabled.model.dynamics.attention_relation_endpoint_binding_enabled
    assert restored.to_dict() == enabled.to_dict()


def test_smooth_event_hazard_is_explicit_and_roundtrips(tmp_path: Path) -> None:
    legacy = load_config(CONFIG_DIR / "default.yaml")
    enabled = load_config(
        CONFIG_DIR / "default.yaml",
        overrides=[
            "model.dynamics.smooth_event_hazard_enabled=true",
            "model.dynamics.event_hazard_gap_temperature_m=0.03",
            "model.dynamics.event_hazard_velocity_temperature_mps=0.2",
            "model.dynamics.event_hazard_resolved_logit_floor=1.5",
        ],
    )
    resolved_path = tmp_path / "smooth-event-hazard-resolved.yaml"

    save_resolved_config(enabled, resolved_path)
    restored = load_config(resolved_path)
    dynamics = DynamicsModel.from_config(enabled)

    assert not legacy.model.dynamics.smooth_event_hazard_enabled
    assert enabled.model.dynamics.smooth_event_hazard_enabled
    assert dynamics.events.smooth_hazard_enabled
    assert dynamics.events.contact_logit_scale == pytest.approx(0.03)
    assert dynamics.events.collision_velocity_logit_scale == pytest.approx(0.2)
    assert dynamics.events.resolved_event_logit_floor == pytest.approx(1.5)
    assert restored.to_dict() == enabled.to_dict()


@pytest.mark.parametrize(
    "override",
    [
        "model.dynamics.event_hazard_gap_temperature_m=0.0",
        "model.dynamics.event_hazard_velocity_temperature_mps=-0.1",
        "model.dynamics.event_hazard_resolved_logit_floor=0.0",
    ],
)
def test_smooth_event_hazard_scales_must_be_positive(override: str) -> None:
    with pytest.raises(ValueError, match="event_hazard"):
        load_config(CONFIG_DIR / "default.yaml", overrides=[override])


@pytest.mark.parametrize(
    "overrides",
    [
        ["training.closed_loop_late_trainable_scope=state_dynamics"],
        ["training.closed_loop_scope_transition_steps=512"],
        [
            "training.closed_loop_late_trainable_scope=unknown",
            "training.closed_loop_scope_transition_steps=512",
        ],
        [
            "training.closed_loop_late_trainable_scope=state_dynamics",
            "training.closed_loop_scope_transition_steps=0",
        ],
    ],
)
def test_closed_loop_scope_transition_is_explicit_and_paired(
    overrides: list[str],
) -> None:
    with pytest.raises(ValueError, match="closed_loop_(late|scope)"):
        load_config(CONFIG_DIR / "tiny_overfit.yaml", overrides=overrides)


@pytest.mark.parametrize(
    "field",
    [
        "handoff_minimum_target_coverage",
        "handoff_minimum_forecast_coverage",
        "handoff_minimum_reference_coverage_ratio",
    ],
)
@pytest.mark.parametrize("value", ["-0.1", "1.1", ".inf"])
def test_handoff_coverage_controls_are_probabilities(
    field: str,
    value: str,
) -> None:
    with pytest.raises(ValueError, match=field):
        load_config(
            CONFIG_DIR / "tiny_overfit.yaml",
            overrides=[f"training.{field}={value}"],
        )


def test_no_gradient_retry_bound_is_nonnegative() -> None:
    with pytest.raises(
        ValueError,
        match="maximum_no_gradient_batches_per_update",
    ):
        load_config(
            CONFIG_DIR / "tiny_overfit.yaml",
            overrides=["training.maximum_no_gradient_batches_per_update=-1"],
        )


@pytest.mark.parametrize("value", ["-1.0", ".inf"])
def test_minimum_effective_gradient_norm_is_finite_and_nonnegative(
    value: str,
) -> None:
    with pytest.raises(ValueError, match="minimum_effective_gradient_norm"):
        load_config(
            CONFIG_DIR / "tiny_overfit.yaml",
            overrides=[f"training.minimum_effective_gradient_norm={value}"],
        )


@pytest.mark.parametrize("value", ["0.0", "-1.0", ".inf"])
def test_fast_roi_pretrain_weight_is_finite_and_positive(value: str) -> None:
    with pytest.raises(ValueError, match="fast_roi_pretrain_weight"):
        load_config(
            CONFIG_DIR / "tiny_overfit.yaml",
            overrides=[f"training.fast_roi_pretrain_weight={value}"],
        )


@pytest.mark.parametrize("value", ["0", "-1", "true", "1.5"])
def test_birth_confirmation_count_must_be_positive_integer(value: str) -> None:
    with pytest.raises(ValueError, match="birth_confirmations must be a positive integer"):
        load_config(
            CONFIG_DIR / "tiny_overfit.yaml",
            overrides=[f"model.lifecycle.birth_confirmations={value}"],
        )


def test_multi_frame_birth_confirmation_is_supported() -> None:
    config = load_config(
        CONFIG_DIR / "tiny_overfit.yaml",
        overrides=["model.lifecycle.birth_confirmations=2"],
    )
    assert config.model.lifecycle.birth_confirmations == 2


@pytest.mark.parametrize("value", ["0.0", "-1.0", ".inf"])
def test_birth_confirmation_distance_must_be_finite_and_positive(value: str) -> None:
    with pytest.raises(ValueError, match="birth_confirmation_distance_m"):
        load_config(
            CONFIG_DIR / "tiny_overfit.yaml",
            overrides=[f"model.lifecycle.birth_confirmation_distance_m={value}"],
        )


@pytest.mark.parametrize("value", ["0", "-1", ".inf", ".nan"])
def test_association_maximum_cost_must_be_finite_and_positive(value: str) -> None:
    with pytest.raises(ValueError, match="model.association.maximum_cost"):
        load_config(
            "configs/default.yaml",
            overrides=[f"model.association.maximum_cost={value}"],
        )


@pytest.mark.parametrize("value", ["0.0", "-1.0", ".inf"])
def test_gradient_clip_norm_is_finite_and_positive(value: str) -> None:
    with pytest.raises(ValueError, match="grad_clip_norm"):
        load_config(
            "configs/tiny_overfit.yaml",
            overrides=[f"training.grad_clip_norm={value}"],
        )


@pytest.mark.parametrize("value", ["0.0", "-1.0", ".inf"])
def test_interaction_gradient_clip_norm_is_finite_and_positive(
    value: str,
) -> None:
    with pytest.raises(ValueError, match="interaction_grad_clip_norm"):
        load_config(
            "configs/tiny_overfit.yaml",
            overrides=[f"training.interaction_grad_clip_norm={value}"],
        )


@pytest.mark.parametrize("value", ["0.0", "-1.0", ".inf"])
def test_closed_loop_perception_gradient_clip_norm_is_finite_and_positive(
    value: str,
) -> None:
    with pytest.raises(ValueError, match="closed_loop_perception_grad_clip_norm"):
        load_config(
            "configs/tiny_overfit.yaml",
            overrides=[f"training.closed_loop_perception_grad_clip_norm={value}"],
        )


@pytest.mark.parametrize("value", ["true", "1.5", "-1"])
def test_maximum_no_gradient_batches_must_be_nonnegative_integer(value: str) -> None:
    with pytest.raises(
        ValueError,
        match="maximum_no_gradient_batches_per_update must be a nonnegative integer",
    ):
        load_config(
            CONFIG_DIR / "tiny_overfit.yaml",
            overrides=[f"training.maximum_no_gradient_batches_per_update={value}"],
        )


@pytest.mark.parametrize(
    "field",
    [
        "normalize_rollout_axes_over_configured_horizons",
        "joint_collision_long_horizon_sampling",
    ],
)
def test_horizon_sampling_controls_are_boolean(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        load_config(
            "configs/tiny_overfit.yaml",
            overrides=[f"training.{field}=not-a-boolean"],
        )


def test_active_sustained_campaign_pins_legacy_horizon_semantics() -> None:
    config = load_config("configs/sustained_accuracy_mps.yaml")

    assert not config.training.normalize_rollout_axes_over_configured_horizons
    assert not config.training.joint_collision_long_horizon_sampling


def test_collision_positive_weight_is_at_least_one() -> None:
    with pytest.raises(ValueError, match="collision_positive_weight_max"):
        load_config(
            CONFIG_DIR / "tiny_overfit.yaml",
            overrides=["training.collision_positive_weight_max=0.5"],
        )


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_attention_node_gradient_clip_is_positive_when_configured(value: str) -> None:
    with pytest.raises(ValueError, match="attention_node_grad_clip_norm"):
        load_config(
            CONFIG_DIR / "tiny_overfit.yaml",
            overrides=[f"training.attention_node_grad_clip_norm={value}"],
        )


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_attention_collision_gradient_clip_is_positive_when_configured(value: str) -> None:
    with pytest.raises(ValueError, match="attention_collision_grad_clip_norm"):
        load_config(
            CONFIG_DIR / "tiny_overfit.yaml",
            overrides=[f"training.attention_collision_grad_clip_norm={value}"],
        )


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_attention_force_gradient_clip_is_positive_when_configured(value: str) -> None:
    with pytest.raises(ValueError, match="attention_force_grad_clip_norm"):
        load_config(
            CONFIG_DIR / "tiny_overfit.yaml",
            overrides=[f"training.attention_force_grad_clip_norm={value}"],
        )


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_attention_impulse_gradient_clip_is_positive_when_configured(value: str) -> None:
    with pytest.raises(ValueError, match="attention_impulse_grad_clip_norm"):
        load_config(
            CONFIG_DIR / "tiny_overfit.yaml",
            overrides=[f"training.attention_impulse_grad_clip_norm={value}"],
        )


@pytest.mark.parametrize("value", ["0", "-1", "1.1", "nan", "inf"])
def test_minimum_interaction_gradient_retention_is_bounded(value: str) -> None:
    with pytest.raises(ValueError, match="minimum_interaction_gradient_retention"):
        load_config(
            CONFIG_DIR / "tiny_overfit.yaml",
            overrides=[f"training.minimum_interaction_gradient_retention={value}"],
        )


@pytest.mark.parametrize(
    "field",
    [
        "attention_node_output_grad_clip_norm",
        "attention_collision_output_grad_clip_norm",
        "attention_force_output_grad_clip_norm",
        "attention_impulse_output_grad_clip_norm",
    ],
)
@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_attention_output_gradient_clips_are_positive_when_configured(
    field: str,
    value: str,
) -> None:
    with pytest.raises(ValueError, match=field):
        load_config(
            CONFIG_DIR / "tiny_overfit.yaml",
            overrides=[f"training.{field}={value}"],
        )


@pytest.mark.parametrize("probability", [-0.1, 1.1])
def test_long_horizon_window_probability_is_bounded(probability: float) -> None:
    with pytest.raises(ValueError, match="long_horizon_window_probability"):
        load_config(
            CONFIG_DIR / "tiny_overfit.yaml",
            overrides=[f"training.long_horizon_window_probability={probability}"],
        )


@pytest.mark.parametrize("anchors", [0, -1])
def test_rollout_anchors_per_window_must_be_positive_or_null(anchors: int) -> None:
    with pytest.raises(ValueError, match="rollout_anchors_per_window"):
        load_config(
            CONFIG_DIR / "tiny_overfit.yaml",
            overrides=[f"training.rollout_anchors_per_window={anchors}"],
        )


def test_rollout_anchors_per_window_accepts_bounded_training_value() -> None:
    config = load_config(
        CONFIG_DIR / "tiny_overfit.yaml",
        overrides=["training.rollout_anchors_per_window=2"],
    )

    assert config.training.rollout_anchors_per_window == 2


def test_validation_rollout_anchors_must_be_positive_or_null() -> None:
    with pytest.raises(
        ValueError,
        match="validation_rollout_anchors_per_episode",
    ):
        load_config(
            CONFIG_DIR / "tiny_overfit.yaml",
            overrides=["training.validation_rollout_anchors_per_episode=0"],
        )


@pytest.mark.parametrize(
    "name",
    [
        "validation_minimum_predictable_target_count_per_scenario_horizon",
        "validation_minimum_matched_target_count_per_scenario_horizon",
        "validation_minimum_supported_episodes_per_scenario",
    ],
)
@pytest.mark.parametrize("value", ["0", "true", "1.5"])
def test_validation_support_floors_must_be_positive_integers(
    name: str,
    value: str,
) -> None:
    with pytest.raises(ValueError, match=name):
        load_config(
            CONFIG_DIR / "tiny_overfit.yaml",
            overrides=[f"training.{name}={value}"],
        )


def test_supported_episode_floor_must_fit_balanced_validation_manifest() -> None:
    with pytest.raises(
        ValueError,
        match="validation_minimum_supported_episodes_per_scenario.*exceeds",
    ):
        load_config(
            CONFIG_DIR / "tiny_all_scenarios.yaml",
            overrides=[
                "training.validation_minimum_supported_episodes_per_scenario=2",
            ],
        )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("structured_disc_threshold", 0.0),
        ("structured_disc_min_pixels", 0),
        ("structured_disc_max_assignment_distance", 0.0),
        ("structured_disc_center_std_pixels", 0.0),
    ],
)
def test_structured_rgb_controls_are_bounded(
    key: str,
    value: float | int,
) -> None:
    with pytest.raises(ValueError, match=key):
        load_config(
            CONFIG_DIR / "toy_smoke.yaml",
            overrides=[f"model.rgb.{key}={value}"],
        )


def test_measurement_loss_weights_are_nonnegative() -> None:
    with pytest.raises(ValueError, match="measurement_loss_weights"):
        load_config(
            CONFIG_DIR / "tiny_overfit.yaml",
            overrides=["training.measurement_loss_weights={rgb_world_position: -1.0}"],
        )


@pytest.mark.parametrize("value", ["-.inf", ".nan", "-1.0"])
def test_closed_loop_loss_weights_are_finite_and_nonnegative(value: str) -> None:
    with pytest.raises(ValueError, match="loss_weights"):
        load_config(
            CONFIG_DIR / "tiny_overfit.yaml",
            overrides=[f"training.loss_weights={{measurement: {value}}}"],
        )


@pytest.mark.parametrize("value", ["-1", "true", "1.5"])
def test_rgb_pretrain_steps_must_be_a_nonnegative_integer(value: str) -> None:
    with pytest.raises(ValueError, match="rgb_pretrain_steps.*nonnegative integer"):
        load_config(
            CONFIG_DIR / "tiny_overfit.yaml",
            overrides=[f"training.rgb_pretrain_steps={value}"],
        )


def test_training_steps_must_be_positive() -> None:
    with pytest.raises(ValueError, match="steps must be positive"):
        load_config(
            CONFIG_DIR / "tiny_overfit.yaml",
            overrides=["training.steps=0"],
        )


def test_scenario_mixture_rejects_duplicates_that_break_validation_coverage() -> None:
    with pytest.raises(ValueError, match="scenario_mixture.*unique"):
        load_config(
            CONFIG_DIR / "tiny_overfit.yaml",
            overrides=[
                "simulator.scenario_mixture=[elastic_pairs,baseline,baseline]",
                "training.validation_episodes=2",
            ],
        )


def test_scenario_balanced_batches_require_complete_equal_batch_support() -> None:
    config = load_config(
        CONFIG_DIR / "tiny_overfit.yaml",
        overrides=[
            "simulator.scenario_mixture=[baseline,elastic_pairs]",
            "training.validation_episodes=2",
            "training.train_episodes=8",
            "training.batch_size=4",
            "training.scenario_balanced_batches=true",
        ],
    )
    assert config.training.scenario_balanced_batches

    with pytest.raises(ValueError, match="multiple of the scenario count"):
        load_config(
            CONFIG_DIR / "tiny_overfit.yaml",
            overrides=[
                "simulator.scenario_mixture=[baseline,elastic_pairs]",
                "training.validation_episodes=2",
                "training.train_episodes=8",
                "training.batch_size=3",
                "training.scenario_balanced_batches=true",
            ],
        )
    with pytest.raises(ValueError, match="train_episodes must be divisible"):
        load_config(
            CONFIG_DIR / "tiny_overfit.yaml",
            overrides=[
                "simulator.scenario_mixture=[baseline,elastic_pairs]",
                "training.validation_episodes=2",
                "training.train_episodes=10",
                "training.batch_size=4",
                "training.scenario_balanced_batches=true",
            ],
        )


@pytest.mark.parametrize("profile_name", ["toy_hard.yaml", "cloud_single_gpu.yaml"])
def test_noisy_profiles_use_noise_robust_structured_rgb_threshold(
    profile_name: str,
) -> None:
    config = load_config(CONFIG_DIR / profile_name)
    rgb_config = config.model.rgb
    assert rgb_config.structured_disc_center_enabled
    assert rgb_config.structured_disc_threshold == pytest.approx(0.08)

    generator = torch.Generator().manual_seed(20260727)
    height, width = config.simulator.image_size
    image = (
        torch.full((2, 3, height, width), 0.2)
        + config.simulator.render_noise_std
        * torch.randn((2, 3, height, width), generator=generator)
    ).clamp(0.0, 1.0)
    proposal_centres = torch.zeros((2, rgb_config.proposal_queries, 2))
    output = structured_disc_centres(
        image,
        proposal_centres,
        threshold=rgb_config.structured_disc_threshold,
        minimum_pixels=rgb_config.structured_disc_min_pixels,
        maximum_assignment_distance=rgb_config.structured_disc_max_assignment_distance,
    )

    # Renderer-strength Gaussian noise may form an occasional tiny connected
    # cluster, but must not consume a meaningful fraction of proposal slots.
    assert int(output.component_count.max()) <= 2


def test_all_scenarios_profile_is_balanced_and_uses_one_shared_model() -> None:
    config = load_config(CONFIG_DIR / "tiny_all_scenarios.yaml")

    assert config.simulator.scenario_mixture == (
        "reference_pairs",
        "baseline",
        "elastic_pairs",
        "damped_contacts",
        "impulse_perturbation",
        "camera_parallax",
        "glancing_impacts",
        "heavy_light_impacts",
    )
    assert config.training.validation_episodes == len(config.simulator.scenario_mixture)
    assert config.model.max_objects >= config.simulator.max_objects
    assert config.runtime.modality == "rgb"
    assert not config.runtime.enable_debug_oracle
    assert config.model.rgb.temporal_velocity_max_age_steps is None


def test_scaled_curriculum_has_capacity_and_thousands_of_diverse_episodes() -> None:
    config = load_config(CONFIG_DIR / "scaled_curriculum.yaml")

    assert config.training.train_episodes == 4096
    assert config.training.validation_episodes == 256
    assert not config.training.fixed_dataset
    assert config.training.steps * config.training.batch_size >= 10 * config.training.train_episodes
    assert config.training.train_episodes % len(config.simulator.scenario_mixture) == 0
    assert config.training.validation_episodes % len(config.simulator.scenario_mixture) == 0
    assert config.model.max_objects >= config.simulator.max_objects
    assert config.model.rgb.feature_dim >= 96
    assert config.model.dynamics.hidden_dim >= 160
    assert config.model.filter.hidden_dim >= 192
    assert config.runtime.modality == "rgb"
    assert not config.runtime.enable_debug_oracle
