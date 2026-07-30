from pathlib import Path

import pytest
import torch

from world_model.datasets.splits import SPLIT_SEED_RANGES
from world_model.observations.rgb.structured_centres import structured_disc_centres
from world_model.utils.config import load_config

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


def test_dotted_override_is_typed(tmp_path: Path) -> None:
    config = load_config(
        CONFIG_DIR / "toy_smoke.yaml",
        overrides=["training.steps=3", "simulator.image_size=[32, 40]"],
    )
    assert config.training.steps == 3
    assert config.simulator.image_size == (32, 40)


def test_simulator_scenario_mixture_is_typed_and_validated() -> None:
    config = load_config(
        CONFIG_DIR / "toy_smoke.yaml",
        overrides=[
            "simulator.scenario_mixture=[elastic_pairs,damped_contacts,impulse_perturbation]",
            "simulator.initial_speed_range=[0.2,1.8]",
            "simulator.external_impulse_range=[0.1,0.9]",
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


def test_temporal_rgb_velocity_is_opt_in_and_typed() -> None:
    default = load_config(CONFIG_DIR / "toy_smoke.yaml")
    assert not default.model.rgb.temporal_velocity_enabled
    assert default.model.rgb.temporal_velocity_history_size == 3
    assert default.model.rgb.temporal_velocity_min_samples == 3
    assert default.model.rgb.temporal_velocity_variance_ceiling is None
    assert not default.model.rgb.temporal_velocity_lateral_only
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


@pytest.mark.parametrize(
    ("key", "value"),
    [
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


def test_closed_loop_global_trainable_steps_is_nonnegative() -> None:
    with pytest.raises(ValueError, match="closed_loop_global_trainable_steps"):
        load_config(
            CONFIG_DIR / "tiny_overfit.yaml",
            overrides=["training.closed_loop_global_trainable_steps=-1"],
        )


def test_closed_loop_trainable_scope_is_explicit() -> None:
    with pytest.raises(ValueError, match="closed_loop_trainable_scope"):
        load_config(
            CONFIG_DIR / "tiny_overfit.yaml",
            overrides=["training.closed_loop_trainable_scope=perception"],
        )


def test_collision_positive_weight_is_at_least_one() -> None:
    with pytest.raises(ValueError, match="collision_positive_weight_max"):
        load_config(
            CONFIG_DIR / "tiny_overfit.yaml",
            overrides=["training.collision_positive_weight_max=0.5"],
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
