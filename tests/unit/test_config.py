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


def test_temporal_rgb_velocity_is_opt_in_and_typed() -> None:
    default = load_config(CONFIG_DIR / "toy_smoke.yaml")
    assert not default.model.rgb.temporal_velocity_enabled
    assert default.model.rgb.temporal_velocity_history_size == 3
    assert default.model.rgb.temporal_velocity_variance_ceiling is None
    assert default.model.rgb.structured_disc_center_enabled

    enabled = load_config(
        CONFIG_DIR / "toy_smoke.yaml",
        overrides=[
            "model.rgb.temporal_velocity_enabled=true",
            "model.rgb.temporal_velocity_history_size=4",
            "model.rgb.temporal_velocity_variance_scale=3.0",
            "model.rgb.temporal_velocity_variance_floor=0.4",
            "model.rgb.temporal_velocity_variance_ceiling=2.0",
        ],
    )
    assert enabled.model.rgb.temporal_velocity_enabled
    assert enabled.model.rgb.temporal_velocity_history_size == 4
    assert enabled.model.rgb.temporal_velocity_variance_scale == 3.0
    assert enabled.model.rgb.temporal_velocity_variance_floor == 0.4
    assert enabled.model.rgb.temporal_velocity_variance_ceiling == 2.0


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("temporal_velocity_min_dt", 0.0),
        ("temporal_velocity_history_size", 2),
        ("temporal_velocity_variance_scale", 0.5),
        ("temporal_velocity_variance_floor", 0.0),
        ("temporal_velocity_variance_ceiling", 0.1),
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


def test_collision_positive_weight_is_at_least_one() -> None:
    with pytest.raises(ValueError, match="collision_positive_weight_max"):
        load_config(
            CONFIG_DIR / "tiny_overfit.yaml",
            overrides=["training.collision_positive_weight_max=0.5"],
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
