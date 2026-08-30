"""Frozen profile contract for seedless per-object variable metric radius."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from world_model.utils.config import load_config, save_resolved_config

PROFILE = Path(__file__).parents[2] / "configs" / "rgbd_variable_radius_cpu.yaml"


def _assert_profile_contract(config: object) -> None:
    project = config.project
    simulator = config.simulator
    model = config.model
    runtime = config.runtime
    training = config.training
    evaluation = config.evaluation

    assert project.name == "orpheus-rgbd-variable-radius-v1-cpu"
    assert project.seed == 0
    assert project.deterministic is True
    assert simulator.image_size == (64, 64)
    assert simulator.frame_rate == 20
    assert simulator.physics_rate == 120
    assert simulator.sequence_frames == 56
    assert simulator.min_objects == simulator.max_objects == 2
    assert simulator.gravity == (0.0, 0.0, 0.0)
    assert simulator.radius_range == (0.19, 0.25)
    assert simulator.drag_range == (0.05, 0.05)
    assert simulator.initial_speed_range == (0.035, 0.05)
    assert simulator.camera_motion == "orbit"
    assert simulator.render_noise_std == 0.0
    assert simulator.ensure_collision is False
    assert simulator.external_impulse_probability == 0.0
    assert simulator.scenario_mixture == ("baseline",)

    assert model.max_objects == 2
    assert model.state.geometry_dim == 1
    assert model.state.appearance_dim == 3
    assert model.rgb.enabled is False
    assert model.rgbd.enabled is True
    assert model.rgbd.proposal_count == 2
    assert model.rgbd.world_radius == 0.21
    assert model.rgbd.metric_radius_estimation_enabled is True
    assert model.rgbd.minimum_world_radius == 0.19
    assert model.rgbd.maximum_world_radius == 0.25
    assert model.rgbd.measurement_radius_variance == 1.0e-5
    assert model.rgbd.linear_drag == 0.05
    assert model.rgbd.temporal_history_size == 16
    assert model.rgbd.temporal_min_samples == 16
    assert model.dynamics.analytic_free_motion_only is True
    assert model.dynamics.attention_residual_enabled is False
    assert model.filter.min_log_variance == -12.0
    assert model.filter.max_log_variance == 8.0
    assert model.filter.enable_learned_corrector is False
    assert model.filter.direct_metric_position_update is True
    assert model.filter.innovation_anchored_correction is True
    assert model.identification.enabled is False

    assert runtime.modality == "rgbd"
    assert runtime.modality_order == ("debug_oracle", "rgbd")
    assert runtime.enable_debug_oracle is False
    assert runtime.strict_timestamps is True
    assert runtime.hypothesis_pool_enabled is False
    assert training.train_episodes == 64
    assert training.validation_episodes == 64
    assert training.fixed_dataset is True
    assert evaluation.episodes == 64
    assert evaluation.horizons_seconds == (0.1, 0.25, 0.5, 1.0, 2.0)
    assert evaluation.rgb_only is False


def test_variable_radius_profile_has_exact_literal_and_resolved_types() -> None:
    raw = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    assert type(raw["project"]["seed"]) is int
    assert type(raw["model"]["rgbd"]["metric_radius_estimation_enabled"]) is bool
    for path in (
        ("simulator", "radius_range"),
        ("simulator", "drag_range"),
        ("simulator", "initial_speed_range"),
    ):
        values = raw[path[0]][path[1]]
        assert type(values) is list
        assert all(type(value) is float for value in values)
    for key in (
        "world_radius",
        "minimum_world_radius",
        "maximum_world_radius",
        "measurement_radius_variance",
        "linear_drag",
    ):
        assert type(raw["model"]["rgbd"][key]) is float

    config = load_config(PROFILE)
    config.validate()
    _assert_profile_contract(config)


def test_variable_radius_profile_roundtrip_is_exact(tmp_path: Path) -> None:
    config = load_config(PROFILE)
    resolved = tmp_path / "resolved.yaml"
    save_resolved_config(config, resolved)
    restored = load_config(resolved)

    assert restored.to_dict() == config.to_dict()
    _assert_profile_contract(restored)


@pytest.mark.parametrize(
    "override",
    [
        "project.seed=1",
        "simulator.radius_range=[0.20,0.25]",
        "simulator.drag_range=[0.04,0.04]",
        "simulator.initial_speed_range=[0.03,0.05]",
        "simulator.camera_motion=fixed",
        "simulator.ensure_collision=true",
        "simulator.external_impulse_probability=0.1",
        "simulator.scenario_mixture=[elastic_pairs]",
        "model.max_objects=3",
        "model.state.geometry_dim=2",
        "model.rgbd.proposal_count=1",
        "model.rgbd.metric_radius_estimation_enabled=false",
        "model.rgbd.minimum_world_radius=0.18",
        "model.rgbd.maximum_world_radius=0.26",
        "model.rgbd.measurement_radius_variance=0.00002",
        "model.rgbd.linear_drag=0.06",
        "model.rgbd.temporal_history_size=15",
        "model.filter.min_log_variance=-11.0",
        "model.filter.max_log_variance=7.0",
        "model.filter.enable_learned_corrector=true",
        "model.filter.direct_metric_position_update=false",
        "model.identification.enabled=true",
        "runtime.enable_debug_oracle=true",
        "runtime.hypothesis_pool_enabled=true",
        "training.train_episodes=63",
        "training.validation_episodes=65",
        "evaluation.episodes=63",
        "evaluation.horizons_seconds=[0.1,0.25,0.5,1.0]",
    ],
)
def test_variable_radius_profile_detects_every_valid_but_wrong_mutation(
    override: str,
) -> None:
    try:
        mutated = load_config(PROFILE, overrides=[override])
    except (TypeError, ValueError):
        return

    with pytest.raises(AssertionError):
        _assert_profile_contract(mutated)
