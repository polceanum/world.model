from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from world_model.datasets.splits import SPLIT_SEED_RANGES
from world_model.dynamics import DynamicsModel
from world_model.observations.rgb.structured_centres import structured_disc_centres
from world_model.simulator.physics import PhysicsConfig
from world_model.utils.config import load_config, save_resolved_config

CONFIG_DIR = Path(__file__).parents[2] / "configs"
IDENTIFIABLE_DRAG_PROFILE = CONFIG_DIR / "rgbd_two_visible_orbital_camera_identifiable_drag_v1.yaml"


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.yaml")))
def test_profiles_resolve_and_validate(path: Path) -> None:
    config = load_config(path)
    assert config.model.max_objects >= config.simulator.max_objects
    assert config.runtime.modality in {"rgb", "rgbd"}
    if config.runtime.modality == "rgbd":
        assert config.model.rgbd.enabled
        assert not config.model.rgb.enabled
        assert config.model.dynamics.analytic_free_motion_only
        assert not config.model.filter.enable_learned_corrector
    assert config.evaluation.rgb_only is (config.runtime.modality == "rgb")
    test_lower, test_upper = SPLIT_SEED_RANGES["test"]
    assert test_lower <= config.demo.seed <= test_upper
    assert config.simulator.split_validation_start == SPLIT_SEED_RANGES["validation"][0]
    assert config.simulator.split_test_start == test_lower
    assert config.simulator.split_ood_start == SPLIT_SEED_RANGES["ood"][0]


def test_rgbd_online_profile_binds_metric_temporal_and_analytic_semantics() -> None:
    config = load_config(CONFIG_DIR / "rgbd_online_free_motion_cpu.yaml")

    assert config.runtime.modality == "rgbd"
    assert config.runtime.modality_order == ("debug_oracle", "rgbd")
    assert config.model.rgbd.enabled
    assert config.model.rgbd.proposal_count == 1
    assert config.model.rgbd.world_radius == pytest.approx(0.21)
    assert config.model.rgbd.linear_drag == pytest.approx(0.05)
    assert config.model.rgbd.temporal_history_size == 16
    assert config.model.rgbd.temporal_min_samples == 16
    assert config.model.dynamics.analytic_free_motion_only
    assert not config.model.filter.enable_learned_corrector
    assert config.model.filter.direct_metric_position_update
    assert not config.model.identification.enabled
    assert not config.evaluation.rgb_only


@pytest.mark.parametrize(
    "override,match",
    [
        ("model.rgbd.enabled=false", "direct metric position updates require the RGB-D runtime"),
        ("model.rgbd.temporal_min_samples=15", "must equal temporal_history_size"),
        ("model.rgbd.temporal_history_size=16.0", "must be an integer"),
        ("model.rgbd.temporal_min_samples=16.0", "must be an integer"),
        ("model.rgbd.global_every_steps=0", "global_every_steps must be a positive integer"),
        ("model.rgbd.proposal_count=0", "proposal_count must be integer one or two"),
        ("model.rgbd.proposal_count=1.0", "proposal_count must be integer one or two"),
        ("model.rgbd.proposal_count=2", "object counts to equal proposal_count"),
        ("model.rgbd.world_radius=0", "world_radius must be finite and positive"),
        ("model.rgbd.linear_drag=0", "linear_drag must be finite and positive"),
        (
            "model.rgbd.maximum_surface_radius_relative_error=1.1",
            "must be no greater than one",
        ),
        (
            "model.rgbd.chromatic_centre_blend=1.1",
            "must be no greater than one",
        ),
        ("model.rgbd.enabled=1", "must be boolean"),
        ("model.dynamics.analytic_free_motion_only=1", "must be boolean"),
        ("model.filter.enable_learned_corrector=1", "must be boolean"),
        ("model.filter.direct_metric_position_update=1", "must be boolean"),
        (
            "model.rgb.enabled=true",
            "direct metric position updates require an exclusive RGB-D observation path",
        ),
        ("evaluation.rgb_only=true", "RGB-D runtime requires evaluation.rgb_only=false"),
    ],
)
def test_rgbd_online_semantics_fail_closed(override: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        load_config(CONFIG_DIR / "rgbd_online_free_motion_cpu.yaml", overrides=[override])


def test_two_object_rgbd_requires_exact_observable_appearance_capacity() -> None:
    base = load_config(CONFIG_DIR / "rgbd_online_free_motion_cpu.yaml")
    two_object = replace(
        base,
        model=replace(
            base.model,
            max_objects=2,
            state=replace(base.model.state, appearance_dim=3),
            rgbd=replace(base.model.rgbd, proposal_count=2),
        ),
        simulator=replace(base.simulator, min_objects=2, max_objects=2),
    )
    two_object.validate()

    for appearance_dim in (2, 4):
        invalid = replace(
            two_object,
            model=replace(
                two_object.model,
                state=replace(two_object.model.state, appearance_dim=appearance_dim),
            ),
        )
        with pytest.raises(ValueError, match="appearance_dim exactly three"):
            invalid.validate()


def _nested_value(mapping: dict[str, object], dotted_path: str) -> object:
    value: object = mapping
    for key in dotted_path.split("."):
        assert isinstance(value, dict)
        value = value[key]
    return value


def _assert_exact_value_and_type(actual: object, expected: object) -> None:
    assert type(actual) is type(expected)
    if isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_exact_value_and_type(actual_item, expected_item)
    else:
        assert actual == expected


def _assert_identifiable_drag_profile_contract(config: object) -> None:
    accepted = load_config(CONFIG_DIR / "rgbd_two_visible_orbital_camera_cpu.yaml")

    project = config.project
    assert project.name == "orpheus-rgbd-two-visible-orbital-camera-identifiable-drag-v1-cpu"
    assert type(project.seed) is int
    assert project.seed == 0
    assert (
        replace(
            project,
            name=accepted.project.name,
            seed=accepted.project.seed,
        )
        == accepted.project
    )

    assert config.device == accepted.device
    assert type(config.device.preference) is str
    assert config.device.preference == "cpu"
    assert type(config.device.closed_loop_preference) is str
    assert config.device.closed_loop_preference == "cpu"
    assert config.device.cuda_amp is False
    assert config.device.compile is False

    simulator = config.simulator
    assert (
        replace(
            simulator,
            drag_range=accepted.simulator.drag_range,
            initial_speed_range=accepted.simulator.initial_speed_range,
        )
        == accepted.simulator
    )
    assert simulator.drag_range == (0.045, 0.325)
    assert all(type(value) is float for value in simulator.drag_range)
    assert simulator.gravity == (0.0, 0.0, 0.0)
    assert all(type(value) is float for value in simulator.gravity)
    assert type(simulator.min_objects) is int
    assert simulator.min_objects == 2
    assert type(simulator.max_objects) is int
    assert simulator.max_objects == 2
    assert simulator.radius_range == (0.21, 0.21)
    assert all(type(value) is float for value in simulator.radius_range)
    assert simulator.initial_speed_range == (0.05, 0.071)
    assert all(type(value) is float for value in simulator.initial_speed_range)
    assert type(simulator.camera_motion) is str
    assert simulator.camera_motion == "orbit"
    assert simulator.known_camera_pose is True
    assert simulator.ensure_collision is False
    assert type(simulator.external_impulse_probability) is float
    assert simulator.external_impulse_probability == 0.0
    assert type(simulator.scenario_mixture) is tuple
    assert simulator.scenario_mixture == ("baseline",)
    assert type(simulator.scenario_mixture[0]) is str
    assert type(simulator.sequence_frames) is int
    assert simulator.sequence_frames == 56

    assert config.model.rgb == accepted.model.rgb
    rgbd = config.model.rgbd
    assert (
        replace(
            rgbd,
            linear_drag=accepted.model.rgbd.linear_drag,
            temporal_drag_estimation_enabled=(accepted.model.rgbd.temporal_drag_estimation_enabled),
        )
        == accepted.model.rgbd
    )
    assert rgbd.enabled is True
    assert type(rgbd.proposal_count) is int
    assert rgbd.proposal_count == 2
    assert type(rgbd.world_radius) is float
    assert rgbd.world_radius == 0.21
    assert type(rgbd.linear_drag) is float
    assert rgbd.linear_drag == 0.185
    assert type(rgbd.minimum_silhouette_gap_pixels) is float
    assert rgbd.minimum_silhouette_gap_pixels == 2.0
    assert type(rgbd.minimum_boundary_clearance_pixels) is float
    assert rgbd.minimum_boundary_clearance_pixels == 2.0
    assert type(rgbd.temporal_history_size) is int
    assert rgbd.temporal_history_size == 16
    assert type(rgbd.temporal_min_samples) is int
    assert rgbd.temporal_min_samples == 16
    assert rgbd.temporal_drag_estimation_enabled is True
    assert type(rgbd.temporal_drag_minimum) is float
    assert rgbd.temporal_drag_minimum == 0.01
    assert type(rgbd.temporal_drag_maximum) is float
    assert rgbd.temporal_drag_maximum == 0.36
    assert type(rgbd.temporal_drag_grid_points) is int
    assert rgbd.temporal_drag_grid_points == 257
    assert type(rgbd.temporal_drag_noise_floor_m) is float
    assert rgbd.temporal_drag_noise_floor_m == 2.0e-5
    assert type(rgbd.temporal_drag_minimum_excitation_m) is float
    assert rgbd.temporal_drag_minimum_excitation_m == 0.015
    assert type(rgbd.temporal_drag_minimum_profile_information) is float
    assert rgbd.temporal_drag_minimum_profile_information == 1.0
    assert type(rgbd.temporal_drag_maximum_boundary_mass) is float
    assert rgbd.temporal_drag_maximum_boundary_mass == 0.01
    assert type(rgbd.temporal_drag_log_parameter_variance_floor) is float
    assert rgbd.temporal_drag_log_parameter_variance_floor == 1.0e-4
    assert type(rgbd.temporal_drag_log_parameter_variance_ceiling) is float
    assert rgbd.temporal_drag_log_parameter_variance_ceiling == 0.25

    assert config.model.dynamics == accepted.model.dynamics
    assert config.model.dynamics.analytic_free_motion_only is True
    assert config.model.association == accepted.model.association
    assert config.model.identification == accepted.model.identification
    assert config.model.identification.enabled is False

    filter_config = config.model.filter
    assert (
        replace(
            filter_config,
            min_log_variance=accepted.model.filter.min_log_variance,
        )
        == accepted.model.filter
    )
    assert type(filter_config.min_log_variance) is float
    assert filter_config.min_log_variance == -30.0
    assert type(filter_config.max_log_variance) is float
    assert filter_config.max_log_variance == 8.0

    assert config.runtime == accepted.runtime
    assert type(config.runtime.modality) is str
    assert config.runtime.modality == "rgbd"
    assert config.runtime.enable_debug_oracle is False
    assert (
        replace(
            config.training,
            train_episodes=accepted.training.train_episodes,
            validation_episodes=accepted.training.validation_episodes,
        )
        == accepted.training
    )
    assert type(config.training.train_episodes) is int
    assert config.training.train_episodes == 64
    assert type(config.training.validation_episodes) is int
    assert config.training.validation_episodes == 64
    assert replace(config.evaluation, episodes=accepted.evaluation.episodes) == accepted.evaluation
    assert type(config.evaluation.episodes) is int
    assert config.evaluation.episodes == 64
    assert config.evaluation.horizons_seconds == (0.1, 0.25, 0.5, 1.0, 2.0)
    assert all(type(value) is float for value in config.evaluation.horizons_seconds)
    assert config.demo == accepted.demo
    assert type(config.demo.seed) is int


def test_identifiable_drag_profile_raw_values_are_explicit_and_exactly_typed() -> None:
    raw = yaml.safe_load(IDENTIFIABLE_DRAG_PROFILE.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    expected = {
        "project.seed": 0,
        "device.preference": "cpu",
        "device.closed_loop_preference": "cpu",
        "device.cuda_amp": False,
        "device.compile": False,
        "simulator.image_size": [64, 64],
        "simulator.sequence_frames": 56,
        "simulator.min_objects": 2,
        "simulator.max_objects": 2,
        "simulator.gravity": [0.0, 0.0, 0.0],
        "simulator.radius_range": [0.21, 0.21],
        "simulator.drag_range": [0.045, 0.325],
        "simulator.initial_speed_range": [0.05, 0.071],
        "simulator.camera_motion": "orbit",
        "simulator.known_camera_pose": True,
        "simulator.ensure_collision": False,
        "simulator.external_impulse_probability": 0.0,
        "simulator.scenario_mixture": ["baseline"],
        "model.rgb.enabled": False,
        "model.rgbd.enabled": True,
        "model.rgbd.proposal_count": 2,
        "model.rgbd.world_radius": 0.21,
        "model.rgbd.linear_drag": 0.185,
        "model.rgbd.minimum_silhouette_gap_pixels": 2.0,
        "model.rgbd.minimum_boundary_clearance_pixels": 2.0,
        "model.rgbd.maximum_surface_radius_relative_error": 0.05,
        "model.rgbd.temporal_history_size": 16,
        "model.rgbd.temporal_min_samples": 16,
        "model.rgbd.temporal_drag_estimation_enabled": True,
        "model.rgbd.temporal_drag_minimum": 0.01,
        "model.rgbd.temporal_drag_maximum": 0.36,
        "model.rgbd.temporal_drag_grid_points": 257,
        "model.rgbd.temporal_drag_noise_floor_m": 2.0e-5,
        "model.rgbd.temporal_drag_minimum_excitation_m": 0.015,
        "model.rgbd.temporal_drag_minimum_profile_information": 1.0,
        "model.rgbd.temporal_drag_maximum_boundary_mass": 0.01,
        "model.rgbd.temporal_drag_log_parameter_variance_floor": 1.0e-4,
        "model.rgbd.temporal_drag_log_parameter_variance_ceiling": 0.25,
        "model.dynamics.analytic_free_motion_only": True,
        "model.filter.min_log_variance": -30.0,
        "model.filter.max_log_variance": 8.0,
        "model.filter.enable_learned_corrector": False,
        "model.filter.direct_metric_position_update": True,
        "model.identification.enabled": False,
        "runtime.modality": "rgbd",
        "runtime.enable_debug_oracle": False,
        "training.train_episodes": 64,
        "training.validation_episodes": 64,
        "evaluation.episodes": 64,
        "evaluation.horizons_seconds": [0.1, 0.25, 0.5, 1.0, 2.0],
        "demo.seed": 200000,
    }

    for dotted_path, expected_value in expected.items():
        _assert_exact_value_and_type(_nested_value(raw, dotted_path), expected_value)


def test_identifiable_drag_profile_resolves_and_roundtrips_exactly(tmp_path: Path) -> None:
    config = load_config(IDENTIFIABLE_DRAG_PROFILE)
    _assert_identifiable_drag_profile_contract(config)

    resolved_path = tmp_path / "identifiable-drag-resolved.yaml"
    save_resolved_config(config, resolved_path)
    restored = load_config(resolved_path)

    _assert_identifiable_drag_profile_contract(restored)
    assert restored.to_dict() == config.to_dict()


def test_identifiable_drag_profile_seed_fields_are_nonauthoritative_placeholders() -> None:
    config = load_config(IDENTIFIABLE_DRAG_PROFILE)

    # Governed scenes are selected only by split and ordinal in the qualification
    # family. These common-tool fields deliberately encode no scene namespace.
    assert config.project.seed == 0
    assert config.demo.seed == 200000
    assert config.training.train_episodes == 64
    assert config.training.validation_episodes == 64
    assert config.evaluation.episodes == 64


def test_identifiable_drag_profile_alone_lowers_filter_variance_clamp() -> None:
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        config = load_config(path)
        expected_minimum = -30.0 if path == IDENTIFIABLE_DRAG_PROFILE else -12.0
        assert type(config.model.filter.min_log_variance) is float
        assert config.model.filter.min_log_variance == expected_minimum
        assert type(config.model.filter.max_log_variance) is float
        assert config.model.filter.max_log_variance == 8.0


@pytest.mark.parametrize(
    "overrides",
    [
        ("project.seed=1",),
        ("project.seed=0.0",),
        ("device.preference=auto",),
        ("device.cuda_amp=true",),
        ("simulator.sequence_frames=55",),
        ("simulator.drag_range=[0.046,0.324]",),
        ("simulator.initial_speed_range=[0.05,0.07]",),
        ("simulator.camera_motion=static",),
        ("model.rgbd.linear_drag=0.184",),
        ("model.rgbd.foreground_threshold=0.05",),
        ("model.rgbd.minimum_silhouette_gap_pixels=2.1",),
        ("model.rgbd.temporal_drag_minimum=0.011",),
        ("model.rgbd.temporal_drag_maximum=0.359",),
        ("model.rgbd.temporal_drag_grid_points=259",),
        ("model.rgbd.temporal_drag_noise_floor_m=0.000021",),
        ("model.rgbd.temporal_drag_minimum_excitation_m=0.014",),
        ("model.rgbd.temporal_drag_minimum_profile_information=1.1",),
        ("model.rgbd.temporal_drag_minimum_profile_information=1",),
        ("model.rgbd.temporal_drag_maximum_boundary_mass=0.02",),
        ("model.rgbd.temporal_drag_log_parameter_variance_floor=0.0002",),
        ("model.rgbd.temporal_drag_log_parameter_variance_ceiling=0.24",),
        ("model.filter.min_log_variance=-29.0",),
        ("model.filter.min_log_variance=-30",),
        ("model.filter.max_log_variance=7.0",),
        ("model.filter.max_log_variance=8",),
        ("model.association.geometry_weight=0.9",),
        ("training.train_episodes=63",),
        ("training.validation_episodes=63",),
        ("evaluation.episodes=63",),
        ("evaluation.horizons_seconds=[0.1,0.25,0.5,1.0,2.1]",),
        ("demo.seed=200000.0",),
    ],
)
def test_identifiable_drag_profile_contract_detects_valid_negative_mutations(
    overrides: tuple[str, ...],
) -> None:
    mutated = load_config(IDENTIFIABLE_DRAG_PROFILE, overrides=overrides)
    with pytest.raises(AssertionError):
        _assert_identifiable_drag_profile_contract(mutated)


def _enabled_rgbd_temporal_drag_config():
    base = load_config(CONFIG_DIR / "rgbd_two_visible_orbital_camera_cpu.yaml")
    return replace(
        base,
        simulator=replace(base.simulator, drag_range=(0.03, 0.28)),
        model=replace(
            base.model,
            rgbd=replace(
                base.model.rgbd,
                temporal_drag_estimation_enabled=True,
            ),
        ),
    )


def test_rgbd_temporal_drag_mode_binds_variable_interior_family_and_exact_defaults() -> None:
    config = _enabled_rgbd_temporal_drag_config()

    config.validate()
    rgbd = config.model.rgbd
    assert rgbd.temporal_drag_estimation_enabled
    assert (rgbd.temporal_drag_minimum, rgbd.temporal_drag_maximum) == (0.01, 0.36)
    assert rgbd.temporal_drag_grid_points == 257
    assert rgbd.temporal_drag_noise_floor_m == pytest.approx(2.0e-5)
    assert rgbd.temporal_drag_minimum_excitation_m == pytest.approx(0.015)
    assert rgbd.temporal_drag_minimum_profile_information == pytest.approx(1.0)
    assert rgbd.temporal_drag_maximum_boundary_mass == pytest.approx(0.01)
    assert rgbd.temporal_drag_log_parameter_variance_floor == pytest.approx(1.0e-4)
    assert rgbd.temporal_drag_log_parameter_variance_ceiling == pytest.approx(0.25)

    # The magnitude range is behaviorally inert while impulses are disabled;
    # an exact qualification profile may bind it without narrowing this mode.
    replace(
        config,
        simulator=replace(config.simulator, external_impulse_range=(0.25, 0.8)),
    ).validate()


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("fixed_drag", "variable simulator drag range"),
        ("boundary_drag", "strictly inside the estimator bounds"),
        ("gravity", "exact float zero gravity"),
        ("boolean_gravity", "exact float zero gravity"),
        ("ensured_collision", "ensure_collision=false"),
        ("external_impulse", "external_impulse_probability=0.0"),
        ("boolean_external_impulse", "exact float"),
        ("integer_external_impulse", "exact float"),
        ("contact_scenarios", "exactly the baseline simulator scenario"),
        ("intervention_scenario", "exactly the baseline simulator scenario"),
        ("nonanalytic", "analytic_free_motion_only"),
        ("identifier", "recurrent identifier to be disabled"),
        ("single_object_family", "exactly two-object family"),
        ("short_history", "exactly 16 history samples"),
        ("even_grid", "odd integer"),
        ("boolean_boundary_mass", "must lie in"),
        ("variance_order", "variance bounds must be ordered"),
    ],
)
def test_rgbd_temporal_drag_mode_rejects_out_of_scope_runtime_semantics(
    mutation: str,
    match: str,
) -> None:
    config = _enabled_rgbd_temporal_drag_config()
    if mutation == "fixed_drag":
        config = replace(
            config,
            simulator=replace(config.simulator, drag_range=(0.08, 0.08)),
        )
    elif mutation == "boundary_drag":
        config = replace(
            config,
            simulator=replace(config.simulator, drag_range=(0.01, 0.28)),
        )
    elif mutation == "gravity":
        config = replace(
            config,
            simulator=replace(config.simulator, gravity=(0.0, -0.1, 0.0)),
        )
    elif mutation == "boolean_gravity":
        config = replace(
            config,
            simulator=replace(config.simulator, gravity=(False, False, False)),
        )
    elif mutation == "ensured_collision":
        config = replace(
            config,
            simulator=replace(config.simulator, ensure_collision=True),
        )
    elif mutation == "external_impulse":
        config = replace(
            config,
            simulator=replace(config.simulator, external_impulse_probability=0.01),
        )
    elif mutation == "boolean_external_impulse":
        config = replace(
            config,
            simulator=replace(config.simulator, external_impulse_probability=False),
        )
    elif mutation == "integer_external_impulse":
        config = replace(
            config,
            simulator=replace(config.simulator, external_impulse_probability=0),
        )
    elif mutation == "contact_scenarios":
        config = replace(
            config,
            simulator=replace(
                config.simulator,
                scenario_mixture=("baseline", "elastic_pairs", "damped_contacts"),
            ),
        )
    elif mutation == "intervention_scenario":
        config = replace(
            config,
            simulator=replace(config.simulator, scenario_mixture=("impulse_perturbation",)),
        )
    elif mutation == "nonanalytic":
        config = replace(
            config,
            model=replace(
                config.model,
                dynamics=replace(config.model.dynamics, analytic_free_motion_only=False),
            ),
        )
    elif mutation == "identifier":
        config = replace(
            config,
            model=replace(
                config.model,
                identification=replace(config.model.identification, enabled=True),
            ),
        )
    elif mutation == "single_object_family":
        config = replace(
            config,
            simulator=replace(config.simulator, min_objects=1, max_objects=1),
            model=replace(
                config.model,
                max_objects=1,
                rgbd=replace(config.model.rgbd, proposal_count=1),
            ),
        )
    elif mutation == "short_history":
        config = replace(
            config,
            model=replace(
                config.model,
                rgbd=replace(
                    config.model.rgbd,
                    temporal_history_size=15,
                    temporal_min_samples=15,
                ),
            ),
        )
    elif mutation == "even_grid":
        config = replace(
            config,
            model=replace(
                config.model,
                rgbd=replace(config.model.rgbd, temporal_drag_grid_points=256),
            ),
        )
    elif mutation == "boolean_boundary_mass":
        config = replace(
            config,
            model=replace(
                config.model,
                rgbd=replace(
                    config.model.rgbd,
                    temporal_drag_maximum_boundary_mass=False,
                ),
            ),
        )
    elif mutation == "variance_order":
        config = replace(
            config,
            model=replace(
                config.model,
                rgbd=replace(
                    config.model.rgbd,
                    temporal_drag_log_parameter_variance_floor=0.3,
                ),
            ),
        )
    else:  # pragma: no cover - parameter table owns this branch
        raise AssertionError(mutation)

    with pytest.raises(ValueError, match=match):
        config.validate()


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
    with pytest.raises(
        ValueError,
        match="simulator.ensured_pair_scene_resample_attempts must be a positive integer",
    ):
        load_config(
            CONFIG_DIR / "default.yaml",
            overrides=["simulator.ensured_pair_scene_resample_attempts=0"],
        )


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


def test_historical_sustained_profile_pins_legacy_horizon_semantics() -> None:
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
