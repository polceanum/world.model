"""Integration coverage for deterministic simulation, rendering, and datasets."""

from __future__ import annotations

from dataclasses import replace

import torch

from world_model.datasets import (
    SyntheticSphereDataset,
    assert_disjoint_manifests,
    collate_episodes,
    make_seed_manifest,
)
from world_model.simulator import SphereWorld, SphereWorldConfig, generate_episode, validate_episode
from world_model.training.loop import _select_rollout_anchor_frames
from world_model.utils.config import load_config


def _test_config() -> SphereWorldConfig:
    return SphereWorldConfig(
        image_size=(32, 40),
        frame_rate=30.0,
        physics_rate=120.0,
        sequence_frames=20,
        min_objects=2,
        max_objects=2,
        padding_max_objects=4,
        camera_motion="combined",
        render_noise_std=0.0,
        ensure_collision=True,
    )


def test_episode_is_padded_labelled_and_physically_exercises_contacts() -> None:
    config = _test_config()
    episode = generate_episode(config, seed=123)
    validate_episode(episode, config)

    assert episode["rgb"].shape == (20, 3, 32, 40)
    assert episode["rgb"].dtype == torch.float32
    assert episode["objects"]["position"].shape == (20, 4, 3)
    assert episode["objects"]["id"].dtype == torch.int64
    assert episode["objects"]["active"].dtype == torch.bool
    assert episode["labels"]["segmentation_mask"].shape == (20, 4, 32, 40)
    assert episode["events"]["pair_collision"].shape == (20, 4, 4)
    assert episode["events"]["wall_collision"].shape == (20, 4, 4)

    inactive = ~episode["objects"]["active"]
    assert torch.all(episode["objects"]["id"][inactive] == -1)
    assert episode["events"]["pair_collision"].any()
    assert torch.all(
        (episode["objects"]["visible_fraction"] >= 0)
        & (episode["objects"]["visible_fraction"] <= 1)
    )
    # Raster visibility assigns no pixel to more than one object.
    visible_masks = episode["labels"]["segmentation_mask"]
    assert int(visible_masks.sum(dim=2).sum(dim=2).max()) <= 32 * 40
    assert int(visible_masks.sum(dim=1).max()) <= 1

    camera = episode["camera"]
    identity = camera["world_from_camera"] @ camera["camera_from_world"]
    torch.testing.assert_close(
        identity,
        torch.eye(4).expand_as(identity),
        atol=3.0e-5,
        rtol=3.0e-5,
    )
    assert not torch.equal(camera["position"][0], camera["position"][-1])


def test_fixed_seed_generation_is_exactly_repeatable_and_seed_sensitive() -> None:
    config = _test_config()
    first = generate_episode(config, seed=7)
    repeated = generate_episode(config, seed=7)
    different = generate_episode(config, seed=8)

    torch.testing.assert_close(first["rgb"], repeated["rgb"], rtol=0, atol=0)
    torch.testing.assert_close(
        first["objects"]["position"],
        repeated["objects"]["position"],
        rtol=0,
        atol=0,
    )
    assert torch.equal(
        first["events"]["pair_collision"],
        repeated["events"]["pair_collision"],
    )
    assert not torch.equal(first["objects"]["mass"][0], different["objects"]["mass"][0])


def test_ensured_pair_collision_is_not_confounded_with_floor_impact() -> None:
    config = SphereWorldConfig(
        image_size=(48, 48),
        frame_rate=20.0,
        physics_rate=120.0,
        sequence_frames=32,
        min_objects=2,
        max_objects=2,
        camera_motion="fixed",
        render_noise_std=0.0,
        ensure_collision=True,
    )
    episode = generate_episode(config, seed=200000)
    pair_frames = torch.nonzero(
        episode["events"]["pair_collision"].any(dim=(1, 2)),
        as_tuple=False,
    ).flatten()

    assert pair_frames.numel() == 1
    collision_frame = int(pair_frames[0])
    assert not episode["events"]["ground_collision"][collision_frame, :2].any()

    position = episode["objects"]["position"][collision_frame, :2]
    velocity = episode["objects"]["velocity"][collision_frame, :2]
    normal = torch.nn.functional.normalize(position[1] - position[0], dim=0)
    separating_speed = torch.dot(velocity[1] - velocity[0], normal)
    assert separating_speed > 0.5


def test_seeded_scenario_mixture_exercises_distinct_physical_regimes() -> None:
    scenarios = (
        "baseline",
        "reference_pairs",
        "elastic_pairs",
        "damped_contacts",
        "impulse_perturbation",
        "camera_parallax",
        "glancing_impacts",
        "heavy_light_impacts",
    )
    config = SphereWorldConfig(
        image_size=(24, 24),
        sequence_frames=4,
        min_objects=3,
        max_objects=3,
        scenario_mixture=scenarios,
    )

    worlds = [SphereWorld(config, seed=index) for index in range(len(scenarios))]
    assert [world.scenario_name for world in worlds] == list(scenarios)
    assert worlds[1].config.mass_range == (0.85, 1.15)
    assert worlds[1].config.friction_range == (0.0, 0.04)
    assert worlds[2].config.restitution_range == (0.78, 0.95)
    assert worlds[2].config.drag_range == (0.005, 0.04)
    assert worlds[3].config.restitution_range == (0.18, 0.42)
    assert worlds[3].config.friction_range == (0.28, 0.55)
    assert worlds[4].config.external_impulse_probability == 0.02
    assert worlds[4].config.external_impulse_range == (0.25, 0.8)
    assert worlds[5].config.camera_motion == "combined"
    assert worlds[5].config.camera_zoom_amplitude == 0.12
    assert worlds[6].config.initial_speed_range == (1.1, 1.8)
    assert worlds[6].config.friction_range == (0.01, 0.08)
    assert worlds[7].config.mass_range == (0.3, 2.5)

    episode = generate_episode(config, seed=3)
    assert episode["metadata"]["scenario"] == "damped_contacts"
    active = episode["objects"]["active"][0]
    assert torch.all(episode["objects"]["restitution"][0, active, 0] <= 0.42)
    assert torch.all(episode["objects"]["drag"][0, active, 0] >= 0.18)


def test_impulse_validation_slice_keeps_real_events_and_causal_horizon_support() -> None:
    config = load_config("configs/sustained_accuracy_mps_v3.yaml")
    simulator = SphereWorldConfig.from_config(config)
    impulse_seeds = (100004, 100012, 100020, 100028)
    anchors = _select_rollout_anchor_frames(
        config,
        window_start=0,
        window_stop=simulator.sequence_frames,
        total_frames=simulator.sequence_frames,
        rollout_anchors_per_window=(config.training.validation_rollout_anchors_per_episode),
    )
    episodes: list[tuple[torch.Tensor, torch.Tensor]] = []
    impulse_count = 0
    for seed in impulse_seeds:
        world = SphereWorld(simulator, seed)
        assert world.scenario_name == "impulse_perturbation"
        active_frames = [world.state.active.clone()]
        actuation_frames = [torch.zeros_like(world.state.active)]
        for frame_index in range(1, simulator.sequence_frames):
            world.apply_lifecycle(frame_index)
            event = world.step(simulator.observation_dt)
            active_frames.append(world.state.active.clone())
            actuated = torch.linalg.vector_norm(event.external_impulse, dim=-1) > 0
            actuation_frames.append(actuated)
            impulse_count += int(actuated.sum())
        episodes.append(
            (
                torch.stack(active_frames),
                torch.stack(actuation_frames),
            )
        )

    assert impulse_count > 0
    for horizon_seconds in config.evaluation.horizons_seconds:
        offset = max(1, round(horizon_seconds * simulator.frame_rate))
        predictable_target_count = 0
        for active, actuated in episodes:
            for anchor in anchors:
                target = anchor + offset
                if target >= simulator.sequence_frames:
                    continue
                if not bool(actuated[anchor + 1 : target + 1].any()):
                    predictable_target_count += int(active[target].sum())
        assert (
            predictable_target_count
            >= config.training.validation_minimum_predictable_target_count_per_scenario_horizon
        )


def test_glancing_scenario_has_nonzero_impact_parameter_and_collides() -> None:
    config = SphereWorldConfig(
        image_size=(16, 16),
        frame_rate=30.0,
        physics_rate=120.0,
        sequence_frames=48,
        min_objects=2,
        max_objects=2,
        padding_max_objects=2,
        camera_motion="fixed",
        render_noise_std=0.0,
        ensure_collision=True,
        scenario_mixture=("glancing_impacts",),
    )
    episode = generate_episode(config, seed=0)
    initial_position = episode["objects"]["position"][0, :2]
    initial_velocity = episode["objects"]["velocity"][0, :2]
    relative_position = initial_position[1] - initial_position[0]
    relative_velocity = initial_velocity[1] - initial_velocity[0]
    impact_parameter = torch.linalg.vector_norm(
        torch.linalg.cross(relative_position, relative_velocity, dim=0)
    ) / torch.linalg.vector_norm(relative_velocity)
    radial_velocity = (
        torch.dot(relative_velocity, relative_position)
        / torch.dot(relative_position, relative_position)
        * relative_position
    )
    tangential_speed = torch.linalg.vector_norm(relative_velocity - radial_velocity)

    assert episode["metadata"]["scenario"] == "glancing_impacts"
    assert impact_parameter > 0.15
    assert tangential_speed > 0.1
    assert episode["events"]["pair_collision"].any()


def test_dataset_splits_and_collation_preserve_batch_time_object_order() -> None:
    config = _test_config()
    train_manifest = make_seed_manifest("train", 2)
    validation_manifest = make_seed_manifest("validation", 2)
    test_manifest = make_seed_manifest("test", 2)
    assert_disjoint_manifests(train_manifest, validation_manifest, test_manifest)

    dataset = SyntheticSphereDataset(
        config,
        split="train",
        seeds=train_manifest,
        memory_cache=True,
    )
    first = dataset[0]
    first["rgb"].zero_()
    # Cached records are cloned, so consumer mutation cannot contaminate data.
    assert dataset[0]["rgb"].abs().sum() > 0
    batch = collate_episodes([dataset[0], dataset[1]])
    assert batch["rgb"].shape == (2, 20, 3, 32, 40)
    assert batch["timestamps"].shape == (2, 20)
    assert batch["objects"]["position"].shape == (2, 20, 4, 3)
    assert batch["seed"].tolist() == list(train_manifest.seeds)
    assert batch["metadata"]["split"] == ["train", "train"]


def test_ood_split_uses_explicit_held_out_parameter_combination() -> None:
    dataset = SyntheticSphereDataset(
        _test_config(),
        split="ood",
        num_episodes=1,
    )
    episode = dataset[0]
    active = episode["objects"]["active"][0]
    restitution = episode["objects"]["restitution"][0, active, 0]
    drag = episode["objects"]["drag"][0, active, 0]

    assert torch.all(restitution <= 0.4)
    assert torch.all(drag >= 0.19)
    assert episode["metadata"]["distribution"] == "ood"


def test_ood_ranges_take_precedence_over_named_scenario_ranges() -> None:
    config = replace(
        _test_config(),
        scenario_mixture=("elastic_pairs",),
    )
    dataset = SyntheticSphereDataset(
        config,
        split="ood",
        num_episodes=1,
    )
    episode = dataset[0]
    active = episode["objects"]["active"][0]
    restitution = episode["objects"]["restitution"][0, active, 0]
    drag = episode["objects"]["drag"][0, active, 0]

    assert episode["metadata"]["scenario"] == "elastic_pairs"
    assert episode["metadata"]["distribution"] == "ood"
    assert torch.all((restitution >= 0.18) & (restitution <= 0.4))
    assert torch.all((drag >= 0.19) & (drag <= 0.32))


def test_render_noise_does_not_change_physical_trajectory_or_events() -> None:
    clean_config = SphereWorldConfig(
        image_size=(8, 8),
        sequence_frames=20,
        min_objects=2,
        max_objects=2,
        padding_max_objects=2,
        scenario_mixture=("impulse_perturbation",),
        render_noise_std=0.0,
    )
    noisy_config = replace(clean_config, render_noise_std=0.1)
    clean = generate_episode(clean_config, seed=0)
    noisy = generate_episode(noisy_config, seed=0)

    torch.testing.assert_close(
        clean["objects"]["position"],
        noisy["objects"]["position"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        clean["objects"]["velocity"],
        noisy["objects"]["velocity"],
        rtol=0,
        atol=0,
    )
    assert torch.equal(
        clean["events"]["external_impulse"],
        noisy["events"]["external_impulse"],
    )
    assert torch.equal(clean["events"]["collision"], noisy["events"]["collision"])
    assert not torch.equal(clean["rgb"], noisy["rgb"])
