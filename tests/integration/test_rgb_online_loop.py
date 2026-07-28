from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from world_model.observations import ObservationPacket
from world_model.observations.rgb import RGBTemporalPositionHistory
from world_model.runtime import OnlineWorldModel
from world_model.simulator import generate_episode
from world_model.utils.config import OrpheusConfig


def _small_rgb_config() -> OrpheusConfig:
    config = OrpheusConfig()
    simulator = replace(
        config.simulator,
        image_size=(32, 32),
        min_objects=2,
        max_objects=3,
    )
    state = replace(
        config.model.state,
        geometry_dim=2,
        appearance_dim=8,
        residual_dynamics_dim=4,
        modal_count=1,
        modal_dim=2,
        parameter_memory_dim=16,
        global_dim=4,
    )
    rgb = replace(
        config.model.rgb,
        backbone_channels=(8, 16, 24, 32),
        feature_dim=16,
        proposal_queries=4,
        roi_size=8,
        global_every_steps=5,
        global_uncertainty_threshold=4.0,
        surprise_threshold=8.0,
    )
    dynamics = replace(
        config.model.dynamics,
        hidden_dim=24,
    )
    filtering = replace(config.model.filter, hidden_dim=32)
    identification = replace(
        config.model.identification,
        hidden_dim=16,
    )
    model = replace(
        config.model,
        max_objects=3,
        state=state,
        rgb=rgb,
        dynamics=dynamics,
        filter=filtering,
        identification=identification,
    )
    return replace(config, simulator=simulator, model=model)


def _rgb_packet(timestamp: float, shift: int = 0) -> ObservationPacket:
    image = torch.zeros(3, 32, 32)
    image[0, 10:17, 8 + shift : 15 + shift] = 1.0
    image[1, 18:24, 20 - shift : 26 - shift] = 0.8
    intrinsics = torch.tensor([[30.0, 0.0, 15.5], [0.0, 30.0, 15.5], [0.0, 0.0, 1.0]])
    world_from_camera = torch.eye(4)
    world_from_camera[2, 3] = -4.0
    return ObservationPacket(
        modality="rgb",
        sensor_id="camera",
        timestamp=timestamp,
        payload=image,
        calibration={
            "intrinsics": intrinsics,
            "world_from_camera": world_from_camera,
        },
        frame_id="camera:camera",
    )


def test_rgb_only_runtime_initialises_then_uses_cached_fast_path() -> None:
    torch.manual_seed(3)
    model = OnlineWorldModel.from_config(_small_rgb_config(), device="cpu")
    initial = model.ingest(_rgb_packet(0.0))
    assert initial.objects.active.any()
    first_ids = initial.objects.object_id[initial.objects.active].clone()
    posterior = model.ingest(_rgb_packet(1.0 / 30.0, shift=1))
    assert model.diagnostics.records[0].observation_mode == "GLOBAL_DISCOVERY"
    assert model.diagnostics.latest is not None
    assert model.diagnostics.latest.observation_mode == "FAST_ROI"
    assert "camera" in model.state.caches
    assert not model.diagnostics.oracle_used
    assert posterior.timestamp.item() == torch.tensor(1.0 / 30.0).item()
    assert torch.equal(
        posterior.objects.object_id[posterior.objects.active][: first_ids.numel()],
        first_ids,
    )
    assert torch.isfinite(posterior.objects.position).all()
    assert torch.isfinite(model.predict([0.1]).positions).all()


def test_runtime_exposes_detached_last_measurements_and_reset_clears_them() -> None:
    model = OnlineWorldModel.from_config(_small_rgb_config(), device="cpu")
    assert model.last_measurements is None
    model.ingest(_rgb_packet(0.0))
    measurements = model.last_measurements
    assert measurements is not None
    assert measurements.modality == "rgb"
    assert not measurements.values.requires_grad
    assert measurements.values.grad_fn is None
    assert all(not value.requires_grad for value in measurements.auxiliary.values())

    model.reset()
    assert model.last_measurements is None


def test_global_pass_invalidates_fast_roi_cache() -> None:
    torch.manual_seed(3)
    model = OnlineWorldModel.from_config(_small_rgb_config(), device="cpu")
    model.ingest(_rgb_packet(0.0))
    model.ingest(_rgb_packet(1.0 / 30.0, shift=1))
    assert "camera" in model.caches

    model.diagnostics.reset()
    scheduler_state = model.scheduler.state_for("camera")
    scheduler_state.last_surprise = 0.0
    scheduler_state.association_failures = 0
    scheduler_state.steps_since_global = model.scheduler.global_every_steps
    model.ingest(_rgb_packet(2.0 / 30.0, shift=1))

    assert model.diagnostics.latest is not None
    assert model.diagnostics.latest.observation_mode == "GLOBAL_DISCOVERY"
    assert "camera" not in model.caches


def test_global_pass_preserves_separate_temporal_history_and_reset_clears_it() -> None:
    torch.manual_seed(3)
    config = _small_rgb_config()
    config = replace(
        config,
        model=replace(
            config.model,
            rgb=replace(
                config.model.rgb,
                temporal_velocity_enabled=True,
                temporal_velocity_history_size=3,
            ),
        ),
    )
    model = OnlineWorldModel.from_config(config, device="cpu")
    model.ingest(_rgb_packet(0.0))
    model.ingest(_rgb_packet(1.0 / 30.0, shift=1))
    assert "camera" in model.caches
    assert "camera" in model.state.temporal_histories
    assert model.last_measurements is not None
    assert {"velocity", "velocity_from_position"}.isdisjoint(
        model.last_measurements.supported_state_fields
    )
    history_before = model.state.temporal_histories["camera"]
    assert isinstance(history_before, RGBTemporalPositionHistory)
    valid_before = int(history_before.valid_mask.sum())

    model.diagnostics.reset()
    scheduler_state = model.scheduler.state_for("camera")
    scheduler_state.last_surprise = 0.0
    scheduler_state.association_failures = 0
    scheduler_state.steps_since_global = model.scheduler.global_every_steps
    model.ingest(_rgb_packet(2.0 / 30.0, shift=1))

    assert model.diagnostics.latest is not None
    assert model.diagnostics.latest.observation_mode == "GLOBAL_DISCOVERY"
    assert "camera" not in model.caches
    history_after = model.state.temporal_histories["camera"]
    assert isinstance(history_after, RGBTemporalPositionHistory)
    assert int(history_after.valid_mask.sum()) >= valid_before
    assert model.last_measurements is not None
    assert {"velocity", "velocity_from_position"}.isdisjoint(
        model.last_measurements.supported_state_fields
    )
    assert "world_velocity_valid_mask" in model.last_measurements.auxiliary
    assert "world_velocity_log_variance" in model.last_measurements.auxiliary

    model.detach_state()
    detached = model.state.temporal_histories["camera"]
    assert isinstance(detached, RGBTemporalPositionHistory)
    assert not detached.positions.requires_grad
    model.reset()
    assert not model.state.temporal_histories


def test_temporal_velocity_can_retain_position_innovation_coupling() -> None:
    torch.manual_seed(3)
    config = _small_rgb_config()
    config = replace(
        config,
        model=replace(
            config.model,
            rgb=replace(
                config.model.rgb,
                temporal_velocity_enabled=True,
                temporal_velocity_position_innovation_coupling=True,
            ),
        ),
    )
    model = OnlineWorldModel.from_config(config, device="cpu")
    model.ingest(_rgb_packet(0.0))
    model.ingest(_rgb_packet(1.0 / 30.0, shift=1))

    assert model.last_measurements is not None
    assert "velocity_from_position" in model.last_measurements.supported_state_fields
    assert "velocity" not in model.last_measurements.supported_state_fields
    assert "world_velocity_valid_mask" in model.last_measurements.auxiliary


def test_fast_roi_cache_is_invalidated_when_object_id_order_changes() -> None:
    torch.manual_seed(3)
    model = OnlineWorldModel.from_config(_small_rgb_config(), device="cpu")
    model.ingest(_rgb_packet(0.0))
    model.ingest(_rgb_packet(1.0 / 30.0, shift=1))
    cache = model.caches["camera"]
    assert model.belief is not None
    assert model._cache_matches_belief(cache, model.belief)

    cache.object_ids = cache.object_ids.roll(1, dims=1)
    cache.object_features.fill_(torch.nan)
    assert not model._cache_matches_belief(cache, model.belief)
    model.diagnostics.reset()
    scheduler_state = model.scheduler.state_for("camera")
    scheduler_state.last_surprise = 0.0
    scheduler_state.association_failures = 0
    scheduler_state.steps_since_global = 1
    model.ingest(_rgb_packet(2.0 / 30.0, shift=1))

    assert model.diagnostics.latest is not None
    assert model.diagnostics.latest.observation_mode == "FAST_ROI"
    fresh_cache = model.caches["camera"]
    assert fresh_cache is not cache
    assert model.belief is not None
    assert model._cache_matches_belief(fresh_cache, model.belief)
    assert torch.isfinite(fresh_cache.object_features).all()


def test_rgb_runtime_rejects_privileged_packet_when_oracle_disabled() -> None:
    model = OnlineWorldModel.from_config(_small_rgb_config(), device="cpu")
    privileged = ObservationPacket(
        modality="debug_oracle",
        sensor_id="state",
        timestamp=0.0,
        payload={"position": torch.zeros(1, 3)},
        calibration={},
        frame_id="world",
    )
    try:
        model.ingest(privileged)
    except (ValueError, KeyError) as error:
        assert "debug_oracle" in str(error)
    else:
        raise AssertionError("RGB-only model accepted privileged simulator state")


def test_runtime_threads_complete_dynamics_configuration() -> None:
    config = _small_rgb_config()
    model = OnlineWorldModel.from_config(config, device="cpu")
    resolved = model.dynamics.config
    requested = config.model.dynamics
    assert resolved.world_bounds == config.simulator.world_bounds
    assert resolved.process_noise_position == requested.process_noise_position
    assert resolved.process_noise_velocity == requested.process_noise_velocity
    assert resolved.penetration_slop == requested.penetration_slop
    assert resolved.max_penetration_correction == requested.max_penetration_correction
    assert resolved.sleep_speed == requested.sleep_speed
    assert (
        model.observation_modules["rgb"].projector.config.uncertainty_roi_scale
        == config.model.rgb.roi_uncertainty_scale
    )
    assert model.lifecycle.config.occluded_existence_delta == pytest.approx(
        -config.model.lifecycle.occlusion_existence_decay
    )


def test_synthetic_episode_runs_through_rgb_only_online_path() -> None:
    config = _small_rgb_config()
    episode = generate_episode(config, seed=41)
    model = OnlineWorldModel.from_config(config, device="cpu")
    for frame_index in range(3):
        packet = ObservationPacket(
            modality="rgb",
            sensor_id="camera",
            timestamp=float(episode["timestamps"][frame_index]),
            payload=episode["rgb"][frame_index],
            calibration={
                "intrinsics": episode["camera"]["intrinsics"][frame_index],
                "world_from_camera": episode["camera"]["world_from_camera"][frame_index],
            },
            frame_id="camera:camera",
        )
        belief = model.ingest(packet)
    assert belief.objects.active.any()
    assert not model.diagnostics.oracle_used
    assert {record.modality for record in model.diagnostics.records} == {"rgb"}
    assert torch.isfinite(belief.objects.position).all()
