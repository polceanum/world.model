from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from world_model.dynamics import OnlineLocalAccelerationDynamics
from world_model.observations import ObservationPacket
from world_model.observations.rgb import RGBTemporalPositionHistory
from world_model.runtime import OnlineWorldModel
from world_model.simulator import generate_episode
from world_model.training.loop import (
    _soft_association_surrogate_losses,
    _soft_posterior_straight_through_belief,
)
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
        structured_disc_center_enabled=True,
        structured_disc_depth_relative_std=0.05,
        structured_disc_position_confidence=0.995,
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


def test_runtime_constructs_online_acceleration_only_when_explicitly_enabled() -> None:
    base = _small_rgb_config()
    config = replace(
        base,
        model=replace(
            base.model,
            rgb=replace(base.model.rgb, temporal_velocity_enabled=True),
        ),
        runtime=replace(
            base.runtime,
            hypothesis_pool_enabled=True,
            hypothesis_local_applicability_enabled=True,
            hypothesis_online_acceleration_enabled=True,
            hypothesis_online_acceleration_minimum_support_count=3,
            hypothesis_online_acceleration_maximum_mps2=12.0,
        ),
    )
    config.validate()

    model = OnlineWorldModel.from_config(config, device="cpu")

    assert model.hypothesis_controller is not None
    candidates = model.hypothesis_controller.pool.dynamics_models
    assert len(candidates) == 5
    assert isinstance(candidates[-1], OnlineLocalAccelerationDynamics)
    assert candidates[-1].minimum_support_count == 3
    assert candidates[-1].maximum_acceleration == pytest.approx(12.0)


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


def test_differentiable_ingest_trace_preserves_hard_runtime_and_reaches_rgb_filter() -> None:
    torch.manual_seed(17)
    config = _small_rgb_config()
    traced = OnlineWorldModel.from_config(config, device="cpu")
    ordinary = OnlineWorldModel.from_config(config, device="cpu")
    ordinary.load_state_dict(traced.state_dict())
    traced.observation_modules["rgb"].requires_grad_(True)
    traced.updater.requires_grad_(True)

    first_traced, first_trace = traced.ingest_with_trace(_rgb_packet(0.0))
    first_ordinary = ordinary.ingest(_rgb_packet(0.0))
    assert first_trace is not None
    torch.testing.assert_close(
        first_traced.objects.position,
        first_ordinary.objects.position,
        rtol=0.0,
        atol=0.0,
    )
    assert torch.equal(first_traced.objects.object_id, first_ordinary.objects.object_id)

    posterior, _ = traced.ingest_with_trace(_rgb_packet(1.0 / 30.0, shift=1))
    ordinary_posterior = ordinary.ingest(_rgb_packet(1.0 / 30.0, shift=1))
    posterior, trace = traced.ingest_with_trace(_rgb_packet(2.0 / 30.0, shift=2))
    ordinary_posterior = ordinary.ingest(_rgb_packet(2.0 / 30.0, shift=2))
    assert trace is not None
    for traced_value, ordinary_value in (
        (posterior.objects.position, ordinary_posterior.objects.position),
        (posterior.objects.velocity, ordinary_posterior.objects.velocity),
        (posterior.objects.fast_log_variance, ordinary_posterior.objects.fast_log_variance),
    ):
        torch.testing.assert_close(traced_value, ordinary_value, rtol=0.0, atol=0.0)
    assert traced.last_measurements is not None
    assert traced.last_measurements.auxiliary["world_position"].grad_fn is None
    assert trace.measurements.auxiliary["world_position"].grad_fn is not None

    active = trace.predicted_belief.objects.active
    target_position = trace.predicted_belief.objects.position.detach() + 0.15
    target_velocity = trace.predicted_belief.objects.velocity.detach() + 0.05
    losses, metrics = _soft_association_surrogate_losses(
        traced,
        trace,
        aligned_target_position=target_position,
        aligned_target_velocity=target_velocity,
        matched_belief_slots=active,
        temperature=0.5,
    )
    assert metrics["soft_association_supported_coordinate_count"] > 0
    sum(losses.values()).backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in traced.observation_modules["rgb"].parameters()
    )
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in traced.updater.parameters()
    )
    assert not hasattr(traced.state, "ingest_trace")


def test_soft_posterior_carrier_is_forward_exact_and_rollout_differentiable() -> None:
    torch.manual_seed(23)
    base = _small_rgb_config()
    config = replace(
        base,
        model=replace(
            base.model,
            rgb=replace(base.model.rgb, global_every_steps=1),
        ),
    )
    model = OnlineWorldModel.from_config(config, device="cpu")
    model.train()
    model.observation_modules["rgb"].requires_grad_(True)
    model.ingest(_rgb_packet(0.0))
    model.ingest(_rgb_packet(1.0 / 30.0, shift=1))
    _, trace = model.ingest_with_trace(_rgb_packet(2.0 / 30.0, shift=2))
    assert trace is not None
    world_position = trace.measurements.auxiliary["world_position"]
    world_position.retain_grad()

    carried, metrics = _soft_posterior_straight_through_belief(
        model,
        trace,
        temperature=0.5,
    )
    assert metrics["soft_posterior_position_coordinate_count"] > 0
    for carried_value, hard_value in (
        (carried.objects.position, trace.posterior.objects.position),
        (carried.objects.velocity, trace.posterior.objects.velocity),
        (carried.objects.fast_log_variance, trace.posterior.objects.fast_log_variance),
    ):
        assert torch.equal(carried_value, hard_value)

    model.state.belief = carried
    trajectory = model.dynamics.rollout(carried, [0.1], return_events=False)
    trajectory.positions.square().sum().backward()
    assert world_position.grad is not None
    assert torch.isfinite(world_position.grad).all()
    assert bool((world_position.grad != 0).any())


def test_opt_in_runtime_pool_uses_rgb_measurements_without_oracle_state() -> None:
    """The normal runtime can accumulate delayed RGB evidence before predicting."""

    torch.manual_seed(3)
    config = _small_rgb_config()
    config = replace(
        config,
        runtime=replace(
            config.runtime,
            hypothesis_pool_enabled=True,
            hypothesis_evidence_horizons_seconds=(1.0 / 30.0,),
            hypothesis_axis_independent_axes=(0,),
        ),
    )
    model = OnlineWorldModel.from_config(config, device="cpu")
    model.ingest(_rgb_packet(0.0))
    model.ingest(_rgb_packet(1.0 / 30.0, shift=1))
    model.ingest(_rgb_packet(2.0 / 30.0, shift=1))
    model.ingest(_rgb_packet(3.0 / 30.0, shift=1))

    controller = model.hypothesis_controller
    assert controller is not None
    assert controller.pool.last_selection is not None
    assert not model.diagnostics.oracle_used
    future = model.predict([0.1])
    assert "hypothesis_axis_index" in future.auxiliary
    assert torch.isfinite(future.positions).all()


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
    assert "velocity_from_position" in model.last_measurements.supported_state_fields
    assert "velocity" not in model.last_measurements.supported_state_fields
    # Two samples cannot satisfy the default three-sample temporal observer.
    # Ordinary position innovation must nevertheless retain its conservative
    # velocity correction path.
    assert not model.last_measurements.auxiliary["world_velocity_valid_mask"].any()
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
    assert "velocity_from_position" in model.last_measurements.supported_state_fields
    assert "velocity" not in model.last_measurements.supported_state_fields
    assert "world_velocity_valid_mask" in model.last_measurements.auxiliary
    assert "world_velocity_axis_valid_mask" in model.last_measurements.auxiliary
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
    assert model.updater.uncertainty.config.missed_fast_variance_increment == pytest.approx(
        config.model.filter.missed_variance_growth
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


def test_combined_camera_fast_depth_supplies_strict_raw_velocity_support() -> None:
    """The promoted raw observer must receive evidence at its real ROI cadence."""

    config = _small_rgb_config()
    config = replace(
        config,
        simulator=replace(
            config.simulator,
            frame_rate=20.0,
            physics_rate=120.0,
            camera_motion="combined",
        ),
        model=replace(
            config.model,
            rgb=replace(
                config.model.rgb,
                global_every_steps=3,
                temporal_velocity_enabled=True,
                temporal_velocity_history_size=3,
                temporal_velocity_min_samples=3,
                temporal_velocity_max_age_steps=None,
                temporal_velocity_lateral_only=False,
                temporal_velocity_independent_raw_history_enabled=True,
                temporal_velocity_continuous_gravity_axis_enabled=True,
                structured_disc_fast_depth_enabled=True,
            ),
        ),
    )
    config.validate()
    episode = generate_episode(config, seed=48)
    model = OnlineWorldModel.from_config(config, device="cpu")
    fast_measurement_count = 0
    supported_fast_velocity_count = 0
    for frame_index in range(8):
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
        model.ingest(packet)
        assert model.last_measurements is not None
        if model.diagnostics.latest is not None and (
            model.diagnostics.latest.observation_mode == "FAST_ROI"
        ):
            fast_measurement_count += 1
            supported_fast_velocity_count += int(
                model.last_measurements.auxiliary["world_velocity_axis_valid_mask"].any()
            )

    assert fast_measurement_count > 0
    assert supported_fast_velocity_count > 0
    assert not model.diagnostics.oracle_used
