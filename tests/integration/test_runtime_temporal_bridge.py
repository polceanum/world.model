from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from world_model.belief import BeliefFactory, MotionMode, ObjectLifecycle, WorldBelief
from world_model.dynamics import AnalyticFreeMotionDynamics
from world_model.observations import DirectVelocityEvidence, MeasurementSet, ObservationPacket
from world_model.observations.rgbd import RGBDTemporalPositionHistory
from world_model.runtime import OnlineWorldModel
from world_model.runtime.state import runtime_stream_key
from world_model.utils.config import load_config

CONFIG_DIR = Path(__file__).parents[2] / "configs"


def _birth_measurements(world_position: torch.Tensor) -> MeasurementSet:
    batch, proposals, _ = world_position.shape
    probabilities = world_position.new_tensor([[0.6, 0.9]]).expand(batch, -1)
    if proposals != probabilities.shape[1]:
        probabilities = world_position.new_full((batch, proposals), 0.9)
    return MeasurementSet(
        modality="rgbd",
        sensor_id="camera0:rgbd",
        timestamp=world_position.new_zeros(batch),
        values=world_position,
        log_variance=world_position.new_zeros(batch, proposals, 3),
        existence_logits=torch.logit(probabilities),
        measurement_mask=torch.ones(
            batch,
            proposals,
            device=world_position.device,
            dtype=torch.bool,
        ),
        appearance=None,
        class_logits=None,
        frame_id="camera:camera0",
        supported_state_fields=("position",),
        auxiliary={"world_position": world_position},
    )


def test_explicit_birth_assignment_preserves_selected_measurement_gradient() -> None:
    belief = BeliefFactory(max_objects=1).create()
    world_position = torch.tensor(
        [[[0.1, 0.0, 0.0], [0.9, 0.2, -0.1]]],
        requires_grad=True,
    )
    measurements = _birth_measurements(world_position)

    born, assignments = ObjectLifecycle().birth_from_measurements_with_assignments(
        belief,
        measurements,
        torch.tensor([[True, True]]),
    )

    torch.testing.assert_close(assignments.batch_indices, torch.tensor([0]))
    torch.testing.assert_close(assignments.measurement_indices, torch.tensor([1]))
    torch.testing.assert_close(assignments.belief_indices, torch.tensor([0]))
    torch.testing.assert_close(assignments.object_ids, torch.tensor([0]))
    torch.testing.assert_close(born.objects.position[0, 0], world_position[0, 1])
    born.objects.position.sum().backward()
    assert world_position.grad is not None
    torch.testing.assert_close(world_position.grad[0, 0], torch.zeros(3))
    torch.testing.assert_close(world_position.grad[0, 1], torch.ones(3))


def _active_free_motion_belief() -> tuple[
    WorldBelief,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    belief = BeliefFactory(max_objects=1).create(gravity=(0.0, -0.3, 0.0))
    position = torch.tensor([[[0.2, 1.1, -0.4]]], requires_grad=True)
    velocity = torch.tensor([[[0.7, -0.1, 0.25]]], requires_grad=True)
    log_drag = torch.tensor([[[math.log(0.2)]]], requires_grad=True)
    active = torch.ones_like(belief.objects.active)
    object_id = torch.zeros_like(belief.objects.object_id)
    motion_mode_logits = belief.objects.motion_mode_logits.new_full(
        belief.objects.motion_mode_logits.shape,
        -4.0,
    )
    motion_mode_logits[..., MotionMode.CREATED] = 4.0
    objects = belief.objects.replace(
        position=position,
        velocity=velocity,
        log_drag=log_drag,
        active=active,
        object_id=object_id,
        motion_mode_logits=motion_mode_logits,
    )
    return belief.replace(objects=objects), position, velocity, log_drag


def test_analytic_free_motion_is_parameterless_differentiable_and_composable() -> None:
    belief, position, velocity, log_drag = _active_free_motion_belief()
    dynamics = AnalyticFreeMotionDynamics()

    assert not tuple(dynamics.parameters())
    assert dynamics.state_dict() == {}
    direct = dynamics.predict(belief, torch.tensor([0.4]))
    composed = dynamics.predict(
        dynamics.predict(belief, torch.tensor([0.15])),
        torch.tensor([0.25]),
    )
    trajectory = dynamics.rollout(belief, [0.1, 0.4])

    torch.testing.assert_close(composed.objects.position, direct.objects.position)
    torch.testing.assert_close(composed.objects.velocity, direct.objects.velocity)
    torch.testing.assert_close(trajectory.positions[:, -1], direct.objects.position)
    torch.testing.assert_close(trajectory.velocities[:, -1], direct.objects.velocity)
    assert torch.equal(
        trajectory.event_logits.argmax(dim=-1),
        trajectory.event_logits.new_full(
            trajectory.event_logits.shape[:-1],
            MotionMode.FREE,
            dtype=torch.long,
        ),
    )
    assert torch.equal(
        direct.objects.motion_mode_logits.argmax(dim=-1),
        direct.objects.motion_mode_logits.new_full(
            direct.objects.motion_mode_logits.shape[:-1],
            MotionMode.FREE,
            dtype=torch.long,
        ),
    )
    collision_probability = trajectory.event_logits.softmax(dim=-1)[..., MotionMode.COLLISION]
    assert float(collision_probability.max()) < 0.001
    (trajectory.positions[:, -1].square().sum() + trajectory.velocities[:, -1].sum()).backward()
    for source in (position, velocity, log_drag):
        assert source.grad is not None
        assert torch.isfinite(source.grad).all()
        assert source.grad.abs().sum() > 0


def test_runtime_stream_keys_isolate_new_modalities_without_changing_legacy_keys() -> None:
    assert runtime_stream_key("rgb", "camera") == "camera"
    assert runtime_stream_key("debug_oracle", "state") == "state"
    assert runtime_stream_key("rgbd", "camera") == "rgbd:camera"
    assert runtime_stream_key("rgbd", "camera") != runtime_stream_key("rgb", "camera")


def _rgbd_packet(
    timestamp: float,
    *,
    requires_grad: bool = False,
) -> tuple[ObservationPacket, torch.Tensor, torch.Tensor]:
    rgb = torch.zeros((1, 3, 32, 32), dtype=torch.float32)
    rgb[:, 0, 8:20, 12:24] = 0.9
    rgb[:, 1, 8:20, 12:24] = 0.35
    depth = torch.full((1, 1, 32, 32), 2.0, dtype=torch.float32)
    if requires_grad:
        rgb.requires_grad_()
        depth.requires_grad_()
    intrinsics = torch.tensor(
        [[[48.0, 0.0, 15.5], [0.0, 48.0, 15.5], [0.0, 0.0, 1.0]]],
        dtype=torch.float32,
    )
    world_from_camera = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    return (
        ObservationPacket(
            modality="rgbd",
            sensor_id="camera0:rgbd",
            timestamp=timestamp,
            payload={"rgb": rgb, "depth": depth},
            calibration={
                "world_from_camera": world_from_camera,
                "intrinsics": intrinsics,
            },
            frame_id="camera:camera0:rgbd",
            metadata={"image_size": (32, 32)},
        ),
        rgb,
        depth,
    )


def test_public_rgbd_runtime_seeds_birth_history_and_keeps_a_sensor_gradient_path() -> None:
    config = load_config(CONFIG_DIR / "rgbd_online_free_motion_cpu.yaml")
    config = replace(
        config,
        model=replace(
            config.model,
            rgb=replace(config.model.rgb, global_every_steps=7),
            rgbd=replace(config.model.rgbd, global_every_steps=3),
        ),
    )
    model = OnlineWorldModel.from_config(config, device="cpu")
    assert model.belief_factory.initial_radius == pytest.approx(config.model.rgbd.world_radius)
    assert model.belief_factory.initial_drag == pytest.approx(config.model.rgbd.linear_drag)
    assert not tuple(model.parameters())
    assert model.state_dict() == {}
    packet, rgb, depth = _rgbd_packet(0.0, requires_grad=True)

    posterior = model.ingest(packet)

    assert isinstance(model.dynamics, AnalyticFreeMotionDynamics)
    assert model.updater.learned_corrector is None
    assert model.updater.config.direct_metric_position_update
    assert model.scheduler.global_every_steps == config.model.rgbd.global_every_steps == 3
    assert posterior.objects.active.tolist() == [[True]]
    assert torch.equal(posterior.objects.velocity, torch.zeros_like(posterior.objects.velocity))
    assert model.last_measurements is not None
    assert model.last_measurements.supported_state_fields == ("position",)
    stream_key = "rgbd:camera0:rgbd"
    history = model.state.temporal_histories[stream_key]
    assert isinstance(history, RGBDTemporalPositionHistory)
    assert history.object_ids.tolist() == [[0]]
    assert history.sample_mask.sum().item() == 1
    assert history.valid_mask.sum().item() == 1
    assert stream_key in model.scheduler._sensor_state
    assert "camera0:rgbd" not in model.scheduler._sensor_state
    assert model.diagnostics.latest is not None
    assert model.diagnostics.latest.sensor_id == stream_key

    rgb_gradient, depth_gradient = torch.autograd.grad(
        posterior.objects.position.sum(),
        (rgb, depth),
    )
    assert torch.isfinite(rgb_gradient).all() and rgb_gradient.abs().sum() > 0
    assert torch.isfinite(depth_gradient).all() and depth_gradient.abs().sum() > 0


def test_malformed_unbatched_rgbd_packet_leaves_runtime_state_untouched() -> None:
    config = load_config(CONFIG_DIR / "rgbd_online_free_motion_cpu.yaml")
    model = OnlineWorldModel.from_config(config, device="cpu")
    packet, _, _ = _rgbd_packet(0.0)
    malformed = replace(
        packet,
        payload={
            "rgb": packet.payload["rgb"][0],
            "depth": packet.payload["depth"][0],
        },
    )

    with pytest.raises(ValueError, match=r"\[B,3,H,W\]"):
        model.ingest(malformed)

    assert model.belief is None
    assert not model.state.caches
    assert not model.state.temporal_histories
    assert not model.scheduler._sensor_state
    assert not model.diagnostics.records


def test_low_precision_rgbd_packet_is_rejected_before_runtime_cast_or_mutation() -> None:
    config = load_config(CONFIG_DIR / "rgbd_online_free_motion_cpu.yaml")
    model = OnlineWorldModel.from_config(config, device="cpu")
    packet, _, _ = _rgbd_packet(0.0)
    low_precision = replace(
        packet,
        payload={
            "rgb": packet.payload["rgb"].half(),
            "depth": packet.payload["depth"].half(),
        },
        calibration={name: value.half() for name, value in packet.calibration.items()},
    )

    with pytest.raises(TypeError, match="supports only float32 and float64"):
        model.ingest(low_precision)

    assert model.belief is None
    assert not model.state.temporal_histories
    assert not model.scheduler._sensor_state
    assert not model.diagnostics.records


def test_duplicate_rgbd_stream_at_one_timestamp_is_transactionally_rejected() -> None:
    config = load_config(CONFIG_DIR / "rgbd_online_free_motion_cpu.yaml")
    model = OnlineWorldModel.from_config(config, device="cpu")
    packet, _, _ = _rgbd_packet(0.0)

    with pytest.raises(ValueError, match="duplicate observation streams"):
        model.ingest([packet, packet])

    assert model.belief is None
    assert model.state.ingest_count == 0
    assert not model.state.temporal_histories
    assert not model.scheduler._sensor_state
    assert not model.diagnostics.records


def test_unknown_modality_is_rejected_before_runtime_initialization() -> None:
    config = load_config(CONFIG_DIR / "rgbd_online_free_motion_cpu.yaml")
    model = OnlineWorldModel.from_config(config, device="cpu")
    packet, _, _ = _rgbd_packet(0.0)
    unsupported = replace(packet, modality="bogus")

    with pytest.raises(KeyError, match="unsupported modality 'bogus'"):
        model.ingest(unsupported)

    assert model.belief is None
    assert model.state.batch_size == 1
    assert model.state.ingest_count == 0
    assert not model.state.caches
    assert not model.state.temporal_histories
    assert not model.state.tentative_births
    assert not model.scheduler._sensor_state
    assert not model.diagnostics.records
    assert model.last_measurements is None
    assert model.last_direct_velocity_evidence is None


def test_rgbd_runtime_consumes_velocity_only_after_complete_raw_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(CONFIG_DIR / "rgbd_online_free_motion_cpu.yaml")
    model = OnlineWorldModel.from_config(config, device="cpu")
    corrections: list[tuple[DirectVelocityEvidence, torch.Tensor, torch.Tensor]] = []
    original = model.updater.correct_direct_velocity

    def recording_correction(
        prior: WorldBelief,
        evidence: DirectVelocityEvidence,
    ) -> WorldBelief:
        before_position = prior.objects.position.clone()
        corrected = original(prior, evidence)
        corrections.append(
            (
                evidence,
                before_position,
                corrected.objects.position.clone(),
            )
        )
        return corrected

    monkeypatch.setattr(model.updater, "correct_direct_velocity", recording_correction)
    for frame_index in range(16):
        packet, _, _ = _rgbd_packet(frame_index * 0.05)
        model.ingest(packet)

    assert len(corrections) == 1
    evidence, position_before, position_after = corrections[0]
    assert evidence.position is None
    assert evidence.position_log_variance is None
    assert evidence.position_valid_mask is None
    assert evidence.valid_mask.all()
    assert torch.equal(position_after, position_before)
    assert model.last_direct_velocity_evidence is not None
    assert not model.last_direct_velocity_evidence.velocity.requires_grad
    assert model.last_direct_velocity_evidence.position is None
    history = model.state.temporal_histories["rgbd:camera0:rgbd"]
    assert isinstance(history, RGBDTemporalPositionHistory)
    assert history.sample_mask.all()
    assert history.valid_mask.all()


def test_stale_rgbd_history_packet_rejects_before_prepared_consumption_or_diagnostics_reset() -> (
    None
):
    config = load_config(CONFIG_DIR / "rgbd_online_free_motion_cpu.yaml")
    model = OnlineWorldModel.from_config(config, device="cpu")
    for frame_index in range(16):
        packet, _, _ = _rgbd_packet(frame_index * 0.05)
        model.ingest(packet)

    belief_before = model.belief
    history_before = model.state.temporal_histories["rgbd:camera0:rgbd"]
    evidence_before = model.last_direct_velocity_evidence
    measurements_before = model.last_measurements
    diagnostics_before = tuple(model.diagnostics.records)
    scheduler_before = dict(model.scheduler._sensor_state)
    ingest_count_before = model.state.ingest_count
    assert belief_before is not None
    assert evidence_before is not None

    stale_timestamp = 0.7505
    prepared = model.prepare_propagation(stale_timestamp)
    stale_packet, _, _ = _rgbd_packet(stale_timestamp)
    with pytest.raises(
        ValueError,
        match="RGB-D temporal timestamps must increase by minimum_dt",
    ):
        model.ingest(stale_packet, prepared=prepared)

    assert not prepared.consumed
    assert model.belief is belief_before
    assert model.state.temporal_histories["rgbd:camera0:rgbd"] is history_before
    assert model.last_direct_velocity_evidence is evidence_before
    assert model.last_measurements is measurements_before
    assert tuple(model.diagnostics.records) == diagnostics_before
    assert model.scheduler._sensor_state == scheduler_before
    assert model.state.ingest_count == ingest_count_before
