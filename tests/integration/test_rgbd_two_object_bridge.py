"""Seed-free public-runtime checks for the visible two-object RGB-D rung."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from world_model.dynamics import free_motion_position_velocity
from world_model.observations import ObservationPacket
from world_model.observations.rgbd import RGBDTemporalPositionHistory
from world_model.observations.rgbd import module as rgbd_module
from world_model.runtime import OnlineWorldModel
from world_model.simulator import CameraFrame, SphereState, make_intrinsics, render_spheres
from world_model.utils.config import load_config

CONFIG_DIR = Path(__file__).parents[2] / "configs"
IMAGE_SIZE = (64, 64)
RADIUS_M = 0.21
DRAG = 0.05
# The accepted one-object bridge used 1e-14.  The two-object rung requires a
# one-million-times stronger floor while leaving margin for the weakest exact
# protocol projection after palette-stability bounding.
MINIMUM_PER_FRAME_SENSOR_VJP_L1 = 1.0e-8


def _config():
    base = load_config(CONFIG_DIR / "rgbd_online_free_motion_cpu.yaml")
    return replace(
        base,
        model=replace(
            base.model,
            max_objects=2,
            state=replace(base.model.state, appearance_dim=3),
            rgbd=replace(base.model.rgbd, proposal_count=2),
            association=replace(base.model.association, appearance_weight=0.25),
        ),
        simulator=replace(base.simulator, min_objects=2, max_objects=2),
    )


def _camera() -> CameraFrame:
    identity = torch.eye(4, dtype=torch.float32)
    return CameraFrame(
        timestamp=0.0,
        world_from_camera=identity,
        camera_from_world=identity,
        intrinsics=make_intrinsics(IMAGE_SIZE, 50.0),
        position=torch.zeros(3),
        target=torch.tensor([0.0, 0.0, 1.0]),
    )


def _state(position: torch.Tensor, velocity: torch.Tensor) -> SphereState:
    return SphereState(
        object_id=torch.arange(2, dtype=torch.int64),
        active=torch.ones(2, dtype=torch.bool),
        position=position,
        velocity=velocity,
        radius=torch.full((2, 1), RADIUS_M),
        mass=torch.ones((2, 1)),
        restitution=torch.zeros((2, 1)),
        drag=torch.full((2, 1), DRAG),
        friction=torch.zeros((2, 1)),
        albedo=torch.tensor(
            [[0.90, 0.20, 0.18], [0.18, 0.82, 0.90]],
            dtype=torch.float32,
        ),
        orientation=torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(2, -1).clone(),
        angular_velocity=torch.zeros((2, 3)),
        sleeping=torch.zeros(2, dtype=torch.bool),
        sleep_counter=torch.zeros(2, dtype=torch.int64),
    )


def _packet(
    position: torch.Tensor,
    velocity: torch.Tensor,
    timestamp: float,
    *,
    requires_grad: bool = False,
    drop_depth_slot: int | None = None,
) -> tuple[ObservationPacket, torch.Tensor, torch.Tensor]:
    camera = _camera()
    rendered = render_spheres(_state(position, velocity), camera, IMAGE_SIZE)
    rgb = rendered.rgb.unsqueeze(0)
    depth = rendered.depth_buffer[None, None]
    if drop_depth_slot is not None:
        if drop_depth_slot not in {0, 1}:
            raise ValueError("drop_depth_slot must be physical slot zero or one")
        depth = depth.clone()
        depth[0, 0, rendered.instance_slot_map == drop_depth_slot] = 0.0
    if requires_grad:
        rgb.requires_grad_()
        depth.requires_grad_()
    return (
        ObservationPacket(
            modality="rgbd",
            sensor_id="camera0:rgbd",
            timestamp=timestamp,
            payload={"rgb": rgb, "depth": depth},
            calibration={
                "world_from_camera": camera.world_from_camera.unsqueeze(0),
                "intrinsics": camera.intrinsics.unsqueeze(0),
            },
            frame_id="camera:camera0:rgbd",
            metadata={"image_size": IMAGE_SIZE},
        ),
        rgb,
        depth,
    )


def _initial_state() -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([[-0.62, -0.10, 4.0], [0.66, 0.14, 4.25]]),
        torch.tensor([[0.035, 0.0, 0.0], [-0.035, 0.0, 0.0]]),
    )


def test_two_object_runtime_births_two_metric_tracks_with_sensor_gradients() -> None:
    config = _config()
    config.validate()
    model = OnlineWorldModel.from_config(config, device="cpu")
    position, velocity = _initial_state()
    packet, rgb, depth = _packet(position, velocity, 0.0, requires_grad=True)

    posterior = model.ingest(packet)

    assert posterior.objects.active.tolist() == [[True, True]]
    assert posterior.objects.object_id.tolist() == [[0, 1]]
    assert model.last_measurements is not None
    assert model.last_measurements.values.shape == (1, 2, 3)
    assert model.last_measurements.appearance is not None
    assert model.last_measurements.appearance.shape == (1, 2, 3)
    for partial_only_diagnostic in (
        "surface_fit_residual_relative_rms",
        "observed_support_fraction",
        "full_silhouette_radius_pixels",
        "full_silhouette_gap_pixels",
        "full_boundary_clearance_pixels",
        "full_silhouette_overlap_fraction",
    ):
        assert partial_only_diagnostic not in model.last_measurements.auxiliary
    history = model.state.temporal_histories["rgbd:camera0:rgbd"]
    assert isinstance(history, RGBDTemporalPositionHistory)
    assert history.object_ids.tolist() == [[0, 1]]
    assert history.sample_mask.sum().item() == 2
    assert history.valid_mask.sum().item() == 2
    rgb_gradient, depth_gradient = torch.autograd.grad(
        posterior.objects.position.square().sum(),
        (rgb, depth),
    )
    for gradient in (rgb_gradient, depth_gradient):
        assert torch.isfinite(gradient).all()
        assert float(gradient.abs().sum()) > 0.0


def test_two_object_history_keeps_ids_separate_and_emits_two_velocities() -> None:
    config = _config()
    config.validate()
    model = OnlineWorldModel.from_config(config, device="cpu")
    initial_position, initial_velocity = _initial_state()
    gravity = torch.zeros((1, 3))
    drag = torch.full((1, 2), DRAG)

    for frame_index in range(16):
        timestamp = frame_index * 0.05
        position, velocity = free_motion_position_velocity(
            initial_position.unsqueeze(0),
            initial_velocity.unsqueeze(0),
            timestamp,
            gravity=gravity,
            drag=drag,
        )
        packet, _, _ = _packet(position[0], velocity[0], timestamp)
        posterior = model.ingest(packet)

    history = model.state.temporal_histories["rgbd:camera0:rgbd"]
    assert isinstance(history, RGBDTemporalPositionHistory)
    assert history.object_ids.tolist() == [[0, 1]]
    assert history.sample_mask.all()
    assert history.valid_mask.all()
    assert posterior.objects.object_id.tolist() == [[0, 1]]
    assert posterior.objects.active.all()
    assert model.last_direct_velocity_evidence is not None
    assert model.last_direct_velocity_evidence.valid_mask.tolist() == [[True, True]]
    assert model.last_direct_velocity_evidence.position is None
    expected_position, expected_velocity = free_motion_position_velocity(
        initial_position.unsqueeze(0),
        initial_velocity.unsqueeze(0),
        0.75,
        gravity=gravity,
        drag=drag,
    )
    # Match the unordered initial birth mapping before checking identity-aware
    # values.  Persistent IDs may legitimately attach to either spectral sign.
    direct = torch.linalg.vector_norm(
        posterior.objects.position[0] - expected_position[0],
        dim=-1,
    ).sum()
    swapped = torch.linalg.vector_norm(
        posterior.objects.position[0] - expected_position[0].flip(0),
        dim=-1,
    ).sum()
    expected_order = expected_position if direct <= swapped else expected_position.flip(1)
    expected_velocity_order = expected_velocity if direct <= swapped else expected_velocity.flip(1)
    torch.testing.assert_close(
        posterior.objects.position,
        expected_order,
        atol=0.01,
        rtol=0.0,
    )
    torch.testing.assert_close(
        posterior.objects.velocity,
        expected_velocity_order,
        atol=0.01,
        rtol=0.0,
    )


def test_two_object_velocity_and_rollout_reach_every_rgbd_history_frame() -> None:
    config = _config()
    config.validate()
    model = OnlineWorldModel.from_config(config, device="cpu")
    initial_position, initial_velocity = _initial_state()
    gravity = torch.zeros((1, 3))
    drag = torch.full((1, 2), DRAG)
    rgb_frames: list[torch.Tensor] = []
    depth_frames: list[torch.Tensor] = []

    for frame_index in range(16):
        timestamp = frame_index * 0.05
        position, velocity = free_motion_position_velocity(
            initial_position.unsqueeze(0),
            initial_velocity.unsqueeze(0),
            timestamp,
            gravity=gravity,
            drag=drag,
        )
        packet, rgb, depth = _packet(
            position[0],
            velocity[0],
            timestamp,
            requires_grad=True,
        )
        model.ingest(packet)
        rgb_frames.append(rgb)
        depth_frames.append(depth)

    assert model.belief is not None
    rollout = model.predict([0.1, 0.25, 0.5, 1.0, 2.0])
    sources = tuple(rgb_frames + depth_frames)
    coefficients = model.belief.objects.position.new_tensor((0.5, -0.75, 1.25))

    def projected(value: torch.Tensor) -> torch.Tensor:
        return (value * coefficients).mean()

    for object_index in range(2):
        current_position = projected(model.belief.objects.position[0, object_index])
        current_position_gradients = torch.autograd.grad(
            current_position,
            sources,
            retain_graph=True,
        )
        assert sum(float(value.abs().sum()) for value in current_position_gradients[:16]) > 0.0
        assert sum(float(value.abs().sum()) for value in current_position_gradients[16:]) > 0.0

        history_targets = [projected(model.belief.objects.velocity[0, object_index])]
        for horizon_index in range(5):
            history_targets.extend(
                (
                    projected(rollout.positions[0, horizon_index, object_index]),
                    projected(rollout.velocities[0, horizon_index, object_index]),
                )
            )
        for target in history_targets:
            gradients = torch.autograd.grad(target, sources, retain_graph=True)
            for gradient in gradients:
                assert torch.isfinite(gradient).all()
                assert float(gradient.abs().sum()) >= MINIMUM_PER_FRAME_SENSOR_VJP_L1


def test_alternating_proposal_order_does_not_switch_persistent_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = rgbd_module.two_disc_geometry_from_rgbd
    call_count = 0

    def alternating(*args: object, **kwargs: object):
        nonlocal call_count
        result = original(*args, **kwargs)
        should_flip = call_count % 2 == 1
        call_count += 1
        if not should_flip:
            return result
        geometry = replace(
            result.geometry,
            centres=result.geometry.centres.flip(1),
            radius_pixels=result.geometry.radius_pixels.flip(1),
            confidence=result.geometry.confidence.flip(1),
            valid_mask=result.geometry.valid_mask.flip(1),
            mass=result.geometry.mass.flip(1),
            effective_masks=result.geometry.effective_masks.flip(1),
        )
        return replace(
            result,
            world_position=result.world_position.flip(1),
            camera_position=result.camera_position.flip(1),
            centres=result.centres.flip(1),
            radius_pixels=result.radius_pixels.flip(1),
            appearance=result.appearance.flip(1),
            surface_depth=result.surface_depth.flip(1),
            centre_depth=result.centre_depth.flip(1),
            confidence=result.confidence.flip(1),
            valid_mask=result.valid_mask.flip(1),
            surface_fit_condition_number=result.surface_fit_condition_number.flip(1),
            surface_fit_radius=result.surface_fit_radius.flip(1),
            surface_fit_radius_relative_error=(result.surface_fit_radius_relative_error.flip(1)),
            surface_fit_residual_relative_rms=(result.surface_fit_residual_relative_rms.flip(1)),
            observed_support_fraction=result.observed_support_fraction.flip(1),
            silhouette_gap_pixels=result.silhouette_gap_pixels,
            boundary_clearance_pixels=result.boundary_clearance_pixels,
            full_silhouette_radius_pixels=result.full_silhouette_radius_pixels.flip(1),
            full_silhouette_gap_pixels=result.full_silhouette_gap_pixels,
            full_boundary_clearance_pixels=result.full_boundary_clearance_pixels.flip(1),
            full_silhouette_overlap_fraction=result.full_silhouette_overlap_fraction,
            chromatic_world_position=result.chromatic_world_position.flip(1),
            slot_logits=result.slot_logits.flip(1),
            provisional_centres=result.provisional_centres.flip(1),
            geometry=geometry,
        )

    monkeypatch.setattr(rgbd_module, "two_disc_geometry_from_rgbd", alternating)
    config = _config()
    model = OnlineWorldModel.from_config(config, device="cpu")
    initial_position, initial_velocity = _initial_state()
    gravity = torch.zeros((1, 3))
    drag = torch.full((1, 2), DRAG)

    birth_mapping: torch.Tensor | None = None
    for frame_index in range(16):
        timestamp = frame_index * 0.05
        position, velocity = free_motion_position_velocity(
            initial_position.unsqueeze(0),
            initial_velocity.unsqueeze(0),
            timestamp,
            gravity=gravity,
            drag=drag,
        )
        packet, _, _ = _packet(position[0], velocity[0], timestamp)
        posterior = model.ingest(packet)
        distance = torch.cdist(posterior.objects.position[0], position[0])
        current_mapping = distance.argmin(dim=-1)
        assert current_mapping.unique().numel() == 2
        if birth_mapping is None:
            birth_mapping = current_mapping
        else:
            assert torch.equal(current_mapping, birth_mapping)
        mapped_truth = position[0, current_mapping]
        assert (
            float(
                torch.linalg.vector_norm(
                    posterior.objects.position[0] - mapped_truth,
                    dim=-1,
                ).max()
            )
            < 0.01
        )
        assert posterior.objects.object_id.tolist() == [[0, 1]]

    assert call_count == 16
    assert birth_mapping is not None
    history = model.state.temporal_histories["rgbd:camera0:rgbd"]
    assert isinstance(history, RGBDTemporalPositionHistory)
    assert history.object_ids.tolist() == [[0, 1]]
    assert history.valid_mask.all()


def test_partial_visibility_one_local_miss_recovers_without_resetting_history() -> None:
    config = load_config(CONFIG_DIR / "rgbd_partial_visibility_recovery_cpu.yaml")
    model = OnlineWorldModel.from_config(config, device="cpu")
    runtime_module = model.observation_modules["rgbd"]
    assert isinstance(runtime_module, rgbd_module.RGBDObservationModule)
    assert runtime_module.config.bounded_partial_visibility is True
    assert runtime_module.config.minimum_observed_support_fraction == pytest.approx(0.35)
    assert runtime_module.config.maximum_surface_residual_relative_rms == pytest.approx(0.05)
    assert runtime_module.config.maximum_full_silhouette_overlap_fraction == pytest.approx(0.60)
    assert runtime_module.config.max_missing_rows == 1
    assert runtime_module.config.require_latest_valid is True
    initial_position = torch.tensor(
        [[-0.16, 0.0, 4.0], [0.16, 0.0, 5.0]],
        dtype=torch.float32,
    )
    initial_velocity = torch.tensor(
        [[0.02, 0.0, 0.0], [0.02, 0.0, 0.0]],
        dtype=torch.float32,
    )
    gravity = torch.zeros((1, 3))
    drag = torch.full((1, 2), DRAG)

    for frame_index in range(16):
        timestamp = frame_index * 0.05
        position, velocity = free_motion_position_velocity(
            initial_position.unsqueeze(0),
            initial_velocity.unsqueeze(0),
            timestamp,
            gravity=gravity,
            drag=drag,
        )
        packet, _, _ = _packet(position[0], velocity[0], timestamp)
        posterior = model.ingest(packet)

    assert model.last_measurements is not None
    assert model.last_measurements.measurement_mask.all()
    for partial_diagnostic in (
        "surface_fit_residual_relative_rms",
        "observed_support_fraction",
        "full_silhouette_radius_pixels",
        "full_silhouette_gap_pixels",
        "full_boundary_clearance_pixels",
        "full_silhouette_overlap_fraction",
    ):
        assert partial_diagnostic in model.last_measurements.auxiliary
    object_ids = posterior.objects.object_id.clone()

    miss_timestamp = 0.8
    miss_position, miss_velocity = free_motion_position_velocity(
        initial_position.unsqueeze(0),
        initial_velocity.unsqueeze(0),
        miss_timestamp,
        gravity=gravity,
        drag=drag,
    )
    miss_packet, _, _ = _packet(
        miss_position[0],
        miss_velocity[0],
        miss_timestamp,
        drop_depth_slot=1,
    )
    missed = model.ingest(miss_packet)

    assert model.last_measurements is not None
    assert model.last_measurements.measurement_mask.sum().item() == 1
    assert model.last_direct_velocity_evidence is not None
    assert sorted(model.last_direct_velocity_evidence.valid_mask[0].tolist()) == [False, True]
    history = model.state.temporal_histories["rgbd:camera0:rgbd"]
    assert isinstance(history, RGBDTemporalPositionHistory)
    assert sorted(history.sample_mask.sum(dim=-1)[0].tolist()) == [16, 16]
    assert sorted(history.valid_mask.sum(dim=-1)[0].tolist()) == [15, 16]
    assert sorted(missed.objects.missed_steps[0].tolist()) == [0, 1]
    assert torch.equal(missed.objects.object_id, object_ids)
    assert missed.objects.active.all()

    recovery_timestamp = 0.85
    recovery_position, recovery_velocity = free_motion_position_velocity(
        initial_position.unsqueeze(0),
        initial_velocity.unsqueeze(0),
        recovery_timestamp,
        gravity=gravity,
        drag=drag,
    )
    recovery_packet, _, _ = _packet(
        recovery_position[0],
        recovery_velocity[0],
        recovery_timestamp,
    )
    recovered = model.ingest(recovery_packet)

    assert model.last_measurements is not None
    assert model.last_measurements.measurement_mask.all()
    assert model.last_direct_velocity_evidence is not None
    assert model.last_direct_velocity_evidence.valid_mask.all()
    history = model.state.temporal_histories["rgbd:camera0:rgbd"]
    assert isinstance(history, RGBDTemporalPositionHistory)
    assert sorted(history.sample_mask.sum(dim=-1)[0].tolist()) == [16, 16]
    assert sorted(history.valid_mask.sum(dim=-1)[0].tolist()) == [15, 16]
    assert recovered.objects.missed_steps.eq(0).all()
    assert torch.equal(recovered.objects.object_id, object_ids)
    assert recovered.objects.active.all()
