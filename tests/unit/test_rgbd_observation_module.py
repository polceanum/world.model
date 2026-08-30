from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from world_model.belief import BeliefFactory, BirthAssignments
from world_model.dynamics import free_motion_position_velocity
from world_model.fusion import AssociationResult
from world_model.observations import MeasurementSet, ObservationPacket, SensorContext
from world_model.observations.rgbd import (
    RGBDObservationConfig,
    RGBDObservationModule,
    RGBDTemporalPositionHistory,
)
from world_model.simulator import CameraFrame, SphereState, make_intrinsics, render_spheres


def _packet(
    *,
    batch: int = 1,
    timestamp: float = 0.0,
    missing_depth: bool = False,
    requires_grad: bool = False,
) -> ObservationPacket:
    rgb = torch.zeros((batch, 3, 32, 32), dtype=torch.float32)
    rgb[:, 0, 8:20, 12:24] = 0.9
    rgb[:, 1, 8:20, 12:24] = 0.35
    depth = torch.full((batch, 1, 32, 32), 2.0, dtype=torch.float32)
    if missing_depth:
        depth.zero_()
    if requires_grad:
        rgb.requires_grad_()
        depth.requires_grad_()
    intrinsics = (
        torch.tensor(
            [[48.0, 0.0, 15.5], [0.0, 48.0, 15.5], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        .expand(batch, -1, -1)
        .clone()
    )
    world_from_camera = torch.eye(4, dtype=torch.float32).expand(batch, -1, -1).clone()
    return ObservationPacket(
        modality="rgbd",
        sensor_id="camera0:rgbd",
        timestamp=timestamp,
        payload={"rgb": rgb, "depth": depth},
        calibration={
            "world_from_camera": world_from_camera,
            "intrinsics": intrinsics,
        },
        frame_id="camera:camera0",
        metadata={"image_size": (32, 32)},
    )


def _two_radius_packet(*, requires_grad: bool = False) -> tuple[ObservationPacket, SphereState]:
    image_size = (64, 80)
    identity = torch.eye(4, dtype=torch.float32)
    camera = CameraFrame(
        timestamp=0.0,
        world_from_camera=identity,
        camera_from_world=identity,
        intrinsics=make_intrinsics(image_size, 50.0),
        position=torch.zeros(3),
        target=torch.tensor([0.0, 0.0, 1.0]),
    )
    state = SphereState(
        object_id=torch.arange(2, dtype=torch.int64),
        active=torch.ones(2, dtype=torch.bool),
        position=torch.tensor([[-0.72, -0.12, 4.0], [0.78, 0.16, 4.35]]),
        velocity=torch.zeros((2, 3)),
        radius=torch.tensor([[0.24], [0.18]]),
        mass=torch.ones((2, 1)),
        restitution=torch.zeros((2, 1)),
        drag=torch.zeros((2, 1)),
        friction=torch.zeros((2, 1)),
        albedo=torch.tensor([[0.90, 0.20, 0.18], [0.18, 0.82, 0.90]]),
        orientation=torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(2, -1).clone(),
        angular_velocity=torch.zeros((2, 3)),
        sleeping=torch.zeros(2, dtype=torch.bool),
        sleep_counter=torch.zeros(2, dtype=torch.int64),
    )
    rendered = render_spheres(state, camera, image_size)
    rgb = rendered.rgb.unsqueeze(0).clone().requires_grad_(requires_grad)
    depth = rendered.depth_buffer[None, None].clone().requires_grad_(requires_grad)
    intrinsics = camera.intrinsics.unsqueeze(0).clone().requires_grad_(requires_grad)
    packet = ObservationPacket(
        modality="rgbd",
        sensor_id="camera0:rgbd",
        timestamp=0.0,
        payload={"rgb": rgb, "depth": depth},
        calibration={
            "world_from_camera": camera.world_from_camera.unsqueeze(0),
            "intrinsics": intrinsics,
        },
        frame_id="camera:camera0",
        metadata={"image_size": image_size},
    )
    return packet, state


def _active_belief(*, timestamp: float = 0.0, object_id: int = 7):
    belief = BeliefFactory(max_objects=1).create(
        batch_size=1,
        timestamp=timestamp,
        gravity=(0.0, 0.0, 0.0),
    )
    return belief.replace(
        objects=belief.objects.replace(
            active=torch.tensor([[True]]),
            object_id=torch.tensor([[object_id]], dtype=torch.int64),
        )
    )


def _measurement(position: torch.Tensor, timestamp: float, *, valid: bool = True) -> MeasurementSet:
    if position.shape != (1, 1, 3):
        raise ValueError("test position must be [1,1,3]")
    mask = torch.tensor([[valid]], dtype=torch.bool, device=position.device)
    log_variance = position.new_full(position.shape, -9.0)
    return MeasurementSet(
        modality="rgbd",
        sensor_id="camera0:rgbd",
        timestamp=position.new_tensor([timestamp]),
        values=torch.where(mask.unsqueeze(-1), position, torch.zeros_like(position)),
        log_variance=log_variance,
        existence_logits=position.new_tensor([[8.0 if valid else -8.0]]),
        measurement_mask=mask,
        appearance=None,
        class_logits=None,
        frame_id="camera:camera0",
        supported_state_fields=("position",),
        auxiliary={
            "world_position": torch.where(
                mask.unsqueeze(-1),
                position,
                torch.zeros_like(position),
            ),
            "world_position_log_variance": log_variance,
        },
    )


def _association(*, matched: bool = True) -> AssociationResult:
    pair_mask = torch.tensor([[matched]], dtype=torch.bool)
    return AssociationResult(
        belief_indices=torch.tensor([[0]], dtype=torch.int64),
        measurement_indices=torch.tensor([[0]], dtype=torch.int64),
        pair_mask=pair_mask,
        pair_cost=torch.tensor([[0.0]]),
        unmatched_beliefs=torch.tensor([[not matched]], dtype=torch.bool),
        unmatched_measurements=torch.tensor([[not matched]], dtype=torch.bool),
        ambiguous=torch.tensor([[False]], dtype=torch.bool),
    )


@pytest.mark.parametrize(
    "field_name",
    ("temporal_history_size", "temporal_min_samples"),
)
def test_rgbd_temporal_sample_counts_require_real_integers(field_name: str) -> None:
    with pytest.raises(ValueError, match=f"{field_name} must be an integer"):
        RGBDObservationConfig(**{field_name: 16.0})


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("unbatched_rgb", r"\[B,3,H,W\]"),
        ("unbatched_calibration", r"\[B,4,4\]"),
        ("missing_image_size", "image_size"),
        ("extra_payload", "exactly 'rgb' and 'depth'"),
        ("depth_dtype", "share dtype and device"),
    ],
)
def test_rgbd_packet_contract_is_strictly_batched_and_composite(
    mutation: str,
    match: str,
) -> None:
    module = RGBDObservationModule()
    packet = _packet()
    payload = dict(packet.payload)
    calibration = dict(packet.calibration)
    metadata = dict(packet.metadata)
    if mutation == "unbatched_rgb":
        payload["rgb"] = payload["rgb"][0]
    elif mutation == "unbatched_calibration":
        calibration["world_from_camera"] = calibration["world_from_camera"][0]
    elif mutation == "missing_image_size":
        metadata.clear()
    elif mutation == "extra_payload":
        payload["object_id"] = torch.zeros((1,), dtype=torch.int64)
    elif mutation == "depth_dtype":
        payload["depth"] = payload["depth"].to(torch.float64)
    else:  # pragma: no cover - parameter table owns this branch
        raise AssertionError(mutation)
    invalid = replace(
        packet,
        payload=payload,
        calibration=calibration,
        metadata=metadata,
    )
    with pytest.raises((TypeError, ValueError), match=match):
        module.validate_packet(invalid)


def test_rgbd_measurement_is_raw_world_position_and_preserves_rgb_depth_gradients() -> None:
    module = RGBDObservationModule()
    packet = _packet(batch=2, requires_grad=True)
    measured = module.initialise_measurements([packet], context=object())

    assert measured.values.shape == (2, 1, 3)
    assert measured.measurement_mask.all()
    assert measured.supported_state_fields == ("position",)
    assert "world_velocity" not in measured.auxiliary
    torch.testing.assert_close(measured.values, measured.auxiliary["world_position"])
    assert measured.auxiliary["world_position_independent_axis_mask"].all()

    rgb = packet.payload["rgb"]
    depth = packet.payload["depth"]
    rgb_gradient, depth_gradient = torch.autograd.grad(
        measured.values.square().sum(),
        (rgb, depth),
    )
    assert torch.isfinite(rgb_gradient).all()
    assert torch.isfinite(depth_gradient).all()
    assert rgb_gradient.abs().sum() > 0.0
    assert depth_gradient.abs().sum() > 0.0


def test_rgbd_metric_radius_estimation_publishes_typed_geometry_evidence() -> None:
    module = RGBDObservationModule(
        RGBDObservationConfig(
            proposal_count=2,
            appearance_dim=3,
            metric_radius_estimation_enabled=True,
            minimum_world_radius=0.10,
            maximum_world_radius=0.35,
            measurement_radius_variance=1.0e-6,
        )
    )
    packet, state = _two_radius_packet(requires_grad=True)

    measured = module.initialise_measurements([packet], context=object())
    direct_error = (measured.values[0] - state.position).square().sum()
    swapped_error = (measured.values[0] - state.position.flip(0)).square().sum()
    expected_radius = state.radius[:, 0]
    if swapped_error < direct_error:
        expected_radius = expected_radius.flip(0)

    assert measured.supported_state_fields == ("position", "radius")
    assert measured.auxiliary["world_radius_valid_mask"].all()
    torch.testing.assert_close(
        measured.auxiliary["world_radius"][0, :, 0],
        expected_radius,
        atol=2.0e-5,
        rtol=0.0,
    )
    torch.testing.assert_close(
        measured.auxiliary["world_radius_log_variance"],
        torch.full((1, 2, 1), torch.tensor(1.0e-6).log()),
    )
    rgb = packet.payload["rgb"]
    depth = packet.payload["depth"]
    intrinsics = packet.calibration["intrinsics"]
    gradients = torch.autograd.grad(
        measured.auxiliary["world_radius"].sum(),
        (rgb, depth, intrinsics),
    )
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert float(gradient.abs().sum()) > 0.0


def test_metric_radius_mode_emits_a_valid_empty_measurement_on_missing_depth() -> None:
    module = RGBDObservationModule(
        RGBDObservationConfig(
            proposal_count=2,
            appearance_dim=3,
            metric_radius_estimation_enabled=True,
            minimum_world_radius=0.10,
            maximum_world_radius=0.35,
        )
    )
    packet, _ = _two_radius_packet()
    packet = replace(
        packet,
        payload={
            "rgb": packet.payload["rgb"],
            "depth": torch.zeros_like(packet.payload["depth"]),
        },
    )

    measured = module.initialise_measurements([packet], context=object())

    measured.validate()
    assert measured.supported_state_fields == ("position", "radius")
    assert not measured.measurement_mask.any()
    assert not measured.auxiliary["world_radius_valid_mask"].any()
    assert not measured.auxiliary["world_radius"].any()


@pytest.mark.parametrize(
    "missing_key",
    ("world_radius", "world_radius_log_variance", "world_radius_valid_mask"),
)
def test_metric_radius_evidence_group_is_atomic(missing_key: str) -> None:
    measured = _measurement(torch.ones((1, 1, 3)), 0.0)
    auxiliary = {
        **measured.auxiliary,
        "world_radius": torch.full((1, 1, 1), 0.2),
        "world_radius_log_variance": torch.full((1, 1, 1), -10.0),
        "world_radius_valid_mask": torch.tensor([[True]]),
    }
    auxiliary.pop(missing_key)
    measured = replace(
        measured,
        supported_state_fields=("position", "radius"),
        auxiliary=auxiliary,
    )

    with pytest.raises(ValueError, match="radius evidence requires"):
        measured.validate()


def test_complete_metric_radius_group_requires_declared_support() -> None:
    measured = _measurement(torch.ones((1, 1, 3)), 0.0)
    measured = replace(
        measured,
        auxiliary={
            **measured.auxiliary,
            "world_radius": torch.full((1, 1, 1), 0.2),
            "world_radius_log_variance": torch.full((1, 1, 1), -10.0),
            "world_radius_valid_mask": torch.tensor([[True]]),
        },
    )

    with pytest.raises(ValueError, match="declare radius support"):
        measured.validate()


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("integer_radius", "floating dtype"),
        ("integer_variance", "floating dtype"),
        ("valid_without_measurement", "subset of measurement_mask"),
    ],
)
def test_metric_radius_group_rejects_nonfloating_and_orphan_validity(
    mutation: str,
    match: str,
) -> None:
    measured = _measurement(torch.ones((1, 1, 3)), 0.0)
    radius: torch.Tensor = torch.full((1, 1, 1), 0.2)
    radius_log_variance: torch.Tensor = torch.full((1, 1, 1), -10.0)
    radius_valid = torch.tensor([[True]])
    measurement_mask = measured.measurement_mask
    if mutation == "integer_radius":
        radius = torch.ones((1, 1, 1), dtype=torch.int64)
    elif mutation == "integer_variance":
        radius_log_variance = torch.full((1, 1, 1), -10, dtype=torch.int64)
    else:
        measurement_mask = torch.tensor([[False]])
    measured = replace(
        measured,
        measurement_mask=measurement_mask,
        supported_state_fields=("position", "radius"),
        auxiliary={
            **measured.auxiliary,
            "world_radius": radius,
            "world_radius_log_variance": radius_log_variance,
            "world_radius_valid_mask": radius_valid,
        },
    )

    with pytest.raises((TypeError, ValueError), match=match):
        measured.validate()


def test_fixed_radius_module_config_ignores_estimator_only_bounds() -> None:
    config = RGBDObservationConfig(
        world_radius=0.02,
        metric_radius_estimation_enabled=False,
        minimum_world_radius=0.30,
        maximum_world_radius=0.10,
    )

    assert config.world_radius == 0.02


def test_missing_depth_emits_no_valid_measurement_and_never_falls_back_to_rgb() -> None:
    module = RGBDObservationModule()
    packet = _packet(missing_depth=True, requires_grad=True)
    measured = module.initialise_measurements([packet], context=object())

    assert not measured.measurement_mask.any()
    assert torch.equal(measured.values, torch.zeros_like(measured.values))
    assert torch.equal(
        measured.auxiliary["metric_surface_depth"],
        torch.zeros_like(measured.auxiliary["metric_surface_depth"]),
    )
    assert torch.all(measured.existence_logits < 0.0)
    rgb = packet.payload["rgb"]
    depth = packet.payload["depth"]
    rgb_gradient, depth_gradient = torch.autograd.grad(
        measured.values.sum(),
        (rgb, depth),
    )
    assert not rgb_gradient.any()
    assert not depth_gradient.any()


def test_positive_depth_without_rgb_foreground_emits_no_measurement() -> None:
    module = RGBDObservationModule()
    packet = _packet(requires_grad=True)
    rgb = packet.payload["rgb"]
    depth = packet.payload["depth"]
    no_foreground = replace(
        packet,
        payload={"rgb": torch.zeros_like(rgb, requires_grad=True), "depth": depth},
    )

    measured = module.initialise_measurements([no_foreground], context=object())

    assert not measured.measurement_mask.any()
    assert torch.equal(measured.values, torch.zeros_like(measured.values))
    assert torch.all(measured.existence_logits < 0.0)
    assert torch.equal(
        measured.auxiliary["metric_confidence"],
        torch.zeros_like(measured.auxiliary["metric_confidence"]),
    )
    no_foreground_rgb = no_foreground.payload["rgb"]
    rgb_gradient, depth_gradient = torch.autograd.grad(
        measured.values.sum(),
        (no_foreground_rgb, depth),
    )
    assert torch.isfinite(rgb_gradient).all() and not rgb_gradient.any()
    assert torch.isfinite(depth_gradient).all() and not depth_gradient.any()


def test_rgbd_projection_uses_world_position_in_one_persistent_slot() -> None:
    module = RGBDObservationModule()
    belief = _active_belief()
    belief = belief.replace(
        objects=belief.objects.replace(position=torch.tensor([[[0.3, -0.2, 2.1]]]))
    )
    projected = module.project(
        belief,
        SensorContext(
            sensor_id="camera0:rgbd",
            timestamp=0.0,
            calibration={},
            frame_id="camera:camera0",
            image_size=(32, 32),
        ),
    )
    torch.testing.assert_close(projected.values, belief.objects.position)
    assert projected.valid_mask.all()
    assert projected.object_ids.item() == 7


def test_birth_assignment_seeds_frame_zero_and_uniform_fit_emits_velocity_only() -> None:
    module = RGBDObservationModule(
        RGBDObservationConfig(
            temporal_velocity_variance_floor=1.0e-10,
            temporal_velocity_variance_ceiling=1.0,
        )
    )
    belief = _active_belief(timestamp=0.0)
    empty = RGBDTemporalPositionHistory.empty(
        object_ids=torch.tensor([[-1]], dtype=torch.int64),
        active_mask=torch.tensor([[False]]),
        history_size=16,
        dtype=torch.float32,
    )
    initial_position = torch.tensor([[[0.1, -0.3, 2.0]]], requires_grad=True)
    initial_velocity = torch.tensor([[[0.2, 0.05, -0.1]]])
    drag = torch.tensor([[[0.05]]])
    gravity = torch.zeros(1, 3)
    raw_positions: list[torch.Tensor] = []

    def position_at(timestamp: float) -> torch.Tensor:
        position, _ = free_motion_position_velocity(
            initial_position,
            initial_velocity,
            timestamp,
            gravity=gravity,
            drag=drag,
        )
        raw = position.clone().requires_grad_(True)
        raw_positions.append(raw)
        return raw

    first = _measurement(position_at(0.0), 0.0)
    history = module.update_temporal_history_after_births(
        posterior=belief,
        measured=first,
        birth_assignments=BirthAssignments(
            batch_indices=torch.tensor([0], dtype=torch.int64),
            measurement_indices=torch.tensor([0], dtype=torch.int64),
            belief_indices=torch.tensor([0], dtype=torch.int64),
            object_ids=torch.tensor([7], dtype=torch.int64),
        ),
        history=empty,
    )
    assert isinstance(history, RGBDTemporalPositionHistory)
    assert history.sample_mask.sum().item() == 1
    evidence = None
    for frame_index in range(1, 16):
        timestamp = 0.05 * frame_index
        belief = belief.with_timestamp(timestamp)
        evidence, history = module.update_temporal_history(
            posterior=belief,
            measured=_measurement(position_at(timestamp), timestamp),
            association=_association(),
            history=history,
        )
        if frame_index < 15:
            assert evidence is None

    assert isinstance(history, RGBDTemporalPositionHistory)
    assert evidence is not None
    assert evidence.valid_mask.all()
    assert evidence.position is None
    assert evidence.position_log_variance is None
    assert evidence.position_valid_mask is None
    _, expected_velocity = free_motion_position_velocity(
        initial_position,
        initial_velocity,
        0.75,
        gravity=gravity,
        drag=drag,
    )
    torch.testing.assert_close(evidence.velocity, expected_velocity, rtol=2.0e-5, atol=2.0e-6)
    gradients = torch.autograd.grad(evidence.velocity.sum(), raw_positions)
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(gradient.abs().sum() > 0.0 for gradient in gradients)


def test_one_invalid_associated_row_fails_complete_uniform_window_closed() -> None:
    module = RGBDObservationModule()
    belief = _active_belief()
    history = None
    for frame_index in range(16):
        timestamp = 0.05 * frame_index
        belief = belief.with_timestamp(timestamp)
        valid = frame_index != 8
        evidence, history = module.update_temporal_history(
            posterior=belief,
            measured=_measurement(
                torch.tensor([[[timestamp, 0.0, 2.0]]]),
                timestamp,
                valid=valid,
            ),
            association=_association(matched=valid),
            history=history,
        )
    assert evidence is None
    assert isinstance(history, RGBDTemporalPositionHistory)
    assert history.sample_mask.all()
    assert not history.valid_mask.all()


def test_persistent_id_replacement_drops_previous_temporal_evidence() -> None:
    module = RGBDObservationModule()
    belief = _active_belief()
    history = None
    for frame_index in range(15):
        timestamp = 0.05 * frame_index
        belief = belief.with_timestamp(timestamp)
        _, history = module.update_temporal_history(
            posterior=belief,
            measured=_measurement(
                torch.tensor([[[timestamp, 0.0, 2.0]]]),
                timestamp,
            ),
            association=_association(),
            history=history,
        )
    belief = _active_belief(timestamp=0.75, object_id=19)
    evidence, history = module.update_temporal_history(
        posterior=belief,
        measured=_measurement(torch.tensor([[[0.75, 0.0, 2.0]]]), 0.75),
        association=_association(),
        history=history,
    )
    assert evidence is None
    assert isinstance(history, RGBDTemporalPositionHistory)
    assert history.object_ids.item() == 19
    assert history.sample_mask.sum().item() == 1
