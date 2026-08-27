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


def _active_two_object_belief(*, timestamp: float = 0.0):
    belief = BeliefFactory(
        max_objects=2,
        appearance_dim=3,
        initial_radius=0.21,
    ).create(
        batch_size=1,
        timestamp=timestamp,
        gravity=(0.0, 0.0, 0.0),
    )
    return belief.replace(
        objects=belief.objects.replace(
            active=torch.tensor([[True, True]]),
            object_id=torch.tensor([[7, 11]], dtype=torch.int64),
        )
    )


def _slot_measurement(
    position: torch.Tensor,
    timestamp: float,
    *,
    valid: tuple[bool, bool],
) -> MeasurementSet:
    if position.shape != (1, 2, 3):
        raise ValueError("two-slot test position must be [1,2,3]")
    mask = torch.tensor([valid], dtype=torch.bool, device=position.device)
    log_variance = position.new_full(position.shape, -9.0)
    values = torch.where(mask.unsqueeze(-1), position, torch.zeros_like(position))
    return MeasurementSet(
        modality="rgbd",
        sensor_id="camera0:rgbd",
        timestamp=position.new_tensor([timestamp]),
        values=values,
        log_variance=log_variance,
        existence_logits=torch.where(
            mask,
            position.new_full(mask.shape, 8.0),
            position.new_full(mask.shape, -8.0),
        ),
        measurement_mask=mask,
        appearance=None,
        class_logits=None,
        frame_id="camera:camera0",
        supported_state_fields=("position",),
        auxiliary={
            "world_position": values,
            "world_position_log_variance": log_variance,
        },
    )


def _slot_association(*, valid: tuple[bool, bool]) -> AssociationResult:
    pair_mask = torch.tensor([valid], dtype=torch.bool)
    return AssociationResult(
        belief_indices=torch.tensor([[0, 1]], dtype=torch.int64),
        measurement_indices=torch.tensor([[0, 1]], dtype=torch.int64),
        pair_mask=pair_mask,
        pair_cost=torch.zeros((1, 2)),
        unmatched_beliefs=~pair_mask,
        unmatched_measurements=torch.zeros_like(pair_mask),
        ambiguous=torch.zeros_like(pair_mask),
    )


@pytest.mark.parametrize(
    "field_name",
    ("temporal_history_size", "temporal_min_samples"),
)
def test_rgbd_temporal_sample_counts_require_real_integers(field_name: str) -> None:
    with pytest.raises(ValueError, match=f"{field_name} must be an integer"):
        RGBDObservationConfig(**{field_name: 16.0})


@pytest.mark.parametrize("value", (True, -1, 2, 1.0))
def test_rgbd_max_missing_rows_is_strictly_zero_or_one(value: object) -> None:
    with pytest.raises(ValueError, match="max_missing_rows"):
        RGBDObservationConfig(max_missing_rows=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", (False, 0, 1, 1.0))
def test_rgbd_temporal_recovery_requires_a_fresh_latest_row(value: object) -> None:
    with pytest.raises(ValueError, match="require_latest_valid"):
        RGBDObservationConfig(require_latest_valid=value)  # type: ignore[arg-type]


def test_rgbd_temporal_recovery_defaults_preserve_complete_sixteen_row_fit() -> None:
    config = RGBDObservationConfig()

    assert config.temporal_history_size == 16
    assert config.temporal_min_samples == 16
    assert config.max_missing_rows == 0
    assert config.require_latest_valid is True


def test_partial_visibility_and_missing_row_opt_in_require_two_proposals() -> None:
    for mutation in (
        {"bounded_partial_visibility": True},
        {"max_missing_rows": 1},
    ):
        with pytest.raises(ValueError, match="require proposal_count two"):
            RGBDObservationConfig(**mutation)

    config = RGBDObservationConfig(
        proposal_count=2,
        appearance_dim=3,
        bounded_partial_visibility=True,
        max_missing_rows=1,
    )
    assert config.bounded_partial_visibility
    assert config.max_missing_rows == 1


@pytest.mark.parametrize(
    ("field_name", "value", "match"),
    (
        ("bounded_partial_visibility", 1, "must be boolean"),
        ("minimum_observed_support_fraction", 1.1, "no greater than one"),
        ("maximum_surface_residual_relative_rms", 1.1, "no greater than one"),
        ("maximum_full_silhouette_overlap_fraction", 1.0, "smaller than one"),
    ),
)
def test_partial_visibility_controls_fail_closed(
    field_name: str,
    value: object,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        RGBDObservationConfig(**{field_name: value})  # type: ignore[arg-type]


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
    assert set(projected.auxiliary) == {"world_position"}
    torch.testing.assert_close(projected.auxiliary["world_position"], belief.objects.position)


def test_rgbd_projection_reports_causal_depth_ordered_partial_visibility() -> None:
    module = RGBDObservationModule(
        RGBDObservationConfig(
            proposal_count=2,
            appearance_dim=3,
            bounded_partial_visibility=True,
        )
    )
    intrinsics = torch.tensor(
        [[50.0, 0.0, 31.5], [0.0, 50.0, 31.5], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    ).unsqueeze(0)
    world_from_camera = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    belief = BeliefFactory(
        max_objects=2,
        appearance_dim=3,
        initial_radius=0.21,
    ).create(
        batch_size=1,
        gravity=(0.0, 0.0, 0.0),
        intrinsics=intrinsics,
        world_from_camera=world_from_camera,
    )
    position = torch.tensor(
        [[[0.0, 0.0, 2.0], [0.12, 0.0, 2.2]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    belief = belief.replace(
        objects=belief.objects.replace(
            active=torch.tensor([[True, True]]),
            object_id=torch.tensor([[7, 11]], dtype=torch.int64),
            position=position,
        )
    )
    projected = module.project(
        belief,
        SensorContext(
            sensor_id="camera0:rgbd",
            timestamp=0.0,
            calibration={
                "world_from_camera": world_from_camera,
                "intrinsics": intrinsics,
            },
            frame_id="camera:camera0",
            image_size=(64, 64),
        ),
    )

    visible = projected.auxiliary["visible_fraction"]
    torch.testing.assert_close(visible[0, 0], visible.new_tensor(1.0))
    assert 0.05 < float(visible[0, 1].detach()) < 1.0
    torch.testing.assert_close(
        projected.auxiliary["occlusion_fraction"],
        1.0 - visible,
    )
    assert projected.auxiliary["projectable_mask"].all()
    assert not projected.auxiliary["fully_occluded_mask"].any()
    assert not projected.auxiliary["unobservable_mask"].any()
    assert projected.auxiliary["pairwise_occlusion_fraction"].shape == (1, 2, 2)
    gradient = torch.autograd.grad(visible[0, 1], position)[0]
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0.0


def test_separated_two_object_projection_retains_the_legacy_auxiliary_contract() -> None:
    module = RGBDObservationModule(RGBDObservationConfig(proposal_count=2, appearance_dim=3))
    belief = _active_two_object_belief()

    projected = module.project(
        belief,
        SensorContext(
            sensor_id="camera0:rgbd",
            timestamp=0.0,
            calibration={},
            frame_id="camera:camera0",
            image_size=None,
        ),
    )

    assert set(projected.auxiliary) == {"world_position"}
    torch.testing.assert_close(projected.values, belief.objects.position)
    torch.testing.assert_close(projected.auxiliary["world_position"], belief.objects.position)


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


def test_opt_in_single_missing_row_recovers_per_slot_and_masks_ols_gradients() -> None:
    module = RGBDObservationModule(
        RGBDObservationConfig(
            proposal_count=2,
            appearance_dim=3,
            max_missing_rows=1,
            require_latest_valid=True,
            temporal_velocity_variance_floor=1.0e-12,
            temporal_velocity_variance_ceiling=1.0,
        )
    )
    belief = _active_two_object_belief()
    initial_position = torch.tensor(
        [[[0.1, -0.3, 2.0], [-0.4, 0.2, 2.4]]],
        dtype=torch.float32,
    )
    initial_velocity = torch.tensor(
        [[[0.2, 0.05, -0.1], [-0.08, 0.12, 0.04]]],
        dtype=torch.float32,
    )

    def raw_position(frame_index: int) -> torch.Tensor:
        timestamp = 0.05 * frame_index
        ideal, _ = free_motion_position_velocity(
            initial_position,
            initial_velocity,
            timestamp,
            gravity=belief.gravity,
            drag=belief.objects.drag,
        )
        phase = -1.0 if frame_index % 2 else 1.0
        jitter = (
            ideal.new_tensor(
                [
                    [[phase, -0.5 * phase, 0.25 * phase], [0.3 * phase, phase, -phase]],
                ]
            )
            * 1.0e-3
        )
        return (ideal.detach() + jitter).requires_grad_(True)

    history = None
    evidence = None
    for frame_index in range(16):
        timestamp = 0.05 * frame_index
        belief = belief.with_timestamp(timestamp)
        evidence, history = module.update_temporal_history(
            posterior=belief,
            measured=_slot_measurement(
                raw_position(frame_index),
                timestamp,
                valid=(True, True),
            ),
            association=_slot_association(valid=(True, True)),
            history=history,
        )
        if frame_index < 15:
            assert evidence is None

    assert evidence is not None and evidence.valid_mask.all()
    assert isinstance(history, RGBDTemporalPositionHistory)
    assert history.sample_mask.sum(dim=-1).tolist() == [[16, 16]]
    assert history.valid_mask.sum(dim=-1).tolist() == [[16, 16]]

    miss_timestamp = 0.8
    belief = belief.with_timestamp(miss_timestamp)
    missing_position = raw_position(16)
    evidence, history = module.update_temporal_history(
        posterior=belief,
        measured=_slot_measurement(
            missing_position,
            miss_timestamp,
            valid=(False, True),
        ),
        association=_slot_association(valid=(False, True)),
        history=history,
    )
    assert evidence is not None
    assert evidence.valid_mask.tolist() == [[False, True]]
    assert isinstance(history, RGBDTemporalPositionHistory)
    assert history.sample_mask.sum(dim=-1).tolist() == [[16, 16]]
    assert history.valid_mask.sum(dim=-1).tolist() == [[15, 16]]

    recovery_timestamp = 0.85
    belief = belief.with_timestamp(recovery_timestamp)
    recovery_position = raw_position(17)
    evidence, history = module.update_temporal_history(
        posterior=belief,
        measured=_slot_measurement(
            recovery_position,
            recovery_timestamp,
            valid=(True, True),
        ),
        association=_slot_association(valid=(True, True)),
        history=history,
    )
    assert evidence is not None and evidence.valid_mask.all()
    assert isinstance(history, RGBDTemporalPositionHistory)
    assert history.sample_mask.sum(dim=-1).tolist() == [[16, 16]]
    assert history.valid_mask.sum(dim=-1).tolist() == [[15, 16]]

    fit, fit_valid = history.fit(
        gravity=belief.gravity,
        drag=belief.objects.drag,
        minimum_support=16,
        minimum_dt=module.config.temporal_min_dt,
        conditioning_limit=module.config.fit_conditioning_limit,
        max_missing_rows=1,
        require_latest_valid=True,
    )
    assert fit.support_count.tolist() == [[15, 16]]
    assert fit_valid.all()
    inverse_normal = torch.linalg.inv(fit.normal_matrix)
    sample_count = fit.support_count.to(fit.normal_matrix.dtype)
    expected_variance = (
        fit.residual_covariance.diagonal(dim1=-2, dim2=-1)
        * (sample_count / (sample_count - 2.0)).unsqueeze(-1)
        * (inverse_normal[..., 1, 1] / sample_count).unsqueeze(-1)
    ).clamp(
        module.config.temporal_velocity_variance_floor,
        module.config.temporal_velocity_variance_ceiling,
    )
    torch.testing.assert_close(evidence.log_variance.exp(), expected_variance)

    missing_gradient, recovery_gradient = torch.autograd.grad(
        evidence.velocity[0, 0].sum(),
        (missing_position, recovery_position),
        allow_unused=True,
        materialize_grads=True,
    )
    torch.testing.assert_close(missing_gradient, torch.zeros_like(missing_gradient))
    assert torch.isfinite(recovery_gradient).all()
    assert recovery_gradient[0, 0].abs().sum() > 0.0
    torch.testing.assert_close(
        recovery_gradient[0, 1],
        torch.zeros_like(recovery_gradient[0, 1]),
    )

    second_miss_timestamp = 0.9
    belief = belief.with_timestamp(second_miss_timestamp)
    evidence, history = module.update_temporal_history(
        posterior=belief,
        measured=_slot_measurement(
            raw_position(18),
            second_miss_timestamp,
            valid=(False, True),
        ),
        association=_slot_association(valid=(False, True)),
        history=history,
    )
    assert evidence is not None
    assert evidence.valid_mask.tolist() == [[False, True]]

    second_recovery_timestamp = 0.95
    belief = belief.with_timestamp(second_recovery_timestamp)
    evidence, history = module.update_temporal_history(
        posterior=belief,
        measured=_slot_measurement(
            raw_position(19),
            second_recovery_timestamp,
            valid=(True, True),
        ),
        association=_slot_association(valid=(True, True)),
        history=history,
    )
    assert evidence is not None
    assert evidence.valid_mask.tolist() == [[False, True]]
    assert isinstance(history, RGBDTemporalPositionHistory)
    assert history.sample_mask.sum(dim=-1).tolist() == [[16, 16]]
    assert history.valid_mask.sum(dim=-1).tolist() == [[14, 16]]


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
