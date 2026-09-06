"""Focused invariants for the explicit variable-set RGB-D front end."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from world_model.belief import BeliefFactory
from world_model.fusion import Associator
from world_model.observations import ObservationPacket, SensorContext
from world_model.observations.rgb.soft_geometry import soft_disc_geometry_from_rgb
from world_model.observations.rgbd import (
    RGBDObservationConfig,
    RGBDObservationModule,
    RGBDSetProposer,
    metric_sphere_centres_from_surface_depth,
)
from world_model.training import validate_checkpoint_config
from world_model.training.dynamic_set_config import RGBDConfig, load_config
from world_model.training.dynamic_set_materializer import materialize_dynamic_set_episode
from world_model.training.dynamic_set_objectives import detached_mean_gaussian_nll
from world_model.training.dynamic_set_protocol import physical_manifest

CONFIG_DIR = Path(__file__).parents[2] / "configs"


def _set_config() -> RGBDObservationConfig:
    return RGBDObservationConfig(
        observation_mode="set",
        max_objects=6,
        proposal_count=8,
        appearance_dim=8,
        temporal_min_samples=3,
    )


def _packet(*, requires_grad: bool = False) -> ObservationPacket:
    rgb = torch.zeros((1, 3, 32, 32), dtype=torch.float32)
    depth = torch.zeros((1, 1, 32, 32), dtype=torch.float32)
    rgb[:, :, 5:11, 2:8] = torch.tensor([0.90, 0.18, 0.12]).view(1, 3, 1, 1)
    rgb[:, :, 21:27, 24:30] = torch.tensor([0.12, 0.72, 0.92]).view(1, 3, 1, 1)
    depth[:, :, 5:11, 2:8] = 2.0
    depth[:, :, 21:27, 24:30] = 2.4
    if requires_grad:
        rgb.requires_grad_()
        depth.requires_grad_()
    intrinsics = torch.tensor(
        [[[48.0, 0.0, 15.5], [0.0, 48.0, 15.5], [0.0, 0.0, 1.0]]],
        dtype=torch.float32,
    )
    return ObservationPacket(
        modality="rgbd",
        sensor_id="camera0:rgbd",
        timestamp=0.0,
        payload={"rgb": rgb, "depth": depth},
        calibration={
            "world_from_camera": torch.eye(4, dtype=torch.float32).unsqueeze(0),
            "intrinsics": intrinsics,
        },
        frame_id="camera:camera0:rgbd",
        metadata={"image_size": (32, 32)},
    )


def _proposer_inputs(packet: ObservationPacket) -> tuple[torch.Tensor, ...]:
    rgb = packet.payload["rgb"]
    depth = packet.payload["depth"]
    valid_depth = torch.isfinite(depth) & (depth > 0.0) & (depth <= 1.0e4)
    safe_depth = torch.where(valid_depth, depth, torch.ones_like(depth))
    log_depth = torch.where(valid_depth, safe_depth.log(), torch.zeros_like(depth))
    foreground = soft_disc_geometry_from_rgb(
        rgb,
        foreground_threshold=0.04,
        foreground_temperature=0.10,
        minimum_mass=4.0,
    ).foreground_probability
    height, width = rgb.shape[-2:]
    y_axis = torch.linspace(-1.0, 1.0, height, dtype=rgb.dtype)
    x_axis = torch.linspace(-1.0, 1.0, width, dtype=rgb.dtype)
    yy, xx = torch.meshgrid(y_axis, x_axis, indexing="ij")
    coordinates = torch.stack((xx, yy), dim=0).unsqueeze(0)
    return rgb, log_depth, valid_depth, foreground, coordinates


def test_set_mode_is_explicit_and_legacy_module_state_dict_stays_empty() -> None:
    legacy = RGBDObservationModule()
    assert not legacy.state_dict()
    assert legacy.config.set_log_variance_residual_limit == 4.0
    positional_observation_config = RGBDObservationConfig(2, 3)
    assert positional_observation_config.proposal_count == 2
    assert positional_observation_config.appearance_dim == 3
    positional_project_config = RGBDConfig(False, 1, 2)
    assert positional_project_config.proposal_count == 2

    try:
        RGBDObservationConfig(proposal_count=8, appearance_dim=8)
    except ValueError as error:
        assert "legacy mode" in str(error)
    else:  # pragma: no cover - the explicit-mode guard owns this branch
        raise AssertionError("eight proposals must require explicit set mode")
    with pytest.raises(ValueError, match="legacy.*set_log_variance"):
        RGBDObservationConfig(set_log_variance_residual_limit=20.0)
    for invalid_limit in (0.0, 33.0, math.inf):
        with pytest.raises(ValueError, match="log_variance_residual_limit"):
            RGBDSetProposer(log_variance_residual_limit=invalid_limit)

    module = RGBDObservationModule(_set_config())
    assert isinstance(module.set_proposer, RGBDSetProposer)
    assert 0 < module.set_proposer.parameter_count() <= 100_000
    assert module.set_proposer.feature_dim == 32
    assert module.set_proposer.memory_projection.kernel_size == (1, 1)
    assert len(module.set_proposer.cross_attention_blocks) == 2
    assert all(
        block.attention.num_heads == 4 for block in module.set_proposer.cross_attention_blocks
    )
    state = module.state_dict()
    assert state
    assert not any(name.endswith("anchor_points") for name in state)

    restored = RGBDObservationModule(_set_config())
    restored.load_state_dict(state, strict=True)
    assert state.keys() == restored.state_dict().keys()


def test_dynamic_set_variance_head_spans_under_target_and_over_coverage() -> None:
    config = replace(_set_config(), set_log_variance_residual_limit=20.0)
    module = RGBDObservationModule(config)
    assert module.set_proposer is not None
    head = module.set_proposer.log_variance_residual_head
    with torch.no_grad():
        head.weight.zero_()

    base_log_variance = math.log(config.measurement_position_variance)
    z_90 = 1.6448536269514722
    coordinate_errors = torch.arange(1, 11, dtype=torch.float64) * 1.0e-7
    target_threshold = 9.5e-7
    target_log_variance = 2.0 * math.log(target_threshold / z_90)
    target_scaled_residual = (
        target_log_variance - base_log_variance
    ) / config.set_log_variance_residual_limit
    target_raw_bias = math.atanh(target_scaled_residual)
    raw_biases = (-4.0, target_raw_bias, 0.0)
    coverages: list[float] = []
    for raw_bias in raw_biases:
        with torch.no_grad():
            head.bias.fill_(raw_bias)
        measured = module.initialise_measurements([_packet()], context=object())
        scalar_log_variance = measured.log_variance[measured.measurement_mask][0, 0]
        threshold = z_90 * float((0.5 * scalar_log_variance.detach()).exp())
        coverages.append(float((coordinate_errors <= threshold).to(torch.float64).mean()))

    assert coverages[0] < 0.85
    assert coverages[1] == 0.9
    assert coverages[2] > 0.95

    with torch.no_grad():
        head.bias.fill_(target_raw_bias)
    target_measurement = module.initialise_measurements([_packet()], context=object())
    target = target_measurement.values.detach() + 1.0e-6
    nll = detached_mean_gaussian_nll(
        target_measurement.values,
        target,
        target_measurement.log_variance,
        target_measurement.measurement_mask,
    )
    (gradient,) = torch.autograd.grad(nll, (head.bias,))
    assert torch.isfinite(nll)
    assert torch.isfinite(gradient).all()
    assert float(gradient.abs().sum()) > 0.0


def test_shared_anchor_zero_residual_is_permutation_equivariant() -> None:
    module = RGBDObservationModule(_set_config())
    assert module.set_proposer is not None
    packet = _packet()
    proposer_inputs = _proposer_inputs(packet)
    zero_proposal = module.set_proposer(*proposer_inputs)
    assert not torch.count_nonzero(zero_proposal.mask_residual)
    assert not torch.count_nonzero(zero_proposal.existence_residual)
    assert not torch.count_nonzero(zero_proposal.appearance_residual)
    assert not torch.count_nonzero(zero_proposal.log_variance_residual)
    zero_masks = zero_proposal.full_mask_probability
    torch.testing.assert_close(zero_masks.sum(dim=1), torch.ones_like(zero_masks[:, 0]))
    expected_background = (~proposer_inputs[2]).to(zero_masks.dtype)
    torch.testing.assert_close(zero_masks[:, :1], expected_background, rtol=0.0, atol=1.0e-7)

    # Exercise equivariance through learned residuals too, rather than letting
    # their required zero initialization make the head checks vacuous.
    with torch.no_grad():
        for parameter in (
            module.set_proposer.mask_residual_projection.weight,
            module.set_proposer.mask_residual_projection.bias,
            module.set_proposer.existence_residual_head.weight,
            module.set_proposer.existence_residual_head.bias,
            module.set_proposer.appearance_residual_head.weight,
            module.set_proposer.appearance_residual_head.bias,
            module.set_proposer.log_variance_residual_head.weight,
            module.set_proposer.log_variance_residual_head.bias,
        ):
            parameter.copy_(torch.linspace(-0.01, 0.01, parameter.numel()).reshape_as(parameter))
    reference_proposal = module.set_proposer(*proposer_inputs)
    assert torch.count_nonzero(reference_proposal.mask_residual)
    assert torch.count_nonzero(reference_proposal.existence_residual)
    assert torch.count_nonzero(reference_proposal.appearance_residual)
    assert torch.count_nonzero(reference_proposal.log_variance_residual)
    reference_measurement = module.initialise_measurements([packet], context=object())

    permutation = torch.tensor([5, 0, 7, 2, 1, 6, 3, 4], dtype=torch.int64)
    reference_anchors = module.set_proposer.anchor_points.detach().clone()
    with torch.no_grad():
        module.set_proposer.anchor_points.copy_(reference_anchors[permutation])
    permuted_proposal = module.set_proposer(*proposer_inputs)
    permuted_measurement = module.initialise_measurements([packet], context=object())

    for reference, permuted in (
        (reference_proposal.slot_mask_logits, permuted_proposal.slot_mask_logits),
        (reference_proposal.base_slot_mask_logits, permuted_proposal.base_slot_mask_logits),
        (reference_proposal.mask_residual, permuted_proposal.mask_residual),
        (reference_proposal.existence_residual, permuted_proposal.existence_residual),
        (reference_proposal.appearance_residual, permuted_proposal.appearance_residual),
        (reference_proposal.log_variance_residual, permuted_proposal.log_variance_residual),
        (reference_proposal.query_features, permuted_proposal.query_features),
        (reference_measurement.values, permuted_measurement.values),
        (reference_measurement.existence_logits, permuted_measurement.existence_logits),
        (reference_measurement.measurement_mask, permuted_measurement.measurement_mask),
        (reference_measurement.appearance, permuted_measurement.appearance),
    ):
        assert reference is not None and permuted is not None
        assert torch.equal(permuted, reference[:, permutation])
    assert torch.equal(
        permuted_proposal.full_mask_logits[:, :1],
        reference_proposal.full_mask_logits[:, :1],
    )
    assert torch.equal(
        permuted_proposal.full_mask_logits[:, 1:],
        reference_proposal.full_mask_logits[:, 1:][:, permutation],
    )
    assert torch.equal(
        permuted_proposal.full_mask_probability[:, 1:],
        reference_proposal.full_mask_probability[:, 1:][:, permutation],
    )
    assert torch.equal(
        permuted_proposal.base_full_mask_logits[:, 1:],
        reference_proposal.base_full_mask_logits[:, 1:][:, permutation],
    )
    assert torch.equal(permuted_proposal.anchor_points, reference_anchors[permutation])


def test_zero_residual_eval_shortcut_is_publicly_exact_and_grad_calls_stay_full() -> None:
    module = RGBDObservationModule(_set_config())
    assert module.set_proposer is not None
    proposer_inputs = _proposer_inputs(_packet())

    # Training mode plus no-grad is the complete learned-path oracle: the
    # attention blocks have no stochastic layers, while the required eval
    # guard deliberately prevents the zero-head shortcut.
    module.train()
    with torch.no_grad():
        full_proposal = module.set_proposer(*proposer_inputs)
        full_measurement = module.initialise_measurements([_packet()], context=object())
    assert bool(torch.count_nonzero(full_proposal.query_features))

    module.eval()
    with torch.no_grad():
        shortcut_proposal = module.set_proposer(*proposer_inputs)
        shortcut_measurement = module.initialise_measurements([_packet()], context=object())
    assert not bool(torch.count_nonzero(shortcut_proposal.query_features))
    for field_name in (
        "slot_mask_logits",
        "base_slot_mask_logits",
        "full_mask_logits",
        "full_mask_probability",
        "base_full_mask_logits",
        "background_mask_logits",
        "mask_residual",
        "existence_residual",
        "appearance_residual",
        "log_variance_residual",
        "anchor_points",
    ):
        assert torch.equal(
            getattr(shortcut_proposal, field_name),
            getattr(full_proposal, field_name),
        )
    for field_name in (
        "timestamp",
        "values",
        "log_variance",
        "existence_logits",
        "measurement_mask",
        "appearance",
    ):
        reference = getattr(full_measurement, field_name)
        candidate = getattr(shortcut_measurement, field_name)
        assert isinstance(reference, torch.Tensor)
        assert isinstance(candidate, torch.Tensor)
        assert torch.equal(candidate, reference)
    assert shortcut_measurement.auxiliary.keys() == full_measurement.auxiliary.keys()
    for name, reference in full_measurement.auxiliary.items():
        candidate = shortcut_measurement.auxiliary[name]
        assert torch.equal(candidate, reference), name

    # Eval mode must still execute the complete architecture whenever autograd
    # is enabled, preserving both sensor and owner-parameter gradients.
    differentiable_inputs = list(proposer_inputs)
    differentiable_inputs[1] = differentiable_inputs[1].detach().requires_grad_(True)
    grad_proposal = module.set_proposer(*differentiable_inputs)
    assert bool(torch.count_nonzero(grad_proposal.query_features))
    log_depth_gradient, output_head_gradient = torch.autograd.grad(
        grad_proposal.query_features.square().sum() + grad_proposal.mask_residual.sum(),
        (
            differentiable_inputs[1],
            module.set_proposer.mask_residual_projection.weight,
        ),
    )
    assert torch.isfinite(log_depth_gradient).all()
    assert float(log_depth_gradient.abs().sum()) > 0.0
    assert torch.isfinite(output_head_gradient).all()
    assert float(output_head_gradient.abs().sum()) > 0.0


def test_set_proposer_consumes_log_depth_validity_foreground_and_coordinates() -> None:
    proposer = RGBDSetProposer()
    image, log_depth, valid_depth, foreground, coordinates = _proposer_inputs(_packet())
    differentiable_log_depth = log_depth.detach().requires_grad_(True)
    output = proposer(
        image,
        differentiable_log_depth,
        valid_depth,
        foreground,
        coordinates,
    )
    gradient = torch.autograd.grad(output.query_features.square().sum(), differentiable_log_depth)[
        0
    ]
    assert torch.isfinite(gradient).all()
    assert float(gradient.abs().sum()) > 0.0

    invalid_output = proposer(
        image,
        torch.zeros_like(log_depth),
        torch.zeros_like(valid_depth),
        foreground,
        coordinates,
    )
    assert not torch.equal(output.query_features, invalid_output.query_features)
    with pytest.raises(ValueError, match="valid_depth must be boolean"):
        proposer(image, log_depth, valid_depth.float(), foreground, coordinates)


def test_attention_memory_is_compact_while_mask_memory_stays_full_resolution() -> None:
    proposer = RGBDSetProposer()
    proposer_inputs = _proposer_inputs(_packet())
    memory_lengths: list[int] = []
    hooks = [
        block.register_forward_pre_hook(
            lambda _module, arguments: memory_lengths.append(int(arguments[1].shape[1]))
        )
        for block in proposer.cross_attention_blocks
    ]
    try:
        output = proposer(*proposer_inputs)
    finally:
        for hook in hooks:
            hook.remove()

    # A 32x32 image retains all 1,024 mask-memory pixels while each attention
    # block reads the stride-four 8x8 summary.
    assert memory_lengths == [64, 64]
    assert output.mask_residual.shape == (1, 8, 32, 32)
    assert output.full_mask_logits.shape == (1, 9, 32, 32)


def test_set_metric_centres_are_geometry_derived_and_sensor_differentiable() -> None:
    module = RGBDObservationModule(_set_config())
    packet = _packet(requires_grad=True)
    measured = module.initialise_measurements([packet], context=object())

    assert measured.values.shape == (1, 8, 3)
    assert measured.appearance is not None
    assert measured.appearance.shape == (1, 8, 8)
    assert int(measured.measurement_mask.sum()) >= 2
    metric = metric_sphere_centres_from_surface_depth(
        measured.auxiliary["image_centres"],
        packet.payload["depth"],
        _set_config().world_radius,
        packet.calibration["world_from_camera"],
        packet.calibration["intrinsics"],
    )
    valid = measured.measurement_mask
    torch.testing.assert_close(measured.values[valid], metric.world_position[valid])

    rgb = packet.payload["rgb"]
    depth = packet.payload["depth"]
    sensor_gradients = torch.autograd.grad(
        measured.values.square().sum(),
        (rgb, depth),
        retain_graph=True,
    )
    for gradient in sensor_gradients:
        assert torch.isfinite(gradient).all()
        assert float(gradient.abs().sum()) > 0.0

    assert module.set_proposer is not None
    appearance_coefficients = measured.appearance.new_tensor(
        (0.5, -0.25, 0.75, -1.0, 1.25, 0.3, -0.6, 0.9)
    )
    objective = (
        measured.values.square().sum()
        + measured.existence_logits.sum()
        + (measured.appearance * appearance_coefficients).sum()
        + measured.log_variance.sum()
    )
    named_parameters = dict(module.set_proposer.named_parameters())
    parameter_names = (
        "mask_residual_projection.weight",
        "existence_residual_head.weight",
        "appearance_residual_head.weight",
        "log_variance_residual_head.weight",
    )
    parameter_gradients = torch.autograd.grad(
        objective,
        tuple(named_parameters[name] for name in parameter_names),
    )
    for gradient in parameter_gradients:
        assert torch.isfinite(gradient).all()
        assert float(gradient.abs().sum()) > 0.0
    full_logits = measured.auxiliary["set_full_mask_logits"]
    assert full_logits.shape == (1, 9, 32, 32)
    full_probability = full_logits.softmax(dim=1)
    assert full_probability.shape == (1, 9, 32, 32)
    assert measured.auxiliary["set_background_mask_logits"].shape == (1, 1, 32, 32)
    variance_residual = measured.auxiliary["set_log_variance_residual"]
    assert variance_residual.shape == (1, 8, 3)
    assert float(variance_residual.detach().abs().max()) <= 4.0


def test_six_beliefs_and_eight_measurements_use_rectangular_association() -> None:
    module = RGBDObservationModule(_set_config())
    measured = module.initialise_measurements([_packet()], context=object())
    valid_measurements = torch.nonzero(measured.measurement_mask[0], as_tuple=False).flatten()
    assert valid_measurements.numel() >= 2

    belief = BeliefFactory(max_objects=6, appearance_dim=8).create()
    active = torch.zeros_like(belief.objects.active)
    active[:, :2] = True
    object_id = torch.full_like(belief.objects.object_id, -1)
    object_id[:, :2] = torch.tensor([10, 11], dtype=torch.int64)
    position = torch.zeros_like(belief.objects.position)
    position[:, :2] = measured.values[:, valid_measurements[:2]].detach()
    appearance = torch.zeros_like(belief.objects.appearance)
    assert measured.appearance is not None
    appearance[:, :2] = measured.appearance[:, valid_measurements[:2]].detach()
    belief = belief.replace(
        objects=belief.objects.replace(
            active=active,
            object_id=object_id,
            position=position,
            appearance=appearance,
        )
    )
    predicted = module.project(
        belief,
        SensorContext(
            sensor_id="camera0:rgbd",
            timestamp=0.0,
            calibration={},
            frame_id="camera:camera0:rgbd",
            image_size=(32, 32),
        ),
    )
    associator = Associator(
        mahalanobis_gate=1.0e6,
        maximum_cost=1.0e6,
        minimum_measurement_confidence=0.0,
    )
    cost = associator.cost_matrix(measured, predicted)
    association = associator.match(belief, measured, predicted)

    assert predicted.values.shape == (1, 6, 3)
    assert cost.shape == (1, 6, 8)
    assert association.pair_mask.shape == (1, 6)
    assert association.unmatched_measurements.shape == (1, 8)
    assert int(association.pair_mask.sum()) == 2


@pytest.mark.parametrize(
    ("cell_index", "object_count"),
    ((0, 1), (2, 2), (6, 3), (10, 4), (14, 5), (18, 6)),
)
def test_zero_residual_component_baseline_is_metric_and_exact_count(
    cell_index: int,
    object_count: int,
) -> None:
    row = physical_manifest("development")[cell_index]
    materialization = materialize_dynamic_set_episode(row)
    frame = next(materialization.public_frames())
    packet = ObservationPacket(
        modality="rgbd",
        sensor_id="camera0:rgbd",
        timestamp=frame.timestamp,
        payload={"rgb": frame.rgb.unsqueeze(0), "depth": frame.depth.unsqueeze(0)},
        calibration={
            "world_from_camera": frame.world_from_camera.unsqueeze(0),
            "intrinsics": frame.intrinsics.unsqueeze(0),
        },
        frame_id="camera:camera0:rgbd",
        metadata={"image_size": tuple(frame.rgb.shape[-2:])},
    )
    module = RGBDObservationModule(replace(_set_config(), fit_conditioning_limit=10_000.0))
    measured = module.initialise_measurements([packet], context=object())
    accepted = measured.measurement_mask & (measured.existence_logits.sigmoid() >= 0.55)

    assert int(accepted.sum()) == object_count
    assert bool(measured.auxiliary["surface_fit_valid"][accepted].all())
    assert bool(measured.values[accepted].ne(0.0).any(dim=-1).all())
    truth_active = materialization.episode["objects"]["active"][0]
    truth_position = materialization.episode["objects"]["position"][0, truth_active]
    distance = torch.cdist(measured.values[accepted], truth_position)
    assert bool((distance.amin(dim=0) < 0.010).all())


def test_zero_residual_component_baseline_emits_no_object_before_birth() -> None:
    row = physical_manifest("development")[1]
    materialization = materialize_dynamic_set_episode(row)
    frame = next(materialization.public_frames())
    assert not bool(materialization.episode["objects"]["active"][0].any())
    packet = ObservationPacket(
        modality="rgbd",
        sensor_id="camera0:rgbd",
        timestamp=frame.timestamp,
        payload={"rgb": frame.rgb.unsqueeze(0), "depth": frame.depth.unsqueeze(0)},
        calibration={
            "world_from_camera": frame.world_from_camera.unsqueeze(0),
            "intrinsics": frame.intrinsics.unsqueeze(0),
        },
        frame_id="camera:camera0:rgbd",
        metadata={"image_size": tuple(frame.rgb.shape[-2:])},
    )
    module = RGBDObservationModule(replace(_set_config(), fit_conditioning_limit=10_000.0))
    measured = module.initialise_measurements([packet], context=object())

    assert not bool(measured.measurement_mask.any())
    assert not bool((measured.existence_logits.sigmoid() >= 0.55).any())
    full_probability = measured.auxiliary["set_full_mask_logits"].softmax(dim=1)
    torch.testing.assert_close(
        full_probability[:, :1],
        torch.ones_like(full_probability[:, :1]),
        rtol=0.0,
        atol=1.0e-7,
    )


def test_project_config_accepts_exact_set_contract_and_migrates_legacy_defaults() -> None:
    legacy = load_config(CONFIG_DIR / "rgbd_online_free_motion_cpu.yaml")
    payload = {"config": deepcopy(legacy.to_dict())}
    payload["config"]["model"]["rgbd"].pop("observation_mode")
    payload["config"]["model"]["rgbd"].pop("max_objects")
    validate_checkpoint_config(payload, legacy)

    set_config = replace(
        legacy,
        model=replace(
            legacy.model,
            max_objects=6,
            state=replace(legacy.model.state, appearance_dim=8),
            rgbd=replace(
                legacy.model.rgbd,
                observation_mode="set",
                max_objects=6,
                proposal_count=8,
                temporal_min_samples=3,
            ),
        ),
        simulator=replace(legacy.simulator, min_objects=1, max_objects=6),
    )
    set_config.validate()
