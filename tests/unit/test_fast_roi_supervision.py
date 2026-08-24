from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import Tensor, nn

import world_model.training.loop as training_loop
from world_model.datasets import collate_episodes
from world_model.observations import MeasurementSet, SensorContext
from world_model.observations.rgb.losses import rgb_measurement_losses
from world_model.runtime import OnlineWorldModel
from world_model.simulator import generate_episode
from world_model.training.loop import (
    _fast_pair_metrics,
    _weighted_measurement_total,
    make_rgb_packet,
    physical_validation_metrics,
    pretrain_rgb_measurements,
    run_closed_loop_batch,
    supervised_measurement_losses,
    supervised_slot_measurement_losses,
)
from world_model.training.trainer import (
    set_closed_loop_trainable_scope,
    set_rgb_pretrain_trainable_scope,
)
from world_model.utils.config import load_config


class _RecordingLossModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.outputs: dict[str, Tensor] = {}
        self.targets: dict[str, Tensor] = {}
        self.masks: dict[str, Tensor] = {}

    def training_losses(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor],
        masks: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        self.outputs = outputs
        self.targets = targets
        self.masks = masks
        error = (outputs["values"] - targets["values"]).abs()
        return {"recorded_error": error[masks["matched"]].mean()}


class _StructuredRecordingLossModule(_RecordingLossModule):
    def training_losses(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor],
        masks: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        self.outputs = outputs
        self.targets = targets
        self.masks = masks
        return rgb_measurement_losses(outputs, targets, masks)


def _batch() -> dict[str, Any]:
    return {
        "labels": {
            "projected_center": torch.tensor(
                [[[[-0.8, -0.2], [0.7, 0.3]]]],
                dtype=torch.float32,
            ),
            "log_apparent_radius_normalized": torch.tensor(
                [[[-2.0, -1.0]]],
                dtype=torch.float32,
            ),
            "inverse_depth": torch.tensor(
                [[[0.2, 0.4]]],
                dtype=torch.float32,
            ),
            "albedo": torch.tensor(
                [[[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]]],
                dtype=torch.float32,
            ),
            "existence": torch.tensor([[[True, True]]]),
            "projected_valid": torch.tensor([[[True, True]]]),
            "visible": torch.tensor([[[True, True]]]),
            "visible_fraction": torch.tensor(
                [[[0.9, 0.6]]],
                dtype=torch.float32,
            ),
        },
        "objects": {
            "position": torch.tensor(
                [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]],
                dtype=torch.float32,
            ),
        },
    }


def _measurements(
    values: Tensor,
    *,
    mask: Tensor | None = None,
    world_position: Tensor | None = None,
    world_position_log_variance: Tensor | None = None,
) -> MeasurementSet:
    batch, measurements, dimensions = values.shape
    if mask is None:
        mask = torch.ones((batch, measurements), dtype=torch.bool)
    auxiliary = {
        "visibility_logits": torch.zeros((batch, measurements)),
    }
    if world_position is not None:
        auxiliary["world_position"] = world_position
    if world_position_log_variance is not None:
        auxiliary["world_position_log_variance"] = world_position_log_variance
    return MeasurementSet(
        modality="rgb",
        sensor_id="camera0",
        timestamp=torch.zeros(batch),
        values=values,
        log_variance=torch.zeros((batch, measurements, dimensions)),
        existence_logits=torch.zeros((batch, measurements)),
        measurement_mask=mask,
        appearance=None,
        class_logits=None,
        frame_id="camera:camera0",
        supported_state_fields=("position",),
        auxiliary=auxiliary,
    )


def test_fast_roi_targets_follow_belief_slot_assignment_not_measurement_order() -> None:
    batch = _batch()
    target_values = torch.cat(
        (
            batch["labels"]["projected_center"][:, 0],
            batch["labels"]["log_apparent_radius_normalized"][:, 0].unsqueeze(-1),
            batch["labels"]["inverse_depth"][:, 0].unsqueeze(-1),
            batch["labels"]["albedo"][:, 0],
        ),
        dim=-1,
    )
    # Each prediction is numerically closest to the other target. A free
    # Hungarian assignment would swap them, but the ROI queries are already
    # bound to belief slots 0 and 1.
    measurements = _measurements(target_values.flip(1))
    module = _RecordingLossModule()

    losses = supervised_slot_measurement_losses(
        module,
        measurements,
        batch,
        frame_index=0,
        target_indices=torch.tensor([[0, 1]], dtype=torch.int64),
        matched_slots=torch.tensor([[True, True]]),
    )

    torch.testing.assert_close(module.targets["values"], target_values)
    assert module.masks["matched"].tolist() == [[True, True]]
    assert float(losses["recorded_error"]) > 0.0


def test_fast_roi_passes_aligned_metric_world_supervision() -> None:
    batch = _batch()
    world_position = torch.tensor(
        [[[40.0, 50.0, 60.0], [10.0, 20.0, 30.0]]],
    )
    world_log_variance = torch.full_like(world_position, -2.0)
    measurements = _measurements(
        torch.zeros((1, 2, 7)),
        world_position=world_position,
        world_position_log_variance=world_log_variance,
    )
    module = _RecordingLossModule()

    supervised_slot_measurement_losses(
        module,
        measurements,
        batch,
        frame_index=0,
        target_indices=torch.tensor([[1, 0]], dtype=torch.int64),
        matched_slots=torch.tensor([[True, True]]),
    )

    torch.testing.assert_close(module.outputs["world_position"], world_position)
    torch.testing.assert_close(
        module.outputs["world_position_log_variance"],
        world_log_variance,
    )
    torch.testing.assert_close(
        module.targets["world_position"],
        batch["objects"]["position"][:, 0].flip(1),
    )


def test_global_proposals_pass_hungarian_aligned_metric_world_supervision() -> None:
    batch = _batch()
    target_values = torch.cat(
        (
            batch["labels"]["projected_center"][:, 0],
            batch["labels"]["log_apparent_radius_normalized"][:, 0].unsqueeze(-1),
            batch["labels"]["inverse_depth"][:, 0].unsqueeze(-1),
            batch["labels"]["albedo"][:, 0],
        ),
        dim=-1,
    )
    world_position = torch.tensor(
        [[[40.0, 50.0, 60.0], [10.0, 20.0, 30.0]]],
    )
    world_log_variance = torch.full_like(world_position, -1.0)
    measurements = _measurements(
        target_values.flip(1),
        world_position=world_position,
        world_position_log_variance=world_log_variance,
    )
    module = _RecordingLossModule()

    supervised_measurement_losses(
        module,
        measurements,
        batch,
        frame_index=0,
    )

    torch.testing.assert_close(module.outputs["world_position"], world_position)
    torch.testing.assert_close(
        module.outputs["world_position_log_variance"],
        world_log_variance,
    )
    torch.testing.assert_close(
        module.targets["world_position"],
        batch["objects"]["position"][:, 0].flip(1),
    )


def test_global_existence_supervision_uses_raw_detector_logits() -> None:
    batch = _batch()
    target_values = torch.cat(
        (
            batch["labels"]["projected_center"][:, 0],
            batch["labels"]["log_apparent_radius_normalized"][:, 0].unsqueeze(-1),
            batch["labels"]["inverse_depth"][:, 0].unsqueeze(-1),
            batch["labels"]["albedo"][:, 0],
        ),
        dim=-1,
    )
    measurements = _measurements(target_values)
    runtime_logits = torch.full((1, 2), torch.logit(torch.tensor(0.995)))
    raw_logits = torch.tensor([[-3.0, 2.0]], requires_grad=True)
    measurements = replace(
        measurements,
        existence_logits=runtime_logits,
        auxiliary={
            **measurements.auxiliary,
            "raw_existence_logits": raw_logits,
        },
    )
    module = _StructuredRecordingLossModule()

    losses = supervised_measurement_losses(module, measurements, batch, frame_index=0)

    assert module.outputs["existence_logits"] is raw_logits
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        raw_logits,
        torch.ones_like(raw_logits),
    )
    torch.testing.assert_close(losses["rgb_existence"], expected)
    assert not torch.isclose(
        losses["rgb_existence"],
        torch.nn.functional.binary_cross_entropy_with_logits(
            runtime_logits,
            torch.ones_like(runtime_logits),
        ),
    )


def test_global_raw_existence_logits_must_match_runtime_shape() -> None:
    measurements = _measurements(torch.zeros((1, 2, 7)))
    measurements = replace(
        measurements,
        auxiliary={
            **measurements.auxiliary,
            "raw_existence_logits": torch.zeros((1, 1)),
        },
    )

    with pytest.raises(ValueError, match="raw_existence_logits must match"):
        supervised_measurement_losses(
            _RecordingLossModule(),
            measurements,
            _batch(),
            frame_index=0,
        )


def test_global_exact_geometry_requires_reliable_visibility_and_an_in_frame_centre() -> None:
    batch = _batch()
    target_values = torch.cat(
        (
            batch["labels"]["projected_center"][:, 0],
            batch["labels"]["log_apparent_radius_normalized"][:, 0].unsqueeze(-1),
            batch["labels"]["inverse_depth"][:, 0].unsqueeze(-1),
            batch["labels"]["albedo"][:, 0],
        ),
        dim=-1,
    )
    batch["labels"]["visible_fraction"][0, 0, 0] = 0.1
    measurements = _measurements(target_values)
    module = _RecordingLossModule()

    supervised_measurement_losses(
        module,
        measurements,
        batch,
        frame_index=0,
    )

    assert module.masks["matched"].tolist() == [[True, True]]
    assert module.masks["existence"].tolist() == [[True, True]]
    assert module.masks["existence_valid"].tolist() == [[True, True]]
    assert module.masks["geometry"].tolist() == [[False, True]]

    batch["labels"]["projected_center"][0, 0, 1, 0] = 1.2
    clipped_values = target_values.clone()
    clipped_values[0, 1, 0] = 1.2
    clipped_module = _RecordingLossModule()
    supervised_measurement_losses(
        clipped_module,
        _measurements(clipped_values),
        batch,
        frame_index=0,
    )

    assert clipped_module.masks["matched"].tolist() == [[True, True]]
    assert clipped_module.masks["geometry"].tolist() == [[False, False]]


def test_fast_roi_supervision_masks_unusable_or_unmatched_slots() -> None:
    batch = _batch()
    batch["labels"]["visible"][0, 0, 1] = False
    measurements = _measurements(
        torch.zeros((1, 3, 7)),
        mask=torch.tensor([[True, True, False]]),
    )
    module = _RecordingLossModule()

    supervised_slot_measurement_losses(
        module,
        measurements,
        batch,
        frame_index=0,
        target_indices=torch.tensor([[0, 1, -1]], dtype=torch.int64),
        matched_slots=torch.tensor([[True, True, False]]),
    )

    assert module.masks["matched"].tolist() == [[True, False, False]]
    assert module.masks["existence"].tolist() == [[True, False, False]]
    # The mapped but invisible target is a valid empty-crop negative. The
    # measurement-masked padding slot remains invalid.
    assert module.masks["existence_valid"].tolist() == [[True, True, False]]
    assert module.masks["visibility_valid"].tolist() == [[True, True, False]]
    torch.testing.assert_close(
        module.targets["visibility"],
        torch.tensor([[0.9, 0.0, 0.0]]),
    )
    torch.testing.assert_close(
        module.targets["values"][0, 2],
        torch.zeros(7),
    )


def test_fast_roi_valid_unmapped_query_trains_only_negative_confidence() -> None:
    batch = _batch()
    values = torch.zeros((1, 3, 7), requires_grad=True)
    measurements = _measurements(values)
    measurements.log_variance = torch.zeros_like(values, requires_grad=True)
    measurements.existence_logits = torch.tensor(
        [[0.0, 0.0, 2.0]],
        requires_grad=True,
    )
    measurements.auxiliary["visibility_logits"] = torch.tensor(
        [[0.0, 0.0, 2.0]],
        requires_grad=True,
    )
    module = _StructuredRecordingLossModule()

    losses = supervised_slot_measurement_losses(
        module,
        measurements,
        batch,
        frame_index=0,
        target_indices=torch.tensor([[0, 1, -1]], dtype=torch.int64),
        matched_slots=torch.tensor([[True, True, False]]),
    )

    assert module.masks["slot_identity"].tolist() == [[True, True, False]]
    assert module.masks["roi_valid"].tolist() == [[True, True, True]]
    assert module.masks["crop_evidence"].tolist() == [[True, True, False]]
    assert module.masks["existence"].tolist() == [[True, True, False]]
    assert module.masks["existence_valid"].tolist() == [[True, True, True]]
    assert module.masks["visibility_valid"].tolist() == [[True, True, True]]
    assert module.targets["visibility"][0, 2].item() == 0.0

    sum(losses.values()).backward()
    # A high-confidence unmapped query receives a real gradient toward zero
    # existence/visibility, while fabricated geometry/colour/NLL targets remain
    # fully disconnected for that query.
    assert measurements.existence_logits.grad is not None
    assert measurements.existence_logits.grad[0, 2] > 0
    visibility_logits = measurements.auxiliary["visibility_logits"]
    assert visibility_logits.grad is not None
    assert visibility_logits.grad[0, 2] > 0
    assert values.grad is not None
    torch.testing.assert_close(values.grad[0, 2], torch.zeros(7))
    assert measurements.log_variance.grad is not None
    torch.testing.assert_close(
        measurements.log_variance.grad[0, 2],
        torch.zeros(7),
    )


def test_fast_roi_stale_nonnegative_identity_cannot_create_positive_crop_evidence() -> None:
    batch = _batch()
    measurements = _measurements(torch.zeros((1, 1, 7)))
    module = _RecordingLossModule()

    supervised_slot_measurement_losses(
        module,
        measurements,
        batch,
        frame_index=0,
        # Association explicitly rejected this slot, but a stale index remains.
        target_indices=torch.tensor([[0]], dtype=torch.int64),
        matched_slots=torch.tensor([[False]]),
        roi_bounds=torch.tensor([[[-1.0, -1.0, 0.0, 0.5]]]),
    )

    assert module.masks["roi_valid"].tolist() == [[True]]
    assert module.masks["crop_evidence"].tolist() == [[False]]
    assert module.masks["existence"].tolist() == [[False]]
    assert module.masks["geometry"].tolist() == [[False]]
    assert module.targets["visibility"].tolist() == [[0.0]]


def test_fast_roi_exact_geometry_requires_target_pixel_support_in_crop() -> None:
    batch = _batch()
    measurements = _measurements(torch.zeros((1, 2, 7)))
    module = _RecordingLossModule()

    supervised_slot_measurement_losses(
        module,
        measurements,
        batch,
        frame_index=0,
        target_indices=torch.tensor([[0, 1]], dtype=torch.int64),
        matched_slots=torch.tensor([[True, True]]),
        roi_bounds=torch.tensor(
            [[[-1.0, -0.5, -0.6, 0.1], [-0.9, -0.9, -0.5, -0.5]]],
        ),
    )

    assert module.masks["slot_identity"].tolist() == [[True, True]]
    assert module.masks["roi_valid"].tolist() == [[True, True]]
    assert module.masks["crop_evidence"].tolist() == [[True, False]]
    assert module.masks["matched"].tolist() == [[True, False]]
    assert module.masks["existence"].tolist() == [[True, False]]
    assert module.masks["existence_valid"].tolist() == [[True, True]]
    assert module.masks["visibility_valid"].tolist() == [[True, True]]
    assert module.masks["geometry"].tolist() == [[True, False]]
    assert module.targets["visibility"][0, 1].item() == 0.0


def test_fast_roi_exact_geometry_also_requires_reliable_visibility_and_centre() -> None:
    batch = _batch()
    batch["labels"]["visible_fraction"][0, 0, 0] = 0.49
    batch["labels"]["projected_center"][0, 0, 1, 0] = 1.1
    measurements = _measurements(torch.zeros((1, 2, 7)))
    module = _RecordingLossModule()

    supervised_slot_measurement_losses(
        module,
        measurements,
        batch,
        frame_index=0,
        target_indices=torch.tensor([[0, 1]], dtype=torch.int64),
        matched_slots=torch.tensor([[True, True]]),
        roi_bounds=torch.tensor(
            [[[-1.0, -1.0, 0.0, 0.5], [0.0, -0.5, 1.5, 1.0]]],
        ),
    )

    assert module.masks["crop_evidence"].tolist() == [[True, True]]
    assert module.masks["geometry"].tolist() == [[False, False]]
    assert module.masks["existence_valid"].tolist() == [[True, True]]


def test_fast_roi_precision_counts_confident_empty_crop_outputs_as_false_positives() -> None:
    batch = _batch()
    target_world = batch["objects"]["position"][:, 0]
    measurements = _measurements(
        torch.zeros((1, 2, 7)),
        world_position=target_world.clone(),
    )
    measurements = MeasurementSet(
        modality=measurements.modality,
        sensor_id=measurements.sensor_id,
        timestamp=measurements.timestamp,
        values=measurements.values,
        log_variance=measurements.log_variance,
        existence_logits=torch.full((1, 2), 10.0),
        measurement_mask=measurements.measurement_mask,
        appearance=measurements.appearance,
        class_logits=measurements.class_logits,
        frame_id=measurements.frame_id,
        supported_state_fields=measurements.supported_state_fields,
        auxiliary=measurements.auxiliary,
    )
    metrics = _fast_pair_metrics(
        SimpleNamespace(
            associator=SimpleNamespace(minimum_measurement_confidence=0.5),
        ),
        measurements,
        SimpleNamespace(auxiliary={"world_position": target_world.clone()}),
        batch,
        0,
        0,
        torch.tensor([[0, 1]], dtype=torch.int64),
        torch.tensor([[True, True]]),
        torch.tensor([[True, True]]),
        torch.tensor([[True, False]]),
    )

    assert metrics["rgb_fast_roi_confident_proposal_count"] == 2.0
    assert metrics["rgb_fast_roi_true_positive_count_at_0_5m"] == 1.0
    assert metrics["rgb_fast_roi_precision_at_0_5m"] == 0.5


def test_fast_roi_precision_counts_confident_unmapped_query_as_false_positive() -> None:
    batch = _batch()
    target_world = batch["objects"]["position"][:, 0]
    measurements = _measurements(
        torch.zeros((1, 2, 7)),
        world_position=target_world.clone(),
    )
    measurements.existence_logits.fill_(10.0)

    metrics = _fast_pair_metrics(
        SimpleNamespace(
            associator=SimpleNamespace(minimum_measurement_confidence=0.5),
        ),
        measurements,
        SimpleNamespace(auxiliary={"world_position": target_world.clone()}),
        batch,
        0,
        0,
        torch.tensor([[0, -1]], dtype=torch.int64),
        torch.tensor([[True, False]]),
        torch.tensor([[True, True]]),
        torch.tensor([[True, False]]),
    )

    assert metrics["rgb_fast_roi_confident_proposal_count"] == 2.0
    assert metrics["rgb_fast_roi_true_positive_count_at_0_5m"] == 1.0
    assert metrics["rgb_fast_roi_precision_at_0_5m"] == 0.5


def _closed_loop_config() -> Any:
    config = load_config("configs/tiny_overfit.yaml")
    config = replace(
        config,
        simulator=replace(
            config.simulator,
            image_size=(32, 32),
            sequence_frames=4,
            min_objects=1,
            max_objects=1,
            camera_motion="fixed",
            render_noise_std=0.0,
        ),
        model=replace(
            config.model,
            rgb=replace(
                config.model.rgb,
                global_every_steps=100,
                global_uncertainty_threshold=1.0e6,
                surprise_threshold=1.0e6,
            ),
            lifecycle=replace(
                config.model.lifecycle,
                birth_confidence=0.0,
            ),
        ),
        training=replace(
            config.training,
            batch_size=1,
            tbptt_steps=4,
            horizon_weights=(1.0,),
        ),
        evaluation=replace(
            config.evaluation,
            horizons_seconds=(0.05,),
            episodes=1,
        ),
    )
    config.validate()
    return config


def test_global_detector_pretrain_real_backward_and_step_have_exact_ownership() -> None:
    config = _closed_loop_config()
    batch = collate_episodes([generate_episode(config, seed=7)])
    model = OnlineWorldModel.from_config(config, device="cpu")
    set_rgb_pretrain_trainable_scope(model, scope="global_detector")
    detector_prefix = "observation_modules.rgb.global_detector."
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.01,
        weight_decay=config.training.weight_decay,
    )

    result = pretrain_rgb_measurements(model, batch, config, frame_index=0)
    assert result.total_loss.requires_grad
    assert torch.isfinite(result.total_loss)
    result.total_loss.backward()

    gradient_owners = {
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
    }
    assert gradient_owners
    assert all(name.startswith(detector_prefix) for name in gradient_owners)
    assert all(
        parameter.grad is None or not bool(torch.count_nonzero(parameter.grad))
        for name, parameter in model.named_parameters()
        if not name.startswith(detector_prefix)
    )

    optimizer.step()
    changed = {
        name
        for name, parameter in model.named_parameters()
        if not torch.equal(before[name], parameter.detach())
    }
    optimizer_owned = {
        name for name, parameter in model.named_parameters() if parameter in optimizer.state
    }
    assert changed == gradient_owners
    assert optimizer_owned == gradient_owners


def test_stage_b_pair_uses_detached_rgb_birth_and_trains_both_perception_paths(
    monkeypatch: Any,
) -> None:
    config = _closed_loop_config()
    batch = collate_episodes([generate_episode(config, seed=7)])
    model = OnlineWorldModel.from_config(config, device="cpu")
    lifecycle_calls = 0
    fast_cache_inputs: list[Any] = []
    original_birth = model.lifecycle.birth_from_measurements
    rgb_module = model.observation_modules["rgb"]
    original_encode = rgb_module.encode_measurements

    def recording_birth(*args: Any, **kwargs: Any) -> Any:
        nonlocal lifecycle_calls
        lifecycle_calls += 1
        measurements = args[1]
        assert isinstance(measurements, MeasurementSet)
        assert not measurements.values.requires_grad
        assert measurements.values.grad_fn is None
        return original_birth(*args, **kwargs)

    def keep_runtime_birth_mapping(
        belief: Any,
        batch_value: Any,
        frame_index: int,
    ) -> tuple[Tensor, Tensor]:
        del batch_value, frame_index
        matched = belief.objects.active.clone()
        indices = torch.where(
            matched,
            torch.zeros_like(belief.objects.object_id),
            torch.full_like(belief.objects.object_id, -1),
        )
        return indices, matched

    def retain_valid_crops(
        batch_value: Any,
        frame_index: int,
        predicted: Any,
        target_indices: Tensor,
        anchor_matched: Tensor,
    ) -> Tensor:
        del batch_value, frame_index, target_indices
        return anchor_matched & predicted.valid_mask

    def recording_encode(
        packets: Any,
        prior: Any,
        predicted: Any,
        cache: Any,
    ) -> Any:
        fast_cache_inputs.append(cache)
        return original_encode(packets, prior, predicted, cache)

    monkeypatch.setattr(model.lifecycle, "birth_from_measurements", recording_birth)
    # The plumbing test makes the support deterministic. Separate unit tests
    # cover metric-distance and target-disc crop censoring.
    monkeypatch.setattr(
        training_loop,
        "_distance_gated_anchor_targets",
        keep_runtime_birth_mapping,
    )
    monkeypatch.setattr(training_loop, "_fast_pair_support", retain_valid_crops)
    monkeypatch.setattr(rgb_module, "encode_measurements", recording_encode)

    result = pretrain_rgb_measurements(
        model,
        batch,
        config,
        frame_index=0,
    )
    result.total_loss.backward()

    rgb = model.observation_modules["rgb"]
    assert lifecycle_calls == 1
    assert result.metrics["rgb_pretrain_pair_anchor_frame"] == 0.0
    assert result.metrics["rgb_pretrain_pair_current_frame"] == 1.0
    assert result.metrics["rgb_pretrain_fast_frame_count"] == 2.0
    assert result.metrics["rgb_pretrain_fast_last_frame"] == 2.0
    assert result.metrics["fast_path_supervised"] == 1.0
    assert result.metrics["fast_supervised_frames"] == 2.0
    assert result.metrics["fast_supervised_slots"] >= 1.0
    assert len(fast_cache_inputs) == 2
    assert fast_cache_inputs[0] is None
    assert fast_cache_inputs[1] is not None
    for component in (
        rgb.backbone.stages[0],
        rgb.backbone.stages[1],
        rgb.backbone.fast_projection,
        rgb.roi_updater,
        rgb.global_detector,
    ):
        assert any(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and bool(torch.count_nonzero(parameter.grad))
            for parameter in component.parameters()
        )


def test_stage_b_pair_trains_unmapped_valid_rgb_births_as_fast_negatives(
    monkeypatch: Any,
) -> None:
    config = _closed_loop_config()
    batch = collate_episodes([generate_episode(config, seed=7)])
    model = OnlineWorldModel.from_config(config, device="cpu")

    def reject_all_birth_mappings(
        belief: Any,
        batch_value: Any,
        frame_index: int,
    ) -> tuple[Tensor, Tensor]:
        del batch_value, frame_index
        return (
            torch.full_like(belief.objects.object_id, -1),
            torch.zeros_like(belief.objects.active),
        )

    monkeypatch.setattr(
        training_loop,
        "_distance_gated_anchor_targets",
        reject_all_birth_mappings,
    )
    result = pretrain_rgb_measurements(
        model,
        batch,
        config,
        frame_index=0,
    )
    result.total_loss.backward()

    rgb = model.observation_modules["rgb"]
    assert result.metrics["fast_path_supervised"] == 1.0
    assert result.metrics["rgb_fast_bootstrap_matched_target_count"] == 0.0
    assert result.metrics["rgb_fast_roi_supported_target_count"] == 0.0
    assert result.metrics["rgb_fast_roi_confident_proposal_count"] >= 0.0
    assert "fast_rgb_existence" in result.metrics
    assert "fast_rgb_visibility" in result.metrics
    assert any(
        parameter.grad is not None for parameter in rgb.backbone.fast_projection.parameters()
    )
    assert any(parameter.grad is not None for parameter in rgb.roi_updater.parameters())
    assert any(parameter.grad is not None for parameter in rgb.global_detector.parameters())


def test_stage_b_anchor_identity_mapping_uses_metric_distance_gate() -> None:
    config = _closed_loop_config()
    batch = collate_episodes([generate_episode(config, seed=7)])
    model = OnlineWorldModel.from_config(config, device="cpu")
    belief = model.belief_factory.create(
        batch_size=1,
        timestamp=float(batch["timestamps"][0, 0]),
        device="cpu",
    )
    target_position = batch["objects"]["position"][:, 0, 0]
    position = belief.objects.position.clone()
    position[:, 0] = target_position + torch.tensor([0.51, 0.0, 0.0])
    active = belief.objects.active.clone()
    active[:, 0] = True
    object_id = belief.objects.object_id.clone()
    object_id[:, 0] = 0
    belief = belief.replace(
        objects=belief.objects.replace(
            position=position,
            active=active,
            object_id=object_id,
        )
    )

    indices, matched = training_loop._distance_gated_anchor_targets(
        belief,
        batch,
        0,
    )

    assert not matched.any()
    assert (indices == -1).all()


def test_stage_b_pair_rejects_anchor_without_an_adjacent_frame() -> None:
    config = _closed_loop_config()
    batch = collate_episodes([generate_episode(config, seed=7)])
    model = OnlineWorldModel.from_config(config, device="cpu")

    with pytest.raises(IndexError, match="adjacent successor"):
        pretrain_rgb_measurements(
            model,
            batch,
            config,
            frame_index=int(batch["rgb"].shape[1]) - 1,
        )


def test_closed_loop_supervises_every_frame_with_a_usable_prior() -> None:
    config = _closed_loop_config()
    batch = collate_episodes([generate_episode(config, seed=7)])
    model = OnlineWorldModel.from_config(config, device="cpu")

    result = run_closed_loop_batch(
        model,
        batch,
        config,
        window_steps=4,
        apply_perturbations=False,
        include_measurement_supervision=True,
    )

    assert result.metrics["fast_path_supervised"] == 1.0
    assert result.metrics["fast_supervised_frames"] == 3.0
    # The runtime contains one mapped track and two valid false tracks. All
    # three queries per frame now carry positive or negative ROI supervision.
    assert result.metrics["fast_supervised_slots"] == 9.0
    fast_weight = config.training.fast_roi_pretrain_weight
    assert result.metrics["measurement"] == pytest.approx(
        (result.metrics["measurement_global"] + fast_weight * result.metrics["measurement_fast"])
        / (1.0 + fast_weight)
    )


def test_state_roi_measurement_backward_is_perception_local(
    monkeypatch: Any,
) -> None:
    """The auxiliary loss must not own the live physical-state graph."""

    config = _closed_loop_config()
    batch = collate_episodes([generate_episode(config, seed=7)])
    model = OnlineWorldModel.from_config(config, device="cpu")
    set_closed_loop_trainable_scope(model, scope="state_roi")
    monkeypatch.setattr(
        training_loop,
        "_PHYSICAL_SELECTION_DISTANCE_THRESHOLD_M",
        10.0,
    )
    live_prior_positions: list[Tensor] = []
    original_prepare = model.prepare_propagation

    def record_live_prior(timestamp: float) -> Any:
        prepared = original_prepare(timestamp)
        position = prepared.prior.objects.position
        if position.requires_grad:
            position.retain_grad()
            live_prior_positions.append(position)
        return prepared

    monkeypatch.setattr(model, "prepare_propagation", record_live_prior)
    result = run_closed_loop_batch(
        model,
        batch,
        config,
        window_steps=4,
        apply_perturbations=False,
        include_measurement_supervision=True,
        rollout_anchors_per_window=1,
        compute_future_correction=False,
        active_trainable_scope="state_roi",
    )

    assert "measurement" in result.loss_terms
    assert result.metrics["fast_supervised_frames"] > 0.0
    model.zero_grad(set_to_none=True)
    result.loss_terms["measurement"].backward(retain_graph=True)

    rgb = model.observation_modules["rgb"]
    for component in (
        rgb.backbone.stages[0],
        rgb.backbone.stages[1],
        rgb.backbone.fast_projection,
        rgb.roi_updater,
    ):
        gradients = [
            parameter.grad
            for parameter in component.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert any(bool(torch.count_nonzero(gradient)) for gradient in gradients)

    assert live_prior_positions
    assert all(position.grad is None for position in live_prior_positions)
    physical_modules: list[nn.Module] = [model.updater, model.dynamics]
    if model.identifier is not None:
        physical_modules.append(model.identifier)
    assert all(
        parameter.grad is None
        for component in physical_modules
        for parameter in component.parameters()
    )

    # The isolation is auxiliary-only: ordinary posterior supervision still
    # reaches the live runtime and its trainable fast-state corrector.
    model.zero_grad(set_to_none=True)
    result.loss_terms["state"].backward()
    updater_gradients = [
        parameter.grad
        for parameter in model.updater.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert updater_gradients
    assert all(torch.isfinite(gradient).all() for gradient in updater_gradients)
    assert any(bool(torch.count_nonzero(gradient)) for gradient in updater_gradients)
    assert any(position.grad is not None for position in live_prior_positions)


def test_detached_fast_auxiliary_preserves_loss_and_perception_gradients() -> None:
    """Detaching the conditioning belief changes ownership, not branch math."""

    config = _closed_loop_config()
    batch = collate_episodes([generate_episode(config, seed=7)])
    model = OnlineWorldModel.from_config(config, device="cpu")
    set_closed_loop_trainable_scope(model, scope="state_roi")
    model.eval()
    first_packet = make_rgb_packet(batch, 0)
    second_packet = make_rgb_packet(batch, 1)
    with torch.no_grad():
        model.ingest(first_packet)
        assert model.belief is not None
        elapsed = second_packet.timestamp - float(model.belief.timestamp[0])
        base_prior = model.dynamics.predict(model.belief, elapsed)

    leaf_position = base_prior.objects.position.detach().clone().requires_grad_(True)
    live_prior = base_prior.replace(
        objects=base_prior.objects.replace(position=leaf_position),
    )
    detached_prior = live_prior.detach().clone()
    context = SensorContext(
        sensor_id=second_packet.sensor_id,
        timestamp=second_packet.timestamp,
        calibration=second_packet.calibration,
        frame_id=second_packet.frame_id,
        image_size=second_packet.metadata["image_size"],
        metadata=second_packet.metadata,
    )
    module = model.observation_modules["rgb"]

    def auxiliary_loss(prior: Any) -> Tensor:
        predicted = module.project(prior, context)
        measured, _ = module.encode_measurements(
            [second_packet],
            prior,
            predicted,
            None,
        )
        matched = prior.objects.active & predicted.valid_mask
        target_indices = torch.where(
            matched,
            torch.zeros_like(prior.objects.object_id),
            torch.full_like(prior.objects.object_id, -1),
        )
        losses = supervised_slot_measurement_losses(
            module,
            measured,
            batch,
            frame_index=1,
            target_indices=target_indices,
            matched_slots=matched,
            roi_bounds=predicted.rois,
        )
        return _weighted_measurement_total(
            losses,
            config.training.measurement_loss_weights,
        )

    live_loss = auxiliary_loss(live_prior)
    detached_loss = auxiliary_loss(detached_prior)
    perception_parameters = tuple(
        parameter
        for component in (
            module.backbone.stages[0],
            module.backbone.stages[1],
            module.backbone.fast_projection,
            module.roi_updater,
        )
        for parameter in component.parameters()
        if parameter.requires_grad
    )
    live_gradients = torch.autograd.grad(
        live_loss,
        (*perception_parameters, leaf_position),
        allow_unused=True,
    )
    detached_gradients = torch.autograd.grad(
        detached_loss,
        perception_parameters,
        allow_unused=True,
    )

    torch.testing.assert_close(detached_loss, live_loss)
    for live_gradient, detached_gradient in zip(
        live_gradients[:-1],
        detached_gradients,
        strict=True,
    ):
        assert (live_gradient is None) == (detached_gradient is None)
        if live_gradient is not None and detached_gradient is not None:
            torch.testing.assert_close(detached_gradient, live_gradient)
    assert any(
        gradient is not None and bool(torch.count_nonzero(gradient))
        for gradient in detached_gradients
    )
    prior_gradient = live_gradients[-1]
    assert prior_gradient is not None
    assert torch.isfinite(prior_gradient).all()
    assert bool(torch.count_nonzero(prior_gradient))


def test_mid_episode_window_burns_in_the_causal_prefix() -> None:
    config = _closed_loop_config()
    batch = collate_episodes([generate_episode(config, seed=7)])
    model = OnlineWorldModel.from_config(config, device="cpu")

    result = run_closed_loop_batch(
        model,
        batch,
        config,
        window_start=2,
        window_steps=2,
        apply_perturbations=False,
        include_measurement_supervision=True,
    )

    # Both trainable frames have a belief derived from frames 0..1. A cold
    # reset at frame 2 would leave only the final frame eligible for ROI loss.
    # Each frame contains one mapped and two valid unmapped persistent queries.
    assert result.metrics["fast_supervised_frames"] == 2.0
    assert result.metrics["fast_supervised_slots"] == 6.0
    assert model.belief is not None
    torch.testing.assert_close(model.belief.timestamp, batch["timestamps"][:, -1])


def test_closed_loop_temporal_observer_supports_multiple_scenes_per_batch() -> None:
    config = _closed_loop_config()
    config = replace(
        config,
        training=replace(config.training, batch_size=2),
    )
    batch = collate_episodes(
        [
            generate_episode(config, seed=7),
            generate_episode(config, seed=8),
        ]
    )
    model = OnlineWorldModel.from_config(config, device="cpu")

    result = run_closed_loop_batch(
        model,
        batch,
        config,
        window_steps=4,
        apply_perturbations=False,
        include_measurement_supervision=True,
    )

    assert torch.isfinite(result.total_loss)
    assert model.belief is not None
    assert model.belief.batch_size == 2
    torch.testing.assert_close(model.belief.timestamp, batch["timestamps"][:, -1])


def test_bounded_rollout_anchors_reuse_posterior_and_emit_physical_metrics(
    monkeypatch: Any,
) -> None:
    # Keep this rollout-call accounting test independent of localization from
    # an untrained RGB detector; matcher gating has dedicated regressions.
    monkeypatch.setattr(
        training_loop,
        "_PHYSICAL_SELECTION_DISTANCE_THRESHOLD_M",
        10.0,
    )
    config = _closed_loop_config()
    batch = collate_episodes([generate_episode(config, seed=7)])
    torch.manual_seed(1)
    model = OnlineWorldModel.from_config(config, device="cpu")
    original_rollout = model.dynamics.rollout
    rollout_calls = 0

    def recording_rollout(*args: Any, **kwargs: Any) -> Any:
        nonlocal rollout_calls
        rollout_calls += 1
        return original_rollout(*args, **kwargs)

    monkeypatch.setattr(model.dynamics, "rollout", recording_rollout)
    result = run_closed_loop_batch(
        model,
        batch,
        config,
        window_steps=4,
        apply_perturbations=False,
        include_measurement_supervision=False,
        rollout_anchors_per_window=2,
        collect_promotion_metrics=True,
    )

    # Eligible anchors are frames 0..2, so the deterministic bound selects
    # frames 0 and 2. Frame 0 needs one posterior rollout; frame 2 needs one
    # prior and one posterior. The posterior also serves correction scoring.
    assert rollout_calls == 3
    assert result.metrics["rollout_anchor_count"] == 2.0
    assert result.metrics["rollout_anchor_candidate_count"] == 3.0
    assert result.metrics["physical_state_position_coordinate_count"] > 0
    assert result.metrics["physical_state_velocity_coordinate_count"] > 0
    assert result.metrics["physical_rollout_position@0.050s_coordinate_count"] > 0
    assert result.metrics["physical_rollout_velocity@0.050s_coordinate_count"] > 0
    for axis_name in ("x", "y", "z"):
        assert f"rollout_position_{axis_name}@0.050s" in result.metrics
    assert result.metrics["physical_target_object_frames"] == 4.0
    assert "physical_collision_f1_proxy" in result.metrics
    assert "physical_identity_switches" in result.metrics
    assert "physical_distance_gated_matched_object_frames" in result.metrics
    assert "physical_distance_gated_identity_switches" in result.metrics
    validation = physical_validation_metrics(result.metrics, config)
    expected_validation_metrics = {
        "validation_position_rmse_m",
        "validation_velocity_rmse_mps",
        "validation_target_coverage",
        "validation_prediction_precision",
        "validation_collision_f1",
        "validation_id_switch_rate",
        "validation_position_coverage90",
        "validation_current_position_coverage90",
        "validation_current_position_gaussian_nll",
        "validation_current_position_sharpness_std",
        "validation_position_rmse@0.050s",
        "validation_velocity_rmse@0.050s",
        "validation_forecast_target_coverage@0.050s",
        "validation_forecast_identity_mismatch_rate@0.050s",
        "validation_forecast_identity_association_coverage@0.050s",
        "validation_position_coverage90@0.050s",
        "validation_position_gaussian_nll@0.050s",
        "validation_position_sharpness_std@0.050s",
        "validation_collision_f1@0.050s",
    }
    for axis_name in ("x", "y", "z"):
        expected_validation_metrics.update(
            {
                f"validation_position_rmse_{axis_name}_m",
                f"validation_velocity_rmse_{axis_name}_mps",
                f"validation_current_position_gaussian_nll_{axis_name}",
                f"validation_current_position_sharpness_std_{axis_name}",
                f"validation_position_rmse_{axis_name}@0.050s",
                f"validation_velocity_rmse_{axis_name}@0.050s",
                f"validation_position_gaussian_nll_{axis_name}@0.050s",
                f"validation_position_sharpness_std_{axis_name}@0.050s",
            }
        )
    assert set(validation) == expected_validation_metrics


def test_validation_can_skip_prior_future_rollouts_without_losing_physical_metrics(
    monkeypatch: Any,
) -> None:
    # Keep this validation-control test independent of localization from an
    # untrained RGB detector; matcher gating has dedicated regressions.
    monkeypatch.setattr(
        training_loop,
        "_PHYSICAL_SELECTION_DISTANCE_THRESHOLD_M",
        10.0,
    )
    config = _closed_loop_config()
    batch = collate_episodes([generate_episode(config, seed=7)])
    torch.manual_seed(1)
    model = OnlineWorldModel.from_config(config, device="cpu")
    original_rollout = model.dynamics.rollout
    rollout_calls = 0

    def recording_rollout(*args: Any, **kwargs: Any) -> Any:
        nonlocal rollout_calls
        rollout_calls += 1
        return original_rollout(*args, **kwargs)

    monkeypatch.setattr(model.dynamics, "rollout", recording_rollout)
    result = run_closed_loop_batch(
        model,
        batch,
        config,
        window_steps=4,
        apply_perturbations=False,
        include_measurement_supervision=False,
        rollout_anchors_per_window=2,
        compute_future_correction=False,
        collect_promotion_metrics=True,
    )

    assert rollout_calls == 2
    assert "correction_future" not in result.metrics
    assert result.metrics["physical_rollout_position@0.050s_coordinate_count"] > 0
    validation = physical_validation_metrics(result.metrics, config)
    assert validation["validation_position_rmse@0.050s"] >= 0.0
