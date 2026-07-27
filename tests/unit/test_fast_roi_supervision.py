from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch
from torch import Tensor, nn

from world_model.datasets import collate_episodes
from world_model.observations import MeasurementSet
from world_model.runtime import OnlineWorldModel
from world_model.simulator import generate_episode
from world_model.training.loop import (
    run_closed_loop_batch,
    supervised_measurement_losses,
    supervised_slot_measurement_losses,
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


def test_fast_roi_supervision_masks_unusable_or_unmatched_slots() -> None:
    batch = _batch()
    batch["labels"]["visible"][0, 0, 1] = False
    measurements = _measurements(
        torch.zeros((1, 3, 7)),
        mask=torch.tensor([[True, True, True]]),
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
    torch.testing.assert_close(
        module.targets["values"][0, 2],
        torch.zeros(7),
    )


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
    assert result.metrics["fast_supervised_frames"] == 2.0
    assert model.belief is not None
    torch.testing.assert_close(model.belief.timestamp, batch["timestamps"][:, -1])
