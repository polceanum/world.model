from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from world_model.datasets import SyntheticSphereDataset, collate_episodes
from world_model.runtime import OnlineWorldModel
from world_model.training.checkpointing import (
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint_config,
)
from world_model.training.loop import (
    move_batch_to_device,
    pretrain_rgb_measurements,
)
from world_model.utils.config import load_config


def _small_config():
    config = load_config("configs/tiny_overfit.yaml")
    return replace(
        config,
        simulator=replace(
            config.simulator,
            image_size=(32, 32),
            sequence_frames=3,
            min_objects=1,
            max_objects=1,
        ),
        training=replace(
            config.training,
            batch_size=1,
            train_episodes=1,
            validation_episodes=1,
            tbptt_steps=2,
            horizon_weights=(1.0,),
        ),
        evaluation=replace(
            config.evaluation,
            horizons_seconds=(0.05,),
            episodes=1,
        ),
    )


def test_checkpoint_roundtrip_preserves_trained_state(tmp_path):
    config = _small_config()
    dataset = SyntheticSphereDataset(
        config,
        split="train",
        num_episodes=1,
        memory_cache=True,
    )
    batch = move_batch_to_device(
        collate_episodes([dataset[0]]),
        "cpu",
    )
    model = OnlineWorldModel.from_config(config, device="cpu")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
    )
    optimizer.zero_grad(set_to_none=True)
    result = pretrain_rgb_measurements(
        model,
        batch,
        config,
        frame_index=0,
    )
    assert torch.isfinite(result.total_loss)
    result.total_loss.backward()
    optimizer.step()

    checkpoint = save_checkpoint(
        tmp_path / "last.pt",
        model=model,
        optimizer=optimizer,
        config=config,
        step=1,
        metrics={"loss_total": float(result.total_loss.detach())},
        device="cpu",
    )
    restored = OnlineWorldModel.from_config(config, device="cpu")
    restored_optimizer = torch.optim.AdamW(
        restored.parameters(),
        lr=config.training.learning_rate,
    )
    payload = load_checkpoint(
        checkpoint,
        model=restored,
        optimizer=restored_optimizer,
        map_location="cpu",
        expected_config=config,
    )

    assert payload["step"] == 1
    assert payload["config"]["model"] == config.to_dict()["model"]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(
            restored.state_dict()[name],
            value,
            rtol=0,
            atol=0,
        )
    assert restored_optimizer.state_dict()["state"]

    changed_bounds = list(config.simulator.world_bounds)
    changed_bounds[0] = (
        changed_bounds[0][0] - 0.25,
        changed_bounds[0][1],
    )
    incompatible = replace(
        config,
        simulator=replace(
            config.simulator,
            world_bounds=tuple(changed_bounds),
        ),
    )
    incompatible.validate()
    with pytest.raises(ValueError, match=r"simulator\.world_bounds"):
        validate_checkpoint_config(payload, incompatible)


def test_rgb_runtime_controls_are_semantic_with_legacy_defaults() -> None:
    config = _small_config()
    payload = {"config": config.to_dict()}
    legacy_payload = deepcopy(payload)
    legacy_rgb = legacy_payload["config"]["model"]["rgb"]
    for field_name in (
        "temporal_velocity_enabled",
        "temporal_velocity_history_size",
        "temporal_velocity_min_dt",
        "temporal_velocity_variance_scale",
        "temporal_velocity_variance_floor",
        "temporal_velocity_variance_ceiling",
        "structured_disc_center_enabled",
        "structured_disc_threshold",
        "structured_disc_min_pixels",
        "structured_disc_max_assignment_distance",
        "structured_disc_center_std_pixels",
    ):
        legacy_rgb.pop(field_name)

    enabled = replace(
        config,
        model=replace(
            config.model,
            rgb=replace(
                config.model.rgb,
                temporal_velocity_enabled=True,
                temporal_velocity_history_size=4,
                temporal_velocity_min_dt=0.002,
                temporal_velocity_variance_scale=2.0,
                temporal_velocity_variance_floor=0.5,
                temporal_velocity_variance_ceiling=4.0,
                structured_disc_center_enabled=True,
                structured_disc_threshold=0.06,
                structured_disc_min_pixels=6,
                structured_disc_max_assignment_distance=0.5,
                structured_disc_center_std_pixels=1.0,
            ),
        ),
    )
    enabled.validate()
    validate_checkpoint_config(payload, config)
    legacy_compatible = replace(
        config,
        model=replace(
            config.model,
            rgb=replace(config.model.rgb, structured_disc_center_enabled=False),
        ),
    )
    legacy_compatible.validate()
    validate_checkpoint_config(legacy_payload, legacy_compatible)
    with pytest.raises(ValueError, match="model"):
        validate_checkpoint_config(legacy_payload, config)
    with pytest.raises(ValueError, match="model"):
        validate_checkpoint_config(payload, enabled)
    with pytest.raises(ValueError, match="model"):
        validate_checkpoint_config(legacy_payload, enabled)

    architecture_change = replace(
        enabled,
        model=replace(
            enabled.model,
            rgb=replace(enabled.model.rgb, roi_size=enabled.model.rgb.roi_size + 1),
        ),
    )
    architecture_change.validate()
    with pytest.raises(ValueError, match="model"):
        validate_checkpoint_config(payload, architecture_change)
