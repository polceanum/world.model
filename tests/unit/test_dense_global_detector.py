from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from world_model.datasets import collate_episodes
from world_model.observations.rgb.global_detector import DenseGlobalObjectDetector
from world_model.runtime import OnlineWorldModel
from world_model.simulator import generate_episode
from world_model.training.checkpointing import (
    load_model_weights,
    save_checkpoint,
    validate_checkpoint_config,
)
from world_model.training.loop import pretrain_rgb_measurements
from world_model.training.trainer import set_rgb_pretrain_trainable_scope
from world_model.utils.config import OrpheusConfig


def _dense_config() -> OrpheusConfig:
    base = OrpheusConfig()
    config = replace(
        base,
        simulator=replace(
            base.simulator,
            image_size=(32, 32),
            sequence_frames=8,
            min_objects=2,
            max_objects=2,
            ensure_collision=False,
        ),
        model=replace(
            base.model,
            max_objects=3,
            rgb=replace(
                base.model.rgb,
                backbone_channels=(8, 16, 24, 32),
                feature_dim=16,
                proposal_queries=3,
                dense_global_detector_enabled=True,
            ),
        ),
        training=replace(
            base.training,
            batch_size=1,
            train_episodes=1,
            validation_episodes=1,
            rgb_pretrain_trainable_scope="dense_global_detector",
            horizon_weights=(1.0,),
        ),
        evaluation=replace(base.evaluation, horizons_seconds=(0.05,), episodes=1),
    )
    config.validate()
    return config


def test_dense_center_branch_is_the_qualified_architecture_and_emits_typed_rows() -> None:
    detector = DenseGlobalObjectDetector(
        feature_dim=96,
        query_count=8,
        appearance_dim=32,
    )
    center_parameters = sum(
        parameter.numel()
        for module in (detector.trunk, detector.center_head)
        for parameter in module.parameters()
    )
    assert center_parameters == 55_553
    feature_map = torch.randn(2, 96, 16, 16, requires_grad=True)
    output = detector(feature_map)
    assert output.centre.shape == (2, 8, 2)
    assert output.log_radius.shape == (2, 8, 1)
    assert output.inverse_depth_residual.shape == (2, 8, 1)
    assert output.colour.shape == (2, 8, 3)
    assert output.existence_logits.shape == (2, 8)
    assert output.visibility_logits.shape == (2, 8)
    assert output.log_variance.shape == (2, 8, 7)
    assert output.appearance.shape == (2, 8, 32)
    assert output.query_features.shape == (2, 8, 64)
    assert output.attention.shape == (2, 8, 256)
    assert output.dense_center_logits is not None
    assert output.dense_center_logits.shape == (2, 1, 16, 16)
    assert torch.isfinite(output.dense_center_logits).all()


def test_dense_detector_scope_has_exact_owners_and_center_focal_gradient() -> None:
    config = _dense_config()
    batch = collate_episodes([generate_episode(config, seed=11)])
    model = OnlineWorldModel.from_config(config, device="cpu")
    set_rgb_pretrain_trainable_scope(model, scope="dense_global_detector")
    result = pretrain_rgb_measurements(model, batch, config, frame_index=0)
    assert "rgb_dense_center_heatmap" in result.metrics
    assert torch.isfinite(result.total_loss)
    result.total_loss.backward()
    dense_prefix = "observation_modules.rgb.dense_global_detector."
    gradient_owners = {
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
    }
    assert gradient_owners
    assert all(name.startswith(dense_prefix) for name in gradient_owners)
    dense = model.observation_modules["rgb"].dense_global_detector
    assert dense is not None
    assert any(parameter.grad is not None for parameter in dense.trunk.parameters())
    assert any(parameter.grad is not None for parameter in dense.center_head.parameters())
    assert any(parameter.grad is not None for parameter in dense.attribute_head.parameters())
    assert not any(
        parameter.grad is not None
        for parameter in model.observation_modules["rgb"].global_detector.parameters()
    )


def test_dense_detector_weight_growth_is_explicit_and_exact_resume_rejects(
    tmp_path: Path,
) -> None:
    target_config = _dense_config()
    source_config = replace(
        target_config,
        model=replace(
            target_config.model,
            rgb=replace(
                target_config.model.rgb,
                dense_global_detector_enabled=False,
            ),
        ),
        training=replace(
            target_config.training,
            rgb_pretrain_trainable_scope="global_detector",
        ),
    )
    source_config.validate()
    source = OnlineWorldModel.from_config(source_config, device="cpu")
    checkpoint = save_checkpoint(
        tmp_path / "source.pt",
        model=source,
        optimizer=None,
        config=source_config,
        step=0,
    )
    target = OnlineWorldModel.from_config(target_config, device="cpu")
    with pytest.raises(RuntimeError, match="missing required model keys"):
        load_model_weights(checkpoint, model=target)
    payload = load_model_weights(
        checkpoint,
        model=target,
        allowed_missing_prefixes=("observation_modules.rgb.dense_global_detector.",),
        architecture_growth_config=target_config,
    )
    assert payload["initialized_missing_module_prefixes"] == (
        "observation_modules.rgb.dense_global_detector.",
    )
    with pytest.raises(ValueError, match="incompatible for: model"):
        validate_checkpoint_config(payload, target_config)


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_dense_global_detector_enabled_requires_boolean(value: object) -> None:
    base = OrpheusConfig()
    config = replace(
        base,
        model=replace(
            base.model,
            rgb=replace(
                base.model.rgb,
                dense_global_detector_enabled=value,  # type: ignore[arg-type]
            ),
        ),
    )
    with pytest.raises(ValueError, match="dense_global_detector_enabled must be boolean"):
        config.validate()
