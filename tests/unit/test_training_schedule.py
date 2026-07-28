from __future__ import annotations

import random
from dataclasses import replace

import pytest
import torch
from torch import nn

from world_model.runtime import OnlineWorldModel
from world_model.training.loop import (
    TrainingBatchResult,
    _globally_weight_horizon_details,
    _group_closed_loop_terms,
    _weighted_closed_loop_total,
    _weighted_measurement_total,
    rollout_horizon_loss_key,
    select_closed_loop_window,
)
from world_model.training.trainer import (
    _rollout_selection_is_compatible,
    _validation_loader_result,
    _validation_step,
    measurement_pretrain_frame_index,
    set_closed_loop_trainable_scope,
    set_global_perception_trainable,
)
from world_model.utils.config import load_config


def test_fixed_pretraining_sweeps_every_frame_for_every_loader_batch() -> None:
    loader_batches = 4
    total_frames = 16
    visited = {batch_index: [] for batch_index in range(loader_batches)}

    for step in range(loader_batches * total_frames):
        batch_index = step % loader_batches
        visited[batch_index].append(
            measurement_pretrain_frame_index(
                step,
                loader_batches=loader_batches,
                total_frames=total_frames,
                fixed_dataset=True,
            )
        )

    expected = list(range(total_frames))
    assert all(frame_indices == expected for frame_indices in visited.values())


def test_pretraining_frame_index_rejects_empty_axes() -> None:
    for loader_batches, total_frames in ((0, 16), (4, 0)):
        with pytest.raises(ValueError, match="must be positive"):
            measurement_pretrain_frame_index(
                0,
                loader_batches=loader_batches,
                total_frames=total_frames,
                fixed_dataset=True,
            )


def test_streaming_pretraining_samples_a_valid_frame() -> None:
    sampled = {
        measurement_pretrain_frame_index(
            step,
            loader_batches=4,
            total_frames=7,
            fixed_dataset=False,
        )
        for step in range(32)
    }
    assert sampled
    assert sampled <= set(range(7))


def test_global_perception_freeze_leaves_fast_roi_trainable() -> None:
    model = OnlineWorldModel.from_config(load_config("configs/tiny_overfit.yaml"))
    rgb = model.observation_modules["rgb"]

    set_global_perception_trainable(model, trainable=False)

    assert not any(parameter.requires_grad for parameter in rgb.backbone.parameters())
    assert not any(parameter.requires_grad for parameter in rgb.global_detector.parameters())
    assert all(parameter.requires_grad for parameter in rgb.roi_updater.parameters())
    assert isinstance(rgb.roi_updater, nn.Module)

    set_global_perception_trainable(model, trainable=True)
    assert all(parameter.requires_grad for parameter in rgb.backbone.parameters())
    assert all(parameter.requires_grad for parameter in rgb.global_detector.parameters())


def test_dynamics_only_scope_preserves_rgb_and_filter_weights() -> None:
    model = OnlineWorldModel.from_config(load_config("configs/tiny_overfit.yaml"))

    set_closed_loop_trainable_scope(model, scope="dynamics")

    assert all(parameter.requires_grad for parameter in model.dynamics.parameters())
    assert not any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("dynamics.")
    )

    set_closed_loop_trainable_scope(model, scope="all")
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_closed_loop_terms_expose_physical_components_without_double_counting() -> None:
    reference = torch.zeros(())
    terms = _group_closed_loop_terms(
        {
            "state_position": torch.tensor(1.0),
            "state_velocity": torch.tensor(3.0),
            "rollout_position": torch.tensor(2.0),
            "rollout_position_x": torch.tensor(1.0),
            "rollout_position_y": torch.tensor(2.0),
            "rollout_position_z": torch.tensor(3.0),
            "rollout_velocity": torch.tensor(6.0),
        },
        reference,
    )

    assert terms["state_position"].item() == 1.0
    assert terms["state_velocity"].item() == 3.0
    assert terms["state"].item() == 2.0
    assert terms["rollout_position"].item() == 2.0
    assert terms["rollout_position_x"].item() == 1.0
    assert terms["rollout_velocity"].item() == 6.0
    assert terms["rollout"].item() == 4.0
    torch.testing.assert_close(
        _weighted_closed_loop_total(
            terms,
            {"state": 2.0, "rollout": 3.0},
        ),
        torch.tensor(16.0),
    )
    torch.testing.assert_close(
        _weighted_closed_loop_total(
            terms,
            {
                "state": 100.0,
                "state_position": 5.0,
                "state_velocity": 0.5,
                "rollout_position": 2.0,
                "rollout_velocity": 0.25,
            },
        ),
        torch.tensor(12.0),
    )
    torch.testing.assert_close(
        _weighted_closed_loop_total(
            terms,
            {
                "rollout_position_x": 2.0,
                "rollout_position_y": 1.0,
                "rollout_position_z": 1.0,
                "rollout_velocity": 0.25,
            },
        ),
        torch.tensor(10.5),
    )


def test_measurement_weights_keep_metric_position_primary() -> None:
    losses = {
        "rgb_world_position": torch.tensor(0.2),
        "rgb_raw_centre": torch.tensor(0.1),
        "rgb_nll": torch.tensor(-3.0),
        "future_term": torch.tensor(0.1),
    }

    total = _weighted_measurement_total(
        losses,
        {
            "rgb_world_position": 8.0,
            "rgb_raw_centre": 2.0,
            "rgb_nll": 0.05,
        },
    )

    torch.testing.assert_close(total, torch.tensor(1.75))


def test_rollout_horizons_are_weighted_after_per_horizon_averaging() -> None:
    config = load_config("configs/tiny_overfit.yaml")
    details = {
        rollout_horizon_loss_key("rollout_position", 0.1): torch.tensor(1.0),
        rollout_horizon_loss_key("rollout_position", 0.25): torch.tensor(2.0),
        rollout_horizon_loss_key("rollout_position", 0.5): torch.tensor(4.0),
        rollout_horizon_loss_key("rollout_velocity", 0.1): torch.tensor(3.0),
        rollout_horizon_loss_key("rollout_velocity", 0.25): torch.tensor(3.0),
        rollout_horizon_loss_key("rollout_velocity", 0.5): torch.tensor(3.0),
    }

    balanced = _globally_weight_horizon_details(details, config, torch.zeros(()))

    torch.testing.assert_close(
        balanced["rollout_position"],
        torch.tensor((1.0 * 1.0 + 1.5 * 2.0 + 2.0 * 4.0) / 4.5),
    )
    torch.testing.assert_close(balanced["rollout_velocity"], torch.tensor(3.0))


def test_missing_long_horizon_does_not_renormalize_short_losses() -> None:
    config = load_config("configs/tiny_overfit.yaml")
    details = {
        rollout_horizon_loss_key("rollout_position", 0.1): torch.tensor(1.0),
        rollout_horizon_loss_key("rollout_position", 0.25): torch.tensor(2.0),
    }

    balanced = _globally_weight_horizon_details(details, config, torch.zeros(()))

    torch.testing.assert_close(
        balanced["rollout_position"],
        torch.tensor((1.0 * 1.0 + 1.5 * 2.0) / 4.5),
    )


def test_closed_loop_window_can_be_conditioned_on_collision() -> None:
    batch = {
        "rgb": torch.zeros((2, 10, 3, 8, 8)),
        "events": {
            "collision": torch.zeros((2, 10, 3), dtype=torch.bool),
        },
    }
    batch["events"]["collision"][1, 7, 2] = True
    random.seed(11)

    start = select_closed_loop_window(
        batch,
        4,
        event_condition_probability=1.0,
    )

    assert 0 <= start <= 6
    assert start <= 7 < start + 4


def test_closed_loop_window_can_require_a_maximum_horizon_anchor() -> None:
    batch = {
        "rgb": torch.zeros((2, 16, 3, 8, 8)),
        "events": {
            "collision": torch.zeros((2, 16, 3), dtype=torch.bool),
        },
    }
    random.seed(19)

    starts = {
        select_closed_loop_window(
            batch,
            6,
            event_condition_probability=0.0,
            maximum_rollout_frame_offset=10,
            long_horizon_probability=1.0,
        )
        for _ in range(32)
    }

    assert starts
    assert starts <= set(range(6))


def test_collision_conditioning_takes_priority_over_long_horizon_window() -> None:
    batch = {
        "rgb": torch.zeros((1, 32, 3, 8, 8)),
        "events": {
            "collision": torch.zeros((1, 32, 2), dtype=torch.bool),
        },
    }
    batch["events"]["collision"][0, 31, 0] = True
    random.seed(23)

    start = select_closed_loop_window(
        batch,
        8,
        event_condition_probability=1.0,
        maximum_rollout_frame_offset=20,
        long_horizon_probability=1.0,
    )

    assert start == 24
    assert start <= 31 < start + 8


def test_legacy_rollout_score_is_not_reused_after_objective_fix() -> None:
    config = load_config("configs/tiny_overfit.yaml")
    payload = {
        "config": config.to_dict(),
        "metrics": {
            "best_rollout_validated": 1.0,
            "best_rollout_position_loss": 0.01,
        },
    }

    assert not _rollout_selection_is_compatible(payload, config)
    payload["metrics"]["rollout_selection_metric_version"] = 2.0
    assert _rollout_selection_is_compatible(payload, config)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("simulator", "scenario_mixture", ["baseline", "elastic_pairs"]),
        ("simulator", "sequence_frames", 17),
        ("simulator", "min_objects", 1),
        ("simulator", "max_objects", 3),
        ("training", "validation_episodes", 7),
        ("project", "seed", 99),
    ],
)
def test_rollout_score_is_not_reused_across_validation_protocols(
    section: str,
    field: str,
    value: object,
) -> None:
    config = load_config("configs/tiny_overfit.yaml")
    checkpoint_config = config.to_dict()
    checkpoint_config[section][field] = value
    payload = {
        "config": checkpoint_config,
        "metrics": {
            "best_rollout_validated": 1.0,
            "best_rollout_position_loss": 0.01,
            "rollout_selection_metric_version": 2.0,
        },
    }

    assert not _rollout_selection_is_compatible(payload, config)


class _ModeOnlyModel:
    def __init__(self) -> None:
        self.training = True

    def eval(self) -> _ModeOnlyModel:
        self.training = False
        return self

    def train(self, mode: bool = True) -> _ModeOnlyModel:
        self.training = mode
        return self


def _result(value: float) -> TrainingBatchResult:
    scalar = torch.tensor(value)
    return TrainingBatchResult(
        total_loss=scalar,
        loss_terms={"rollout": scalar},
        metrics={"value": value},
        phase="closed_loop_rgb",
    )


def test_closed_loop_validation_uses_the_full_episode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, int | bool] = {}

    def fake_closed_loop(
        model: object,
        batch: object,
        config: object,
        **kwargs: int | bool,
    ) -> TrainingBatchResult:
        del model, batch, config
        observed.update(kwargs)
        return _result(1.0)

    monkeypatch.setattr(
        "world_model.training.trainer.run_closed_loop_batch",
        fake_closed_loop,
    )
    config = load_config("configs/tiny_overfit.yaml")
    config = replace(
        config,
        training=replace(config.training, tbptt_steps=3),
    )
    model = _ModeOnlyModel()
    batch = {
        "rgb": torch.zeros((2, 9, 3, 8, 8)),
        "timestamps": torch.zeros((2, 9)),
    }

    _validation_step(model, batch, config, closed_loop=True)  # type: ignore[arg-type]

    assert observed["window_start"] == 0
    assert observed["window_steps"] == 9
    assert observed["apply_perturbations"] is False
    assert model.training


def test_validation_aggregates_every_loader_batch_by_episode_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[int] = []

    def fake_validation(
        model: object,
        batch: dict[str, torch.Tensor],
        config: object,
        *,
        closed_loop: bool,
    ) -> TrainingBatchResult:
        del model, config
        assert closed_loop
        seen.append(int(batch["rgb"].shape[0]))
        return _result(float(batch["rgb"][0, 0, 0, 0, 0]))

    monkeypatch.setattr(
        "world_model.training.trainer._validation_step",
        fake_validation,
    )
    loader = [
        {
            "rgb": torch.ones((2, 3, 3, 8, 8)),
            "timestamps": torch.zeros((2, 3)),
        },
        {
            "rgb": torch.full((1, 3, 3, 8, 8), 4.0),
            "timestamps": torch.zeros((1, 3)),
        },
    ]

    result = _validation_loader_result(
        _ModeOnlyModel(),  # type: ignore[arg-type]
        loader,  # type: ignore[arg-type]
        load_config("configs/tiny_overfit.yaml"),
        device=torch.device("cpu"),
        closed_loop=True,
    )

    assert seen == [2, 1]
    torch.testing.assert_close(result.total_loss, torch.tensor(2.0))
    torch.testing.assert_close(result.loss_terms["rollout"], torch.tensor(2.0))
    assert result.metrics["value"] == 2.0
