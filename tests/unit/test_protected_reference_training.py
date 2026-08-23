from __future__ import annotations

import random
from dataclasses import replace
from types import SimpleNamespace

import torch

import world_model.training.trainer as training_trainer
from world_model.training.loop import ProtectedObjectiveCell, TrainingBatchResult
from world_model.utils.config import load_config


def test_protected_reference_pair_replays_rng_and_adds_only_the_hinge(monkeypatch) -> None:
    source = load_config("configs/protected_state_event_updater_xy_repair_cpu.yaml")
    config = replace(
        source,
        training=replace(
            source.training,
            closed_loop_protected_reference_nonregression_weight=2.0,
        ),
    )
    candidate_model = SimpleNamespace(is_reference=False)
    reference_model = SimpleNamespace(is_reference=True)
    draws: list[tuple[bool, float, float]] = []

    def fake_run(model, *_args, **kwargs) -> TrainingBatchResult:
        assert kwargs["collect_protected_objective_cells"]
        draws.append((model.is_reference, random.random(), float(torch.rand(()))))
        error = torch.tensor(
            [1.0 if model.is_reference else 2.0],
            requires_grad=not model.is_reference,
        )
        return TrainingBatchResult(
            total_loss=torch.tensor(3.0, requires_grad=not model.is_reference),
            loss_terms={"state_position": torch.tensor(3.0)},
            metrics={},
            phase="closed_loop_rgb",
            protected_objective_cells={
                "state_position_x@current": ProtectedObjectiveCell(
                    error_sum=error,
                    coordinate_count=torch.ones(1, dtype=torch.int64),
                )
            },
        )

    monkeypatch.setattr(training_trainer, "run_closed_loop_batch", fake_run)
    random.seed(123)
    torch.manual_seed(123)
    result = training_trainer._closed_loop_result_with_protected_reference(
        candidate_model,
        reference_model,
        {},
        config,
        device=torch.device("cpu"),
        window_start=0,
        window_steps=1,
        active_trainable_scope=config.training.closed_loop_trainable_scope,
    )
    next_python = random.random()
    next_torch = float(torch.rand(()))

    assert draws[0][0] is False
    assert draws[1][0] is True
    assert draws[0][1:] == draws[1][1:]
    torch.testing.assert_close(result.total_loss, torch.tensor(5.0))
    torch.testing.assert_close(
        result.loss_terms["protected_reference_nonregression"],
        torch.tensor(1.0),
    )
    assert result.metrics["protected_reference_nonregression_active"] == 1.0
    assert result.metrics["protected_reference_replay_call_count"] == 1.0

    random.seed(123)
    torch.manual_seed(123)
    random.random()
    torch.rand(())
    assert next_python == random.random()
    assert next_torch == float(torch.rand(()))


def test_zero_weight_protected_reference_path_is_the_single_legacy_call(monkeypatch) -> None:
    config = load_config("configs/protected_state_event_updater_xy_repair_cpu.yaml")
    calls = 0

    def fake_run(*_args, **kwargs) -> TrainingBatchResult:
        nonlocal calls
        calls += 1
        assert "collect_protected_objective_cells" not in kwargs
        return TrainingBatchResult(
            total_loss=torch.tensor(1.0),
            loss_terms={},
            metrics={},
            phase="closed_loop_rgb",
        )

    monkeypatch.setattr(training_trainer, "run_closed_loop_batch", fake_run)
    result = training_trainer._closed_loop_result_with_protected_reference(
        SimpleNamespace(),
        None,
        {},
        config,
        device=torch.device("cpu"),
        window_start=0,
        window_steps=1,
        active_trainable_scope=config.training.closed_loop_trainable_scope,
    )

    assert calls == 1
    assert result.total_loss.item() == 1.0
    assert "protected_reference_nonregression" not in result.loss_terms
