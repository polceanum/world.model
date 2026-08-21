from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from typing import Any

import pytest
import torch
from torch import Tensor

from world_model.datasets import collate_episodes
from world_model.runtime import OnlineWorldModel
from world_model.simulator import generate_episode
from world_model.training.checkpointing import validate_training_resume_config
from world_model.training.event_windows import observation_window_query_plan
from world_model.training.loop import (
    TrainingBatchResult,
    _rollout_metadata_equal,
    _select_rollout_anchor_frames,
    _valid_rollout_offsets,
    run_closed_loop_batch,
)
from world_model.training.trainer import (
    _rollout_validation_protocol,
    _rollout_validation_protocol_hash,
)
from world_model.utils.config import OrpheusConfig, load_config
from world_model.utils.version import SIMULATOR_VERSION


def _validation_config() -> OrpheusConfig:
    config = load_config("configs/tiny_overfit.yaml")
    config = replace(
        config,
        simulator=replace(
            config.simulator,
            image_size=(24, 24),
            sequence_frames=16,
            min_objects=2,
            max_objects=2,
            camera_motion="fixed",
            render_noise_std=0.0,
        ),
        model=replace(
            config.model,
            rgb=replace(
                config.model.rgb,
                backbone_channels=(8, 16, 24, 32),
                feature_dim=16,
                proposal_queries=3,
                roi_size=8,
                global_every_steps=100,
                global_uncertainty_threshold=1.0e6,
                surprise_threshold=1.0e6,
            ),
            dynamics=replace(
                config.model.dynamics,
                hidden_dim=24,
                learned_effect_interval_seconds=None,
                smooth_event_hazard_enabled=True,
            ),
            filter=replace(config.model.filter, hidden_dim=32),
            lifecycle=replace(
                config.model.lifecycle,
                birth_confidence=0.0,
                birth_confirmations=1,
            ),
        ),
        training=replace(
            config.training,
            batch_size=1,
            tbptt_steps=16,
            minimum_rollout_age_steps=1,
            validation_rollout_anchors_per_episode=4,
            validation_rollout_anchor_batch_size=4,
            horizon_weights=(1.0, 1.5, 2.0),
        ),
        evaluation=replace(
            config.evaluation,
            horizons_seconds=(0.10, 0.25, 0.50),
            episodes=1,
        ),
    )
    config.validate()
    return config


def _assert_nested_close(actual: Any, expected: Any, *, path: str = "root") -> None:
    if isinstance(actual, Tensor) or isinstance(expected, Tensor):
        assert isinstance(actual, Tensor) and isinstance(expected, Tensor), path
        torch.testing.assert_close(
            actual,
            expected,
            rtol=1.0e-6,
            atol=1.0e-7,
            equal_nan=True,
            msg=lambda message: f"{path}: {message}",
        )
        return
    if is_dataclass(actual) or is_dataclass(expected):
        assert is_dataclass(actual) and is_dataclass(expected), path
        assert type(actual) is type(expected), path
        for item in fields(actual):
            _assert_nested_close(
                getattr(actual, item.name),
                getattr(expected, item.name),
                path=f"{path}.{item.name}",
            )
        return
    if isinstance(actual, Mapping) or isinstance(expected, Mapping):
        assert isinstance(actual, Mapping) and isinstance(expected, Mapping), path
        assert actual.keys() == expected.keys(), path
        for name in actual:
            _assert_nested_close(actual[name], expected[name], path=f"{path}[{name!r}]")
        return
    if isinstance(actual, (tuple, list)) or isinstance(expected, (tuple, list)):
        assert isinstance(actual, type(expected)) and len(actual) == len(expected), path
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected, strict=True)):
            _assert_nested_close(actual_item, expected_item, path=f"{path}[{index}]")
        return
    assert actual == expected, path


def _assert_result_close(actual: TrainingBatchResult, expected: TrainingBatchResult) -> None:
    assert actual.phase == expected.phase
    torch.testing.assert_close(
        actual.total_loss,
        expected.total_loss,
        rtol=1.0e-5,
        atol=1.0e-6,
        equal_nan=True,
    )
    assert actual.loss_terms.keys() == expected.loss_terms.keys()
    for name in actual.loss_terms:
        torch.testing.assert_close(
            actual.loss_terms[name],
            expected.loss_terms[name],
            rtol=1.0e-5,
            atol=1.0e-6,
            equal_nan=True,
            msg=lambda message, key=name: f"loss_terms[{key!r}]: {message}",
        )
    assert actual.metrics.keys() == expected.metrics.keys()
    for name in actual.metrics:
        if name.startswith("rollout_execution_"):
            continue
        torch.testing.assert_close(
            torch.as_tensor(actual.metrics[name]),
            torch.as_tensor(expected.metrics[name]),
            rtol=1.0e-5,
            atol=1.0e-6,
            equal_nan=True,
            msg=lambda message, key=name: f"metrics[{key!r}]: {message}",
        )


def test_batched_validation_rollouts_preserve_serial_losses_events_metrics_and_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _validation_config()
    batch = collate_episodes([generate_episode(config, seed=100003)])
    torch.manual_seed(23)
    serial_model = OnlineWorldModel.from_config(config, device="cpu").eval()
    batched_model = OnlineWorldModel.from_config(config, device="cpu").eval()
    batched_model.load_state_dict(serial_model.state_dict())

    serial_queries: list[Tensor] = []
    batched_queries: list[Tensor] = []
    serial_rollout = serial_model.dynamics.rollout
    batched_rollout = batched_model.dynamics.rollout

    def record_serial(belief: Any, query_times: Any, **kwargs: Any) -> Any:
        serial_queries.append(torch.as_tensor(query_times).detach().cpu().clone())
        return serial_rollout(belief, query_times, **kwargs)

    def record_batched(belief: Any, query_times: Any, **kwargs: Any) -> Any:
        assert kwargs["return_events"] is True
        assert kwargs["return_auxiliary"] is True
        assert kwargs["auxiliary_names"] == ("pair_event_logits",)
        batched_queries.append(torch.as_tensor(query_times).detach().cpu().clone())
        return batched_rollout(belief, query_times, **kwargs)

    monkeypatch.setattr(serial_model.dynamics, "rollout", record_serial)
    monkeypatch.setattr(batched_model.dynamics, "rollout", record_batched)
    with torch.no_grad():
        serial_result = run_closed_loop_batch(
            serial_model,
            batch,
            config,
            window_steps=16,
            apply_perturbations=False,
            include_measurement_supervision=True,
            rollout_anchors_per_window=4,
            validation_rollout_anchor_batch_size=1,
            compute_future_correction=False,
        )
        batched_result = run_closed_loop_batch(
            batched_model,
            batch,
            config,
            window_steps=16,
            apply_perturbations=False,
            include_measurement_supervision=True,
            rollout_anchors_per_window=4,
            validation_rollout_anchor_batch_size=4,
            compute_future_correction=False,
        )

    _assert_result_close(batched_result, serial_result)
    _assert_nested_close(batched_model.state, serial_model.state)
    assert len(serial_queries) == 4
    assert len(batched_queries) == 1

    anchors = _select_rollout_anchor_frames(
        config,
        window_start=0,
        window_stop=16,
        total_frames=16,
        rollout_anchors_per_window=4,
    )
    assert anchors == (1, 5, 9, 13)
    plans = [
        observation_window_query_plan(
            _valid_rollout_offsets(config, anchor, 16)[0],
            frame_rate=config.simulator.frame_rate,
        )
        for anchor in anchors
    ]
    maximum_queries = max(len(plan.query_seconds) for plan in plans)
    expected_rows = torch.tensor(
        [
            plan.query_seconds
            + plan.query_seconds[-1:] * (maximum_queries - len(plan.query_seconds))
            for plan in plans
        ]
    )
    torch.testing.assert_close(batched_queries[0], expected_rows)
    assert plans[-1].target_frame_offsets == (2,)
    assert plans[-1].query_frame_offsets == (1, 2)
    assert batched_result.metrics["rollout_anchor_count"] == 4.0
    assert serial_result.metrics["rollout_execution_batch_requested_anchor_count"] == 0.0
    assert serial_result.metrics["rollout_execution_batched_anchor_count"] == 0.0
    assert serial_result.metrics["rollout_execution_serial_fallback_anchor_count"] == 0.0
    assert serial_result.metrics["rollout_execution_posterior_call_count"] == 4.0
    assert batched_result.metrics["rollout_execution_batch_requested_anchor_count"] == 4.0
    assert batched_result.metrics["rollout_execution_batched_anchor_count"] == 4.0
    assert batched_result.metrics["rollout_execution_serial_fallback_anchor_count"] == 0.0
    assert batched_result.metrics["rollout_execution_posterior_call_count"] == 1.0
    assert any(name.startswith("physical_collision_") for name in batched_result.metrics)


def test_incompatible_lifecycle_metadata_falls_back_to_serial_anchor_rollouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _validation_config()
    batch = collate_episodes([generate_episode(config, seed=100003)])
    torch.manual_seed(29)
    serial_model = OnlineWorldModel.from_config(config, device="cpu").eval()
    batched_model = OnlineWorldModel.from_config(config, device="cpu").eval()
    batched_model.load_state_dict(serial_model.state_dict())

    def alternate_anchor_metadata(model: OnlineWorldModel) -> None:
        original_ingest = model.ingest

        def ingest(packet: Any, *, prepared: Any = None) -> Any:
            belief = original_ingest(packet, prepared=prepared)
            frame_index = int(packet.metadata["training_frame_index"])
            belief = belief.replace(metadata={"initialised": frame_index in {1, 9}})
            model.state.belief = belief
            return belief

        monkeypatch.setattr(model, "ingest", ingest)

    alternate_anchor_metadata(serial_model)
    alternate_anchor_metadata(batched_model)
    with torch.no_grad():
        serial_result = run_closed_loop_batch(
            serial_model,
            batch,
            config,
            window_steps=16,
            apply_perturbations=False,
            include_measurement_supervision=False,
            rollout_anchors_per_window=4,
            validation_rollout_anchor_batch_size=1,
            compute_future_correction=False,
        )
        batched_result = run_closed_loop_batch(
            batched_model,
            batch,
            config,
            window_steps=16,
            apply_perturbations=False,
            include_measurement_supervision=False,
            rollout_anchors_per_window=4,
            validation_rollout_anchor_batch_size=4,
            compute_future_correction=False,
        )

    _assert_result_close(batched_result, serial_result)
    _assert_nested_close(batched_model.state, serial_model.state)
    assert batched_result.metrics["rollout_execution_batch_requested_anchor_count"] == 4.0
    assert batched_result.metrics["rollout_execution_batched_anchor_count"] == 0.0
    assert batched_result.metrics["rollout_execution_serial_fallback_anchor_count"] == 4.0
    assert batched_result.metrics["rollout_execution_posterior_call_count"] == 4.0


def test_rollout_metadata_comparison_handles_tensor_values_without_ambiguous_truth() -> None:
    assert _rollout_metadata_equal(
        {"initialised": True, "row": torch.tensor([1, 2])},
        {"initialised": True, "row": torch.tensor([1, 2])},
    )
    assert not _rollout_metadata_equal(
        {"initialised": True, "row": torch.tensor([1, 2])},
        {"initialised": False, "row": torch.tensor([1, 3])},
    )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("train", "model.eval"),
        ("grad", "torch.no_grad"),
        ("episode_batch", "loader batch size one"),
        ("perturbed", "posterior-only"),
        ("future_correction", "posterior-only"),
        ("multi_rate", "multi-rate"),
    ],
)
def test_validation_rollout_batching_rejects_non_parity_execution(
    case: str,
    message: str,
) -> None:
    config = _validation_config()
    model = OnlineWorldModel.from_config(config, device="cpu")
    batch_size = 2 if case == "episode_batch" else 1
    batch = {"rgb": torch.zeros((batch_size, 16, 3, 24, 24))}
    if case != "train":
        model.eval()
    selected_config = (
        replace(
            config,
            model=replace(
                config.model,
                dynamics=replace(
                    config.model.dynamics,
                    learned_effect_interval_seconds=config.model.dynamics.max_substep,
                ),
            ),
        )
        if case == "multi_rate"
        else config
    )
    kwargs = {
        "apply_perturbations": case == "perturbed",
        "compute_future_correction": case == "future_correction",
        "validation_rollout_anchor_batch_size": 4,
    }
    context = torch.enable_grad() if case == "grad" else torch.no_grad()
    with context, pytest.raises(ValueError, match=message):
        run_closed_loop_batch(model, batch, selected_config, **kwargs)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_validation_rollout_anchor_batch_size_must_be_a_positive_integer(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="validation_rollout_anchor_batch_size"):
        load_config(
            "configs/tiny_overfit.yaml",
            overrides=[f"training.validation_rollout_anchor_batch_size={str(value).lower()}"],
        )


def test_anchor_batch_size_is_checkpoint_and_validation_protocol_semantics() -> None:
    serial = load_config("configs/tiny_overfit.yaml")
    batched = replace(
        serial,
        training=replace(serial.training, validation_rollout_anchor_batch_size=8),
    )
    serial_protocol = _rollout_validation_protocol(serial)
    batched_protocol = _rollout_validation_protocol(batched)

    assert serial_protocol["execution"]["validation_batch_size"] == 1
    assert serial_protocol["execution"]["validation_rollout_anchor_batch_size"] == 1
    assert batched_protocol["execution"]["validation_rollout_anchor_batch_size"] == 8
    assert _rollout_validation_protocol_hash(serial) != _rollout_validation_protocol_hash(batched)

    legacy_config = serial.to_dict()
    del legacy_config["training"]["validation_rollout_anchor_batch_size"]
    payload = {
        "simulator_version": SIMULATOR_VERSION,
        "config": legacy_config,
    }
    validate_training_resume_config(payload, serial)
    with pytest.raises(ValueError, match="validation_rollout_anchor_batch_size"):
        validate_training_resume_config(payload, batched)
