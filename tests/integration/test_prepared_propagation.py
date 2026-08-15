from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from typing import Any

import pytest
import torch
from torch import Tensor

from world_model.observations import ObservationPacket
from world_model.runtime import (
    OnlineWorldModel,
    PreparedPropagationError,
)
from world_model.utils.config import OrpheusConfig


def _rgb_config() -> OrpheusConfig:
    config = OrpheusConfig()
    return replace(
        config,
        simulator=replace(
            config.simulator,
            image_size=(32, 32),
            min_objects=1,
            max_objects=2,
        ),
        model=replace(
            config.model,
            max_objects=2,
            state=replace(
                config.model.state,
                geometry_dim=2,
                appearance_dim=8,
                residual_dynamics_dim=4,
                modal_count=1,
                modal_dim=2,
                parameter_memory_dim=16,
                global_dim=4,
            ),
            rgb=replace(
                config.model.rgb,
                backbone_channels=(8, 16, 24, 32),
                feature_dim=16,
                proposal_queries=3,
                roi_size=8,
                global_every_steps=5,
                global_uncertainty_threshold=1.0e6,
                surprise_threshold=1.0e6,
            ),
            dynamics=replace(config.model.dynamics, hidden_dim=24),
            filter=replace(config.model.filter, hidden_dim=32),
            identification=replace(config.model.identification, hidden_dim=16),
            lifecycle=replace(
                config.model.lifecycle,
                birth_confidence=0.0,
                birth_confirmations=1,
            ),
        ),
    )


def _rgb_hypothesis_config() -> OrpheusConfig:
    config = _rgb_config()
    return replace(
        config,
        runtime=replace(
            config.runtime,
            hypothesis_pool_enabled=True,
            hypothesis_evidence_horizons_seconds=(1.0 / 30.0,),
            hypothesis_axis_independent_axes=(0,),
        ),
    )


def _rgb_packet(timestamp: float, shift: int = 0) -> ObservationPacket:
    image = torch.zeros(3, 32, 32)
    image[0, 10:17, 8 + shift : 15 + shift] = 1.0
    intrinsics = torch.tensor([[30.0, 0.0, 15.5], [0.0, 30.0, 15.5], [0.0, 0.0, 1.0]])
    world_from_camera = torch.eye(4)
    world_from_camera[2, 3] = -4.0
    return ObservationPacket(
        modality="rgb",
        sensor_id="camera",
        timestamp=timestamp,
        payload=image,
        calibration={
            "intrinsics": intrinsics,
            "world_from_camera": world_from_camera,
        },
        frame_id="camera:camera",
    )


def _oracle_config() -> OrpheusConfig:
    config = OrpheusConfig()
    return replace(
        config,
        runtime=replace(
            config.runtime,
            modality="debug_oracle",
            enable_debug_oracle=True,
        ),
        model=replace(
            config.model,
            lifecycle=replace(
                config.model.lifecycle,
                birth_confidence=0.0,
                birth_confirmations=1,
            ),
        ),
        evaluation=replace(config.evaluation, rgb_only=False),
    )


def _oracle_packet(
    timestamp: float,
    positions: Tensor,
    velocities: Tensor | None = None,
) -> ObservationPacket:
    if velocities is None:
        velocities = torch.zeros_like(positions)
    object_count = positions.shape[-2]
    return ObservationPacket(
        modality="debug_oracle",
        sensor_id="state",
        timestamp=timestamp,
        payload={
            "position": positions,
            "velocity": velocities,
            "active": torch.ones(object_count, dtype=torch.bool),
            "id": torch.arange(50, 50 + object_count),
            "radius": torch.full((object_count, 1), 0.1),
        },
        calibration={},
        frame_id="world",
    )


def _assert_nested_close(actual: Any, expected: Any) -> None:
    if isinstance(actual, Tensor) or isinstance(expected, Tensor):
        assert isinstance(actual, Tensor) and isinstance(expected, Tensor)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0, equal_nan=True)
        return
    if is_dataclass(actual) or is_dataclass(expected):
        assert type(actual) is type(expected)
        for item in fields(actual):
            if item.name == "elapsed_milliseconds":
                continue
            _assert_nested_close(
                getattr(actual, item.name),
                getattr(expected, item.name),
            )
        return
    if isinstance(actual, dict) or isinstance(expected, dict):
        assert isinstance(actual, dict) and isinstance(expected, dict)
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_nested_close(actual[key], expected[key])
        return
    if isinstance(actual, (tuple, list)) or isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_nested_close(actual_item, expected_item)
        return
    assert actual == expected


def test_prepared_ingest_matches_normal_runtime_state_and_diagnostics() -> None:
    torch.manual_seed(17)
    normal = OnlineWorldModel.from_config(_rgb_config(), device="cpu")
    prepared_runtime = OnlineWorldModel.from_config(_rgb_config(), device="cpu")
    prepared_runtime.load_state_dict(normal.state_dict())

    normal.ingest(_rgb_packet(0.0))
    prepared_runtime.ingest(_rgb_packet(0.0))
    propagation = prepared_runtime.prepare_propagation(1.0 / 30.0)

    normal_posterior = normal.ingest(_rgb_packet(1.0 / 30.0, shift=1))
    prepared_posterior = prepared_runtime.ingest(
        _rgb_packet(1.0 / 30.0, shift=1),
        prepared=propagation,
    )

    assert propagation.consumed
    _assert_nested_close(prepared_posterior, normal_posterior)
    _assert_nested_close(prepared_runtime.state, normal.state)
    _assert_nested_close(prepared_runtime.last_measurements, normal.last_measurements)
    _assert_nested_close(
        prepared_runtime.scheduler.state_for("camera"),
        normal.scheduler.state_for("camera"),
    )
    assert len(prepared_runtime.diagnostics.records) == len(normal.diagnostics.records)
    for prepared_record, normal_record in zip(
        prepared_runtime.diagnostics.records,
        normal.diagnostics.records,
        strict=True,
    ):
        _assert_nested_close(prepared_record, normal_record)


def test_runtime_pool_reuses_scheduled_learned_step_for_next_prepared_ingest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = OnlineWorldModel.from_config(_rgb_hypothesis_config(), device="cpu")
    model.eval()
    calls = 0
    original_predict_step = model.dynamics.predict_step

    def recording_predict_step(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return original_predict_step(*args, **kwargs)

    monkeypatch.setattr(model.dynamics, "predict_step", recording_predict_step)
    model.ingest(_rgb_packet(0.0))
    # Initialization performs one zero-delta prior plus the first delayed-
    # evidence forecast.
    assert calls == 2

    prepared = model.prepare_propagation(1.0 / 30.0)

    assert calls == 2
    model.ingest(_rgb_packet(1.0 / 30.0, shift=1), prepared=prepared)
    assert calls == 3  # only the next delayed-evidence forecast was added
    model.ingest(_rgb_packet(2.0 / 30.0, shift=2))
    assert calls == 4  # ordinary ingest reuses the due forecast as well


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("source", "source_tensor_revision"),
        ("result", "scheduled_result_revision"),
        ("weight", "dynamics_revision"),
        ("mode", "dynamics_mode"),
    ],
)
def test_runtime_pool_rejects_stale_scheduled_propagation(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    reason: str,
) -> None:
    model = OnlineWorldModel.from_config(_rgb_hypothesis_config(), device="cpu")
    model.eval()
    calls = 0
    original_predict_step = model.dynamics.predict_step

    def recording_predict_step(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return original_predict_step(*args, **kwargs)

    monkeypatch.setattr(model.dynamics, "predict_step", recording_predict_step)
    model.ingest(_rgb_packet(0.0))
    assert calls == 2
    assert model.hypothesis_controller is not None
    pending = model.hypothesis_controller.pending[0]
    if mutation == "source":
        assert model.belief is not None
        model.belief.objects.position.add_(0.01)
    elif mutation == "result":
        pending.learned_step.belief.objects.position.add_(0.01)
    elif mutation == "weight":
        with torch.no_grad():
            next(model.dynamics.parameters()).add_(0.01)
    else:
        model.train()

    prepared = model.prepare_propagation(1.0 / 30.0)

    assert calls == 3
    assert torch.isfinite(prepared.prior.objects.position).all()
    assert model.hypothesis_controller.pending_invalidation_counts[reason] >= 1


def test_prepared_propagation_rejects_wrong_target_stale_revision_and_reuse() -> None:
    model = OnlineWorldModel.from_config(_oracle_config(), device="cpu")
    model.ingest(_oracle_packet(0.0, torch.tensor([[0.0, 1.0, 0.0]])))
    propagation = model.prepare_propagation(0.1)

    with pytest.raises(PreparedPropagationError, match="target timestamp"):
        model.ingest(
            _oracle_packet(0.2, torch.tensor([[0.0, 1.0, 0.0]])),
            prepared=propagation,
        )
    assert not propagation.consumed

    model.state.ingest_count += 1
    with pytest.raises(PreparedPropagationError, match="revision is stale"):
        model.ingest(
            _oracle_packet(0.1, torch.tensor([[0.0, 1.0, 0.0]])),
            prepared=propagation,
        )
    assert not propagation.consumed
    model.state.ingest_count -= 1

    model.ingest(
        _oracle_packet(0.1, torch.tensor([[0.0, 1.0, 0.0]])),
        prepared=propagation,
    )
    assert propagation.consumed
    with pytest.raises(PreparedPropagationError, match="already been consumed"):
        model.ingest(
            _oracle_packet(0.1, torch.tensor([[0.0, 1.0, 0.0]])),
            prepared=propagation,
        )


def test_prepared_propagation_rejects_wrong_runtime_and_source_identity() -> None:
    first = OnlineWorldModel.from_config(_oracle_config(), device="cpu")
    second = OnlineWorldModel.from_config(_oracle_config(), device="cpu")
    packet = _oracle_packet(0.0, torch.tensor([[0.0, 1.0, 0.0]]))
    first.ingest(packet)
    second.ingest(packet)
    propagation = first.prepare_propagation(0.1)

    with pytest.raises(PreparedPropagationError, match="another runtime"):
        second.ingest(
            _oracle_packet(0.1, torch.tensor([[0.0, 1.0, 0.0]])),
            prepared=propagation,
        )

    assert first.belief is not None
    first.state.belief = first.belief.clone()
    with pytest.raises(PreparedPropagationError, match="source belief is stale"):
        first.ingest(
            _oracle_packet(0.1, torch.tensor([[0.0, 1.0, 0.0]])),
            prepared=propagation,
        )


def test_prepared_propagation_rejects_source_time_device_and_batch_changes() -> None:
    model = OnlineWorldModel.from_config(_oracle_config(), device="cpu")
    model.ingest(_oracle_packet(0.0, torch.tensor([[0.0, 1.0, 0.0]])))
    propagation = model.prepare_propagation(0.1)
    assert model.belief is not None

    model.belief.timestamp.add_(0.01)
    with pytest.raises(PreparedPropagationError, match="source timestamp has changed"):
        model.ingest(
            _oracle_packet(0.1, torch.tensor([[0.0, 1.0, 0.0]])),
            prepared=propagation,
        )
    model.belief.timestamp.sub_(0.01)

    wrong_device = replace(propagation, source_device=torch.device("meta"))
    with pytest.raises(PreparedPropagationError, match="device/dtype"):
        model.ingest(
            _oracle_packet(0.1, torch.tensor([[0.0, 1.0, 0.0]])),
            prepared=wrong_device,
        )

    batched_packet = ObservationPacket(
        modality="debug_oracle",
        sensor_id="state",
        timestamp=0.1,
        payload={
            "position": torch.tensor([[[0.0, 1.0, 0.0]], [[0.1, 1.0, 0.0]]]),
            "velocity": torch.zeros(2, 1, 3),
            "active": torch.ones(2, 1, dtype=torch.bool),
            "id": torch.tensor([[50], [50]]),
            "radius": torch.full((2, 1, 1), 0.1),
        },
        calibration={},
        frame_id="world",
    )
    with pytest.raises(PreparedPropagationError, match="batch"):
        model.ingest(batched_packet, prepared=propagation)
    assert not propagation.consumed


def test_prepared_propagation_rejects_source_tensor_mutation_and_replacement() -> None:
    model = OnlineWorldModel.from_config(_oracle_config(), device="cpu")
    model.ingest(_oracle_packet(0.0, torch.tensor([[0.0, 1.0, 0.0]])))
    propagation = model.prepare_propagation(0.1)
    assert model.belief is not None

    model.belief.objects.velocity.add_(0.25)
    with pytest.raises(PreparedPropagationError, match="source belief tensors have changed"):
        model.ingest(
            _oracle_packet(0.1, torch.tensor([[0.0, 1.0, 0.0]])),
            prepared=propagation,
        )
    assert not propagation.consumed

    replacement_model = OnlineWorldModel.from_config(_oracle_config(), device="cpu")
    replacement_model.ingest(_oracle_packet(0.0, torch.tensor([[0.0, 1.0, 0.0]])))
    replacement = replacement_model.prepare_propagation(0.1)
    assert replacement_model.belief is not None
    replacement_model.belief.gravity = replacement_model.belief.gravity.clone()
    with pytest.raises(PreparedPropagationError, match="source belief tensors have changed"):
        replacement_model.ingest(
            _oracle_packet(0.1, torch.tensor([[0.0, 1.0, 0.0]])),
            prepared=replacement,
        )
    assert not replacement.consumed


def test_prepared_propagation_rejects_prior_tensor_mutation() -> None:
    model = OnlineWorldModel.from_config(_oracle_config(), device="cpu")
    model.ingest(_oracle_packet(0.0, torch.tensor([[0.0, 1.0, 0.0]])))
    propagation = model.prepare_propagation(0.1)

    propagation.prior.objects.position.add_(0.25)
    with pytest.raises(PreparedPropagationError, match="result tensors have changed"):
        model.ingest(
            _oracle_packet(0.1, torch.tensor([[0.0, 1.0, 0.0]])),
            prepared=propagation,
        )
    assert not propagation.consumed


def test_prepared_propagation_rejects_prior_graph_metadata_mutation() -> None:
    model = OnlineWorldModel.from_config(_oracle_config(), device="cpu")
    model.ingest(_oracle_packet(0.0, torch.tensor([[0.0, 1.0, 0.0]])))
    propagation = model.prepare_propagation(0.1)
    prior_position = propagation.prior.objects.position
    assert prior_position.requires_grad
    assert not prior_position.is_leaf
    original_version = prior_position._version

    prior_position.detach_()
    assert prior_position._version == original_version
    with pytest.raises(PreparedPropagationError, match="result tensors have changed"):
        model.ingest(
            _oracle_packet(0.1, torch.tensor([[0.0, 1.0, 0.0]])),
            prepared=propagation,
        )
    assert not propagation.consumed


def test_prepared_propagation_rejects_nonuniform_batch_target_timestamp() -> None:
    model = OnlineWorldModel.from_config(_oracle_config(), device="cpu")
    model.reset(batch_size=2)
    model.state.belief = model.belief_factory.create(
        batch_size=2,
        timestamp=torch.zeros(2),
        device="cpu",
        dtype=torch.float32,
        active_modalities=("debug_oracle",),
    )

    with pytest.raises(PreparedPropagationError, match="uniform across batch"):
        model.prepare_propagation(torch.tensor([0.1, 0.2]))


def test_prepared_propagation_rejects_dynamics_weight_and_mode_changes() -> None:
    model = OnlineWorldModel.from_config(_oracle_config(), device="cpu")
    model.ingest(_oracle_packet(0.0, torch.tensor([[0.0, 1.0, 0.0]])))
    propagation = model.prepare_propagation(0.1)
    parameter = next(model.dynamics.parameters())

    with torch.no_grad():
        parameter.add_(0.01)
    with pytest.raises(
        PreparedPropagationError,
        match="dynamics parameters or buffers have changed",
    ):
        model.ingest(
            _oracle_packet(0.1, torch.tensor([[0.0, 1.0, 0.0]])),
            prepared=propagation,
        )
    assert not propagation.consumed

    mode_model = OnlineWorldModel.from_config(_oracle_config(), device="cpu")
    mode_model.ingest(_oracle_packet(0.0, torch.tensor([[0.0, 1.0, 0.0]])))
    mode_propagation = mode_model.prepare_propagation(0.1)
    mode_model.dynamics.eval()
    with pytest.raises(
        PreparedPropagationError,
        match="training/evaluation mode has changed",
    ):
        mode_model.ingest(
            _oracle_packet(0.1, torch.tensor([[0.0, 1.0, 0.0]])),
            prepared=mode_propagation,
        )
    assert not mode_propagation.consumed


def test_prepared_ingest_retains_interval_collision_and_real_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = OnlineWorldModel.from_config(_oracle_config(), device="cpu")
    model.ingest(
        _oracle_packet(
            0.0,
            torch.tensor([[-0.15, 1.0, 0.0], [0.15, 1.0, 0.0]]),
            torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
        )
    )
    assert model.belief is not None
    objects = model.belief.objects.clone()
    objects.position[0, :2] = torch.tensor([[-0.15, 1.0, 0.0], [0.15, 1.0, 0.0]])
    objects.velocity[0, :2] = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    objects.geometry[0, :2, 0] = 0.1
    objects.log_drag[0, :2].fill_(-16.0)
    objects.fast_log_variance[0, :2].fill_(-10.0)
    model.state.belief = model.belief.replace(
        objects=objects,
        gravity=torch.zeros_like(model.belief.gravity),
    )
    propagation = model.prepare_propagation(0.06)
    assert "pair_collision" in propagation.auxiliary
    assert propagation.interval_collision_mask is not None
    assert propagation.interval_collision_mask[0, :2].all()
    propagation.interval_collision_mask.logical_not_()
    with pytest.raises(PreparedPropagationError, match="result tensors have changed"):
        model.ingest(
            _oracle_packet(
                0.06,
                torch.tensor([[-0.107, 1.0, 0.0], [0.107, 1.0, 0.0]]),
                torch.tensor([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]]),
            ),
            prepared=propagation,
        )
    assert not propagation.consumed
    propagation = model.prepare_propagation(0.06)
    assert propagation.interval_collision_mask is not None

    observed_deltas: list[Tensor] = []
    original_correct = model.updater.correct

    def recording_correct(*args: Any, **kwargs: Any) -> Any:
        observed_deltas.append(kwargs["dt"].detach().clone())
        return original_correct(*args, **kwargs)

    monkeypatch.setattr(model.updater, "correct", recording_correct)
    model.ingest(
        _oracle_packet(
            0.06,
            torch.tensor([[-0.107, 1.0, 0.0], [0.107, 1.0, 0.0]]),
            torch.tensor([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]]),
        ),
        prepared=propagation,
    )

    assert len(observed_deltas) == 1
    torch.testing.assert_close(observed_deltas[0], torch.tensor([0.06]))
    assert model.last_measurements is not None
    retained = model.last_measurements.auxiliary["prior_interval_collision_mask"]
    torch.testing.assert_close(retained, propagation.interval_collision_mask)
