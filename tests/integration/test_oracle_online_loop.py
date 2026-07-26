from __future__ import annotations

from dataclasses import replace

import torch

from world_model.observations import ObservationPacket
from world_model.runtime import OnlineWorldModel
from world_model.utils.config import OrpheusConfig


def _oracle_config() -> OrpheusConfig:
    config = OrpheusConfig()
    return replace(
        config,
        runtime=replace(
            config.runtime,
            modality="debug_oracle",
            enable_debug_oracle=True,
        ),
        evaluation=replace(config.evaluation, rgb_only=False),
    )


def _packet(timestamp: float, x_position: float) -> ObservationPacket:
    return ObservationPacket(
        modality="debug_oracle",
        sensor_id="state",
        timestamp=timestamp,
        payload={
            "position": torch.tensor([[x_position, 1.0, 3.0]]),
            "velocity": torch.tensor([[0.0, 0.0, 0.0]]),
            "active": torch.tensor([True]),
            "id": torch.tensor([50]),
        },
        calibration={},
        frame_id="world",
    )


def test_oracle_debug_loop_corrects_perturbation_and_keeps_persistent_id() -> None:
    model = OnlineWorldModel.from_config(_oracle_config(), device="cpu")
    initial = model.ingest(_packet(0.0, 0.0))
    assert initial.objects.active.sum() == 1
    persistent_id = initial.objects.object_id[0, 0].item()
    perturbed_objects = initial.objects.replace(
        position=initial.objects.position
        + torch.tensor([[[0.8, 0.0, 0.0]]]).expand_as(initial.objects.position)
        * initial.objects.active.unsqueeze(-1),
        fast_log_variance=torch.where(
            initial.objects.active.unsqueeze(-1)
            & (torch.arange(initial.objects.fast_state_dim).reshape(1, 1, -1) < 3),
            torch.full_like(initial.objects.fast_log_variance, -1.0),
            initial.objects.fast_log_variance,
        ),
    )
    model.state.belief = initial.replace(objects=perturbed_objects)
    before_error = model.belief.objects.position[0, 0, 0].abs()
    posterior = model.ingest(_packet(1.0 / 30.0, 0.0))
    after_error = posterior.objects.position[0, 0, 0].abs()
    assert after_error < before_error
    assert posterior.objects.object_id[0, 0].item() == persistent_id
    assert model.diagnostics.oracle_used
    trajectory = model.predict([0.1, 0.2])
    assert trajectory.positions.shape[:2] == (1, 2)


def test_online_loop_rejects_delayed_oracle_packet() -> None:
    model = OnlineWorldModel.from_config(_oracle_config(), device="cpu")
    model.ingest(_packet(1.0, 0.0))
    try:
        model.ingest(_packet(0.5, 0.0))
    except ValueError as error:
        assert "precedes current belief" in str(error)
    else:
        raise AssertionError("delayed observation should be rejected")
