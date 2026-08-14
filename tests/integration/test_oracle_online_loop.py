from __future__ import annotations

from dataclasses import replace

import torch

from world_model.belief import TentativeBirthState
from world_model.dynamics import HypothesisDynamicsPool, RolloutStep
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


def _pair_packet(
    timestamp: float,
    positions: torch.Tensor,
    velocities: torch.Tensor,
) -> ObservationPacket:
    return ObservationPacket(
        modality="debug_oracle",
        sensor_id="state",
        timestamp=timestamp,
        payload={
            "position": positions,
            "velocity": velocities,
            "active": torch.tensor([True, True]),
            "id": torch.tensor([50, 51]),
            "radius": torch.tensor([[0.1], [0.1]]),
        },
        calibration={},
        frame_id="world",
    )


class _FixedDynamics:
    def __init__(self, position: float) -> None:
        self.position = position

    def predict_step(self, belief, delta_time):
        objects = belief.objects.clone()
        objects.position[..., 0] = self.position
        return RolloutStep(
            belief=belief.replace(
                timestamp=belief.timestamp + delta_time,
                objects=objects,
            ),
            event_logits=torch.zeros(belief.batch_size, objects.max_objects, 2),
            auxiliary={},
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


def test_runtime_exposes_persistent_hypothesis_pool_without_replacing_belief() -> None:
    model = OnlineWorldModel.from_config(_oracle_config(), device="cpu")
    belief = model.ingest(_packet(0.0, 0.0))
    pool = HypothesisDynamicsPool([_FixedDynamics(1.0), _FixedDynamics(0.0)])
    trajectories = model.predict_hypotheses(pool, [0.1, 0.2])
    target_positions = torch.zeros_like(trajectories[0].positions)
    target_mask = torch.zeros_like(trajectories[0].active_mask)
    target_mask[:, :, 0] = True
    selection = model.assimilate_hypotheses(
        pool,
        target_positions,
        target_mask,
        trajectories,
        uncertainty_aware=False,
    )
    assert selection.selected_index.tolist() == [1]
    axis_selection = model.selected_hypothesis_axes(pool)
    assert axis_selection.shape == (1, 3)
    assert axis_selection.tolist() == [[1, 0, 0]]
    assert model.belief is not None
    torch.testing.assert_close(model.belief.objects.position, belief.objects.position)


def test_runtime_assimilates_robust_hypothesis_ensemble_without_replacing_belief() -> None:
    model = OnlineWorldModel.from_config(_oracle_config(), device="cpu")
    belief = model.ingest(_packet(0.0, 0.0))
    pool = HypothesisDynamicsPool([_FixedDynamics(0.0), _FixedDynamics(0.45)])
    first = model.predict_hypotheses(pool, [0.1])
    nearby = HypothesisDynamicsPool([_FixedDynamics(0.6), _FixedDynamics(0.45)])
    second = model.predict_hypotheses(nearby, [0.1])
    target_positions = torch.zeros_like(first[0].positions)
    target_mask = torch.zeros_like(first[0].active_mask)
    target_mask[:, :, 0] = True
    selection = model.assimilate_hypothesis_ensemble(
        pool,
        target_positions,
        target_mask,
        [first, second],
        risk_penalty=0.2,
        uncertainty_aware=False,
    )
    assert selection.sample_count == 2
    assert selection.selected_index.tolist() == [1]
    assert model.belief is not None
    torch.testing.assert_close(model.belief.objects.position, belief.objects.position)


def test_runtime_assigns_permanent_id_only_after_configured_birth_confirmations() -> None:
    config = _oracle_config()
    config = replace(
        config,
        model=replace(
            config.model,
            lifecycle=replace(
                config.model.lifecycle,
                birth_confirmations=2,
                birth_confirmation_distance_m=0.5,
            ),
        ),
    )
    model = OnlineWorldModel.from_config(config, device="cpu")

    tentative = model.ingest(_packet(0.0, 0.0))
    assert not tentative.objects.active.any()
    assert model.diagnostics.latest is not None
    assert model.diagnostics.latest.tentative_birth_candidates == 1
    assert model.diagnostics.latest.confirmed_births == 0
    assert model.state.tentative_births[("debug_oracle", "state")].confirmation_count[0, 0] == 1

    confirmed = model.ingest(_packet(0.1, 0.1))
    assert confirmed.objects.active.sum() == 1
    assert confirmed.objects.object_id[0, 0] == 0
    assert model.diagnostics.latest is not None
    assert model.diagnostics.latest.tentative_birth_candidates == 0
    assert model.diagnostics.latest.confirmed_births == 1
    assert ("debug_oracle", "state") not in model.state.tentative_births

    model.reset()
    assert not model.state.tentative_births


def test_tentative_birth_confirmation_is_independent_per_modality_and_sensor() -> None:
    config = _oracle_config()
    config = replace(
        config,
        model=replace(
            config.model,
            lifecycle=replace(
                config.model.lifecycle,
                birth_confirmations=2,
                birth_confirmation_distance_m=0.5,
            ),
        ),
    )
    model = OnlineWorldModel.from_config(config, device="cpu")
    model.state.tentative_births[("rgb", "state")] = TentativeBirthState(
        world_position=torch.tensor([[[0.0, 1.0, 3.0]]]),
        active=torch.tensor([[True]]),
        confirmation_count=torch.tensor([[1]], dtype=torch.int64),
        timestamp=torch.tensor([[-0.1]]),
    ).validate()

    posterior = model.ingest(_packet(0.0, 0.0))

    assert not posterior.objects.active.any()
    assert set(model.state.tentative_births) == {
        ("rgb", "state"),
        ("debug_oracle", "state"),
    }
    assert model.state.tentative_births[("debug_oracle", "state")].confirmation_count[0, 0] == 1
    assert model.state.tentative_births[("rgb", "state")].confirmation_count[0, 0] == 1


def test_online_loop_rejects_delayed_oracle_packet() -> None:
    model = OnlineWorldModel.from_config(_oracle_config(), device="cpu")
    model.ingest(_packet(1.0, 0.0))
    try:
        model.ingest(_packet(0.5, 0.0))
    except ValueError as error:
        assert "precedes current belief" in str(error)
    else:
        raise AssertionError("delayed observation should be rejected")


def test_runtime_preserves_collision_that_occurs_between_observation_times() -> None:
    model = OnlineWorldModel.from_config(_oracle_config(), device="cpu")
    model.ingest(
        _pair_packet(
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
    model.ingest(
        _pair_packet(
            0.06,
            torch.tensor([[-0.107, 1.0, 0.0], [0.107, 1.0, 0.0]]),
            torch.tensor([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]]),
        )
    )

    assert model.last_measurements is not None
    interval_collision = model.last_measurements.auxiliary["prior_interval_collision_mask"]
    assert interval_collision[0, :2].all()
    # Endpoint mode remains a state-at-time contract rather than being
    # overwritten with an event that occurred earlier in the interval.
    assert not (model.belief.objects.mode[0, :2] == 3).all()
