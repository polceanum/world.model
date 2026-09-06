"""Executable supervision bridge checks for the 1.61 dynamic-set campaign."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import world_model.training.dynamic_set_adapter as adapter_implementation
from world_model.runtime import OnlineWorldModel
from world_model.training.dynamic_set_adapter import DynamicSetEpisodeObjectiveAdapter
from world_model.training.dynamic_set_config import load_config
from world_model.training.dynamic_set_materializer import materialize_dynamic_set_episode
from world_model.training.dynamic_set_objectives import dynamic_set_objective
from world_model.training.dynamic_set_protocol import physical_manifest
from world_model.training.dynamic_set_trainer import DynamicSetTrainingMicrobatch

CONFIG_PATH = Path(__file__).parents[2] / "configs" / "rgbd_dynamic_set_planning_cpu.yaml"


def _four_regimes() -> tuple[object, ...]:
    selected: list[object] = []
    seen: set[tuple[bool, bool]] = set()
    for row in physical_manifest("training"):
        key = (row.contact, row.known_action)
        if key not in seen:
            selected.append(row)
            seen.add(key)
        if len(selected) == 4:
            break
    # Contact origin makes (contact=False, known_action=True) and both contact
    # origins real; there are exactly four physical/action regimes.
    assert len(selected) == 4
    return tuple(selected)


@pytest.fixture(scope="module")
def objective_case() -> tuple[OnlineWorldModel, DynamicSetTrainingMicrobatch]:
    rows = _four_regimes()
    materializations = tuple(materialize_dynamic_set_episode(row) for row in rows)
    model = OnlineWorldModel.from_config(load_config(CONFIG_PATH), device="cpu")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    proposer = model.observation_modules["rgbd"].set_proposer
    assert proposer is not None
    for parameter in proposer.parameters():
        parameter.requires_grad_(True)
    for parameter in model.dynamics.interactions.edge_network.parameters():
        parameter.requires_grad_(True)
    microbatch = DynamicSetTrainingMicrobatch(
        update_index=0,
        microbatch_index=0,
        dataset_indices=(0, 1, 2, 3),
        rows=rows,
        materializations=materializations,
    )
    return model, microbatch


def test_real_episode_adapter_builds_only_declared_physical_objectives(
    objective_case: tuple[OnlineWorldModel, DynamicSetTrainingMicrobatch],
) -> None:
    model, microbatch = objective_case
    adapter = DynamicSetEpisodeObjectiveAdapter()
    inputs = adapter.build_objective_inputs(model, microbatch)

    assert inputs.perception.mask_logits.shape == (4, 9, 64, 64)
    assert inputs.perception.target_exists.shape == (4, 8)
    torch.testing.assert_close(
        inputs.perception.target_masks.sum(dim=1),
        torch.ones((4, 64, 64)),
        rtol=0.0,
        atol=0.0,
    )
    assert inputs.dynamics.predicted_state.shape == (4, 7, 6, 6)
    assert inputs.dynamics.collision_logits.shape == (4, 7, 6, 6)
    assert inputs.dynamics.relation_residual.shape == (4, 7, 6, 12)
    assert inputs.causal_support.externally_actuated.shape == (4, 56, 6)
    assert inputs.causal_support.target_frame_index.shape == (4, 7)
    assert bool(
        (
            inputs.causal_support.target_frame_index[:, 1:]
            > inputs.causal_support.target_frame_index[:, :-1]
        ).all()
    )
    assert torch.equal(
        inputs.causal_support.known_action_observed,
        torch.stack(
            [
                item.episode["events"]["known_action_observed"]
                for item in microbatch.materializations
            ]
        ),
    )

    losses = dynamic_set_objective(inputs.perception, inputs.dynamics)
    assert torch.isfinite(losses.total)
    assert losses.total.requires_grad
    assert not hasattr(losses, "winner")
    assert not hasattr(losses, "regret")
    assert not hasattr(losses, "task_success")


def test_real_episode_adapter_reaches_every_and_only_allowed_gradient_owner(
    objective_case: tuple[OnlineWorldModel, DynamicSetTrainingMicrobatch],
) -> None:
    model, microbatch = objective_case
    model.zero_grad(set_to_none=True)
    inputs = DynamicSetEpisodeObjectiveAdapter().build_objective_inputs(model, microbatch)
    dynamic_set_objective(inputs.perception, inputs.dynamics).total.backward()

    proposer = model.observation_modules["rgbd"].set_proposer
    assert proposer is not None
    perception_parameters = tuple(proposer.parameters())
    relation_parameters = tuple(model.dynamics.interactions.edge_network.parameters())
    assert perception_parameters and relation_parameters
    assert all(parameter.grad is not None for parameter in perception_parameters)
    assert all(parameter.grad is not None for parameter in relation_parameters)
    assert all(
        bool(torch.isfinite(parameter.grad).all())
        for parameter in (*perception_parameters, *relation_parameters)
        if parameter.grad is not None
    )
    assert all(
        parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad
    )
    relation_output = model.dynamics.interactions.edge_network.output
    assert relation_output.weight.grad is not None
    # Collision confidence, bounded impulse multiplier/additive correction,
    # and process uncertainty are all live on the real four-regime batch.
    for output_index in (1, 4, 5, 6):
        assert relation_output.weight.grad[output_index].abs().sum() > 0.0


def test_fast_training_interval_preserves_analytic_collision_authority(
    objective_case: tuple[OnlineWorldModel, DynamicSetTrainingMicrobatch],
) -> None:
    model, microbatch = objective_case
    inputs = DynamicSetEpisodeObjectiveAdapter().build_objective_inputs(model, microbatch)
    supported = inputs.dynamics.collision_support
    predicted = inputs.dynamics.collision_logits > 0.0

    assert torch.equal(
        predicted & supported,
        inputs.dynamics.collision_target & supported,
    )


def test_natural_contact_supervision_spans_pre_contact_event_and_aftermath(
    objective_case: tuple[OnlineWorldModel, DynamicSetTrainingMicrobatch],
) -> None:
    model, microbatch = objective_case
    inputs = DynamicSetEpisodeObjectiveAdapter().build_objective_inputs(model, microbatch)
    row_index = next(
        index for index, row in enumerate(microbatch.rows) if row.contact and not row.known_action
    )
    episode = microbatch.materializations[row_index].episode
    collision_frame = int(
        torch.nonzero(
            episode["events"]["pair_collision"].any(dim=(-2, -1)),
            as_tuple=False,
        )[0]
    )
    targets = inputs.causal_support.target_frame_index[row_index]
    assert targets.tolist() == list(range(collision_frame - 2, collision_frame + 5))
    collision_step = int(torch.nonzero(targets == collision_frame, as_tuple=False)[0])
    assert collision_step == 2
    participants = inputs.dynamics.collision_target[row_index, collision_step].any(dim=-1)
    assert bool(participants.any())
    assert bool(inputs.dynamics.contact_window_mask[row_index, :, participants].all())


def test_action_induced_trajectory_applies_action_first_and_reaches_collision(
    objective_case: tuple[OnlineWorldModel, DynamicSetTrainingMicrobatch],
) -> None:
    model, _ = objective_case
    row = next(
        value
        for value in physical_manifest("training")
        if value.contact_origin == "action_induced" and value.action_time_stratum == 0
    )
    materialization = materialize_dynamic_set_episode(row)
    microbatch = DynamicSetTrainingMicrobatch(
        update_index=0,
        microbatch_index=0,
        dataset_indices=(0, 1, 2, 3),
        rows=(row, row, row, row),
        materializations=(materialization,) * 4,
    )
    episodes = tuple(
        adapter_implementation._episode(value) for value in microbatch.materializations
    )
    dynamics, support = adapter_implementation._dynamics_inputs(model, microbatch, episodes)
    action_frame = int(
        torch.nonzero(
            materialization.episode["events"]["known_action_observed"].any(dim=-1),
            as_tuple=False,
        )[0]
    )
    assert support.target_frame_index[0, 0].item() == action_frame
    collision_steps = torch.nonzero(
        dynamics.collision_target[0].any(dim=(-2, -1)),
        as_tuple=False,
    ).flatten()
    assert collision_steps.tolist() == [6]
    participants = dynamics.collision_target[0, 6].any(dim=-1)
    assert bool(dynamics.contact_window_mask[0, :, participants].all())
    assert bool((dynamics.collision_logits[0, 6][dynamics.collision_target[0, 6]] > 0.0).all())


def test_adapter_checkpoint_state_is_exact_and_rejects_schema_drift() -> None:
    adapter = DynamicSetEpisodeObjectiveAdapter()
    state = adapter.state_dict()
    adapter.load_state_dict(state)
    with pytest.raises(ValueError, match="state schema"):
        adapter.load_state_dict({**state, "cursor": 1})
    with pytest.raises(ValueError, match="checkpoint schema"):
        adapter.load_state_dict({"schema": "dynamic_set_episode_objective_adapter_v3"})


def test_batched_dynamics_matches_the_b1_serial_oracle(
    objective_case: tuple[OnlineWorldModel, DynamicSetTrainingMicrobatch],
) -> None:
    model, microbatch = objective_case
    episodes = tuple(
        adapter_implementation._episode(value) for value in microbatch.materializations
    )
    windows = tuple(
        adapter_implementation._select_dynamics_window(
            episode,
            known_action=row.known_action,
            contact=row.contact,
            selector=(microbatch.update_index * 24 + microbatch.microbatch_index * 4 + index),
        )
        for index, (row, episode) in enumerate(zip(microbatch.rows, episodes, strict=True))
    )
    serial = tuple(
        adapter_implementation._dynamics_row(
            model,
            episode,
            anchor=window[0],
            target=window[1],
            action_event=window[2],
        )
        for episode, window in zip(episodes, windows, strict=True)
    )

    batched, _ = adapter_implementation._dynamics_inputs(model, microbatch, episodes)

    expected = (
        torch.cat([row[0] for row in serial]),
        torch.cat([row[1] for row in serial]),
        torch.cat([row[2] for row in serial]),
        torch.cat([row[3] for row in serial]),
        torch.cat([row[4] for row in serial]),
    )
    actual = (
        batched.predicted_state,
        batched.target_state,
        batched.process_log_variance,
        batched.collision_logits,
        batched.relation_residual,
    )
    for batch_value, serial_value in zip(actual, expected, strict=True):
        torch.testing.assert_close(batch_value, serial_value, rtol=1.0e-6, atol=1.0e-7)
