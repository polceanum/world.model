"""Active-Aqua MPS qualification for the complete attention-pilot loop."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from world_model.datasets import collate_episodes
from world_model.runtime import OnlineWorldModel
from world_model.simulator import generate_episode
from world_model.training.loop import move_batch_to_device, run_closed_loop_batch
from world_model.training.trainer import set_closed_loop_trainable_scope
from world_model.utils.config import load_config


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="requires an active Aqua MPS device",
)
def test_attention_pilot_rgb_closed_loop_z_scope_is_finite_on_mps() -> None:
    """Exercise the actual RGB/filter/association/rollout graph on MPS.

    The one-episode, unbalanced override is a numerical smoke only. The
    production protocol remains scenario-balanced and must still pass its
    complete fixed-manifest selector before any model is promoted.
    """

    source = load_config("configs/attention_pilot_mps.yaml")
    config = replace(
        source,
        training=replace(
            source.training,
            batch_size=1,
            scenario_balanced_batches=False,
            tbptt_steps=4,
            rollout_anchors_per_window=1,
        ),
        evaluation=replace(source.evaluation, episodes=1),
    )
    config.validate()
    model = OnlineWorldModel.from_config(config, device="mps")
    set_closed_loop_trainable_scope(model, scope="attention_node_z")
    batch = move_batch_to_device(
        collate_episodes([generate_episode(config, seed=100000)]),
        "mps",
    )

    result = run_closed_loop_batch(
        model,
        batch,
        config,
        window_steps=4,
        apply_perturbations=False,
        include_measurement_supervision=True,
        rollout_anchors_per_window=1,
        compute_future_correction=False,
    )
    result.total_loss.backward()
    attention = model.dynamics.attention_interactions
    assert attention is not None
    z_gradient = attention.node_decoder.weight.grad[2]

    assert torch.isfinite(result.total_loss)
    assert z_gradient is not None
    assert torch.isfinite(z_gradient).all()
