"""Active-Aqua MPS qualification for the complete attention-pilot loop."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from world_model.datasets import collate_episodes
from world_model.observations import ObservationPacket
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


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="requires an active Aqua MPS device",
)
def test_opt_in_rgb_hypothesis_pool_is_finite_on_mps() -> None:
    """Exercise the stateful RGB-only delayed-evidence adapter on MPS."""

    source = load_config("configs/attention_pilot_mps.yaml")
    config = replace(
        source,
        runtime=replace(
            source.runtime,
            hypothesis_pool_enabled=True,
            hypothesis_evidence_horizons_seconds=(0.05,),
            hypothesis_axis_independent_axes=(0,),
            hypothesis_local_applicability_enabled=True,
            hypothesis_composition_step_seconds=0.05,
        ),
    )
    config.validate()
    model = OnlineWorldModel.from_config(config, device="mps")
    intrinsics = torch.tensor([[56.0, 0.0, 31.5], [0.0, 56.0, 31.5], [0.0, 0.0, 1.0]])
    world_from_camera = torch.eye(4)
    world_from_camera[2, 3] = -4.0
    for frame in range(4):
        image = torch.zeros(3, 64, 64)
        image[0, 14:27, 10 + frame : 23 + frame] = 1.0
        image[1, 35:47, 39 - frame : 51 - frame] = 0.8
        model.ingest(
            ObservationPacket(
                modality="rgb",
                sensor_id="camera",
                timestamp=frame * 0.05,
                payload=image,
                calibration={
                    "intrinsics": intrinsics,
                    "world_from_camera": world_from_camera,
                },
                frame_id=f"camera:{frame}",
            )
        )
    assert model.hypothesis_controller is not None
    assert model.hypothesis_controller.pool.last_selection is not None
    future = model.predict([0.1])
    assert torch.isfinite(future.positions).all()
    choices = future.auxiliary["hypothesis_axis_index"]
    supported = future.auxiliary["hypothesis_axis_supported"]
    candidate_steps = future.auxiliary["hypothesis_composed_candidate_step_count"]
    total_steps = future.auxiliary["hypothesis_composed_total_step_count"]
    regime_steps = future.auxiliary["hypothesis_composed_regime_step_count"]
    regimes = future.auxiliary["hypothesis_interaction_regime"]
    object_count = model.state.belief.objects.max_objects
    assert choices.shape == supported.shape == total_steps.shape == (1, 1, object_count, 3)
    assert candidate_steps.shape == (1, 1, object_count, 3, 4)
    assert regime_steps.shape == (1, 1, object_count, 6)
    assert regimes.shape == (1, 1, object_count)
    assert torch.equal(candidate_steps.sum(dim=-1), total_steps)
    assert torch.equal(regime_steps.sum(dim=-1), total_steps[..., 0])
