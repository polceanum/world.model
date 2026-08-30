"""Focused contract tests for the known-action RGB-D profile."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from world_model.belief import MotionMode
from world_model.dynamics import AnalyticFreeMotionDynamics, WorldImpulseAction
from world_model.runtime import OnlineWorldModel
from world_model.training.checkpointing import checkpoint_payload
from world_model.utils.config import load_config

REPOSITORY_ROOT = Path(__file__).parents[2]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "rgbd_known_action_planning_cpu.yaml"
ACCEPTED_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "rgbd_two_visible_orbital_camera_cpu.yaml"


def _assert_trajectory_equal(actual: object, expected: object) -> None:
    tensor_fields = (
        "timestamps",
        "positions",
        "velocities",
        "orientations",
        "motion_mode_logits",
        "fast_log_variance",
        "active_mask",
        "event_logits",
    )
    for name in tensor_fields:
        actual_value = getattr(actual, name)
        expected_value = getattr(expected, name)
        if actual_value is None or expected_value is None:
            assert actual_value is expected_value
        else:
            assert torch.equal(actual_value, expected_value), name
    assert actual.auxiliary.keys() == expected.auxiliary.keys()
    for name in actual.auxiliary:
        assert torch.equal(actual.auxiliary[name], expected.auxiliary[name]), name


def _synthetic_runtime() -> OnlineWorldModel:
    config = load_config(CONFIG_PATH)
    model = OnlineWorldModel.from_config(config, device="cpu")
    belief = model.belief_factory.create(
        batch_size=1,
        timestamp=3.0,
        device="cpu",
        dtype=torch.float32,
        gravity=config.simulator.gravity,
    )
    objects = belief.objects.clone()
    objects.active.fill_(True)
    objects.object_id[0] = torch.tensor([11, 29], dtype=torch.int64)
    objects.position[0] = torch.tensor(
        [[-0.6, -0.1, 4.0], [0.65, 0.15, 4.2]],
        dtype=belief.dtype,
    )
    objects.velocity[0] = torch.tensor(
        [[0.04, 0.01, 0.0], [-0.03, 0.0, 0.01]],
        dtype=belief.dtype,
    )
    objects.log_mass.zero_()
    objects.log_drag.fill_(math.log(config.model.rgbd.linear_drag))
    objects.motion_mode_logits.fill_(-4.0)
    objects.motion_mode_logits[..., MotionMode.FREE] = 4.0
    model.state.belief = belief.replace(
        objects=objects,
        next_object_id=belief.next_object_id.new_tensor([30]),
    ).validate()
    return model


def test_profile_loads_and_exactly_inherits_the_accepted_runtime() -> None:
    config = load_config(CONFIG_PATH)
    accepted = load_config(ACCEPTED_CONFIG_PATH)

    assert config.project.name == "orpheus-rgbd-known-action-planning-v1-cpu"
    assert config.project.seed == 0
    assert config.project.output_dir == accepted.project.output_dir == "runs"
    assert config.project.deterministic is accepted.project.deterministic is True
    inherited = config.to_dict()
    accepted_inherited = accepted.to_dict()
    inherited.pop("project")
    accepted_inherited.pop("project")
    assert inherited == accepted_inherited

    assert config.simulator.min_objects == config.simulator.max_objects == 2
    assert config.simulator.radius_range == (0.21, 0.21)
    assert config.simulator.mass_range == (1.0, 1.0)
    assert config.simulator.drag_range == (0.05, 0.05)
    assert config.simulator.gravity == (0.0, 0.0, 0.0)
    assert config.simulator.camera_motion == "orbit"
    assert config.simulator.render_noise_std == 0.0
    assert config.simulator.external_impulse_probability == 0.0
    assert config.model.rgbd.enabled
    assert not config.model.rgb.enabled
    assert config.model.rgbd.proposal_count == 2
    assert config.model.rgbd.temporal_history_size == 16
    assert config.model.rgbd.temporal_min_samples == 16
    assert config.model.dynamics.analytic_free_motion_only
    assert not config.model.dynamics.attention_residual_enabled
    assert not config.model.filter.enable_learned_corrector
    assert not config.model.identification.enabled
    assert config.runtime.modality == "rgbd"
    assert not config.runtime.hypothesis_pool_enabled


def test_profile_keeps_model_and_checkpoint_state_empty() -> None:
    config = load_config(CONFIG_PATH)
    model = OnlineWorldModel.from_config(config, device="cpu")

    assert isinstance(model.dynamics, AnalyticFreeMotionDynamics)
    assert not tuple(model.parameters())
    assert not tuple(model.buffers())
    assert model.state_dict() == {}

    payload = checkpoint_payload(
        model=model,
        optimizer=None,
        scheduler=None,
        config=config,
        step=0,
        metrics={},
        device="cpu",
        source_provenance={
            "commit": "0" * 40,
            "dirty": False,
            "worktree_fingerprint": "1" * 64,
            "runtime_source_fingerprint": "2" * 64,
        },
    )
    assert payload["model_state"] == {}
    assert payload["optimizer_state"] is None
    assert payload["scheduler_state"] is None
    assert payload["config"] == config.to_dict()


def test_profile_exposes_synthetic_action_and_exact_none_parity() -> None:
    model = _synthetic_runtime()
    assert model.belief is not None
    source = model.belief.clone()
    query_times = [0.1, 0.25, 0.5]

    legacy = model.predict(query_times)
    explicit_none = model.predict(query_times, action=None)
    _assert_trajectory_equal(legacy, explicit_none)

    impulse = source.timestamp.new_tensor([[0.02, -0.01, 0.005]])
    action = WorldImpulseAction(
        timestamp=source.timestamp + 0.25,
        object_id=torch.tensor([29], dtype=torch.int64),
        impulse_world=impulse,
    )
    acted = model.predict(query_times, action=action)

    assert torch.equal(acted.positions[:, 0], legacy.positions[:, 0])
    assert torch.equal(acted.velocities[:, 0], legacy.velocities[:, 0])
    assert torch.equal(acted.positions[:, :, 0], legacy.positions[:, :, 0])
    assert torch.equal(acted.velocities[:, :, 0], legacy.velocities[:, :, 0])
    assert torch.equal(acted.positions[:, 1, 1], legacy.positions[:, 1, 1])
    torch.testing.assert_close(
        acted.velocities[:, 1, 1],
        legacy.velocities[:, 1, 1] + impulse,
        rtol=0.0,
        atol=0.0,
    )

    applied = acted.auxiliary["known_action_applied"]
    known_impulse = acted.auxiliary["known_impulse_world"]
    assert applied.dtype is torch.bool
    assert applied.shape == (1, 3, 2)
    assert applied.sum().item() == 1
    assert applied[0, 1, 1]
    assert known_impulse.shape == (1, 3, 2, 3)
    assert known_impulse.dtype == source.dtype
    assert known_impulse.device == source.device
    assert torch.equal(known_impulse[0, 1, 1], impulse[0])
    assert model.belief is not None
    assert torch.equal(model.belief.objects.position, source.objects.position)
    assert torch.equal(model.belief.objects.velocity, source.objects.velocity)
