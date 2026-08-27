from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from world_model.datasets import SyntheticSphereDataset, collate_episodes
from world_model.runtime import OnlineWorldModel
from world_model.training.checkpointing import (
    checkpoint_payload,
    load_checkpoint,
    load_model_weights,
    save_checkpoint,
    validate_checkpoint_config,
    validate_training_resume_config,
)
from world_model.training.loop import (
    move_batch_to_device,
    pretrain_rgb_measurements,
)
from world_model.training.trainer import train_from_config
from world_model.utils.config import load_config
from world_model.utils.device import select_device
from world_model.utils.version import SIMULATOR_VERSION, SPECIFICATION_VERSION


def test_checkpoint_specification_version_matches_authoritative_contract() -> None:
    specification = (Path(__file__).resolve().parents[2] / "PROJECT_SPEC.md").read_text(
        encoding="utf-8"
    )
    match = re.search(r"^\*\*Version:\*\* ([0-9.]+)$", specification, re.MULTILINE)

    assert match is not None
    assert match.group(1) == SPECIFICATION_VERSION


def _small_config():
    config = load_config("configs/tiny_overfit.yaml")
    return replace(
        config,
        simulator=replace(
            config.simulator,
            image_size=(32, 32),
            sequence_frames=3,
            min_objects=1,
            max_objects=1,
        ),
        training=replace(
            config.training,
            batch_size=1,
            train_episodes=1,
            validation_episodes=1,
            tbptt_steps=2,
            horizon_weights=(1.0,),
        ),
        evaluation=replace(
            config.evaluation,
            horizons_seconds=(0.05,),
            episodes=1,
        ),
    )


def test_checkpoint_roundtrip_preserves_trained_state(tmp_path):
    config = _small_config()
    dataset = SyntheticSphereDataset(
        config,
        split="train",
        num_episodes=1,
        memory_cache=True,
    )
    batch = move_batch_to_device(
        collate_episodes([dataset[0]]),
        "cpu",
    )
    model = OnlineWorldModel.from_config(config, device="cpu")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
    )
    optimizer.zero_grad(set_to_none=True)
    result = pretrain_rgb_measurements(
        model,
        batch,
        config,
        frame_index=0,
    )
    assert torch.isfinite(result.total_loss)
    result.total_loss.backward()
    optimizer.step()

    checkpoint = save_checkpoint(
        tmp_path / "last.pt",
        model=model,
        optimizer=optimizer,
        config=config,
        step=1,
        metrics={"loss_total": float(result.total_loss.detach())},
        device="cpu",
    )
    restored = OnlineWorldModel.from_config(config, device="cpu")
    restored_optimizer = torch.optim.AdamW(
        restored.parameters(),
        lr=config.training.learning_rate,
    )
    payload = load_checkpoint(
        checkpoint,
        model=restored,
        optimizer=restored_optimizer,
        map_location="cpu",
        expected_config=config,
    )

    assert payload["step"] == 1
    assert payload["config"]["model"] == config.to_dict()["model"]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(
            restored.state_dict()[name],
            value,
            rtol=0,
            atol=0,
        )
    assert restored_optimizer.state_dict()["state"]

    changed_bounds = list(config.simulator.world_bounds)
    changed_bounds[0] = (
        changed_bounds[0][0] - 0.25,
        changed_bounds[0][1],
    )
    incompatible = replace(
        config,
        simulator=replace(
            config.simulator,
            world_bounds=tuple(changed_bounds),
        ),
    )
    incompatible.validate()
    with pytest.raises(ValueError, match=r"simulator\.world_bounds"):
        validate_checkpoint_config(payload, incompatible)


def test_prior_simulator_checkpoint_is_transferable_but_not_exactly_resumable(
    tmp_path,
) -> None:
    config = _small_config()
    source = OnlineWorldModel.from_config(config, device="cpu")
    checkpoint = save_checkpoint(
        tmp_path / "prior-simulator.pt",
        model=source,
        optimizer=None,
        config=config,
        step=7,
        device="cpu",
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["simulator_version"] == SIMULATOR_VERSION
    payload["simulator_version"] = "sphere_world_v4"
    torch.save(payload, checkpoint)

    restored = OnlineWorldModel.from_config(config, device="cpu")
    loaded = load_checkpoint(
        checkpoint,
        model=restored,
        expected_config=config,
    )
    assert loaded["simulator_version"] == "sphere_world_v4"
    for name, value in source.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0.0, atol=0.0)

    initialized = OnlineWorldModel.from_config(config, device="cpu")
    transferred = load_model_weights(
        checkpoint,
        model=initialized,
        expected_config=config,
    )
    assert transferred["simulator_version"] == "sphere_world_v4"
    for name, value in source.state_dict().items():
        torch.testing.assert_close(
            initialized.state_dict()[name],
            value,
            rtol=0.0,
            atol=0.0,
        )

    with pytest.raises(
        ValueError,
        match="checkpoint simulator version differs from this exact resume",
    ):
        validate_training_resume_config(payload, config)


def test_checkpoint_loaders_require_simulator_provenance(tmp_path) -> None:
    config = _small_config()
    source = OnlineWorldModel.from_config(config, device="cpu")
    checkpoint = save_checkpoint(
        tmp_path / "missing-simulator-version.pt",
        model=source,
        optimizer=None,
        config=config,
        step=0,
        device="cpu",
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload.pop("simulator_version")
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="simulator_version"):
        load_checkpoint(
            checkpoint,
            model=OnlineWorldModel.from_config(config, device="cpu"),
            expected_config=config,
        )
    with pytest.raises(ValueError, match="simulator_version"):
        load_model_weights(
            checkpoint,
            model=OnlineWorldModel.from_config(config, device="cpu"),
            expected_config=config,
        )


def test_weight_only_transfer_allows_only_new_typed_attention_parameters(tmp_path) -> None:
    control_config = _small_config()
    control = OnlineWorldModel.from_config(control_config, device="cpu")
    checkpoint = save_checkpoint(
        tmp_path / "control.pt",
        model=control,
        optimizer=None,
        config=control_config,
        step=0,
    )
    pilot_config = replace(
        control_config,
        model=replace(
            control_config.model,
            dynamics=replace(
                control_config.model.dynamics,
                attention_residual_enabled=True,
                attention_width=128,
                attention_heads=4,
                attention_layers=4,
                attention_feed_forward_width=512,
            ),
        ),
    )
    pilot_config.validate()
    pilot = OnlineWorldModel.from_config(pilot_config, device="cpu")

    with pytest.raises(RuntimeError, match="missing required model keys"):
        load_model_weights(checkpoint, model=pilot)
    payload = load_model_weights(
        checkpoint,
        model=pilot,
        allowed_missing_prefixes=("dynamics.attention_interactions.",),
    )

    missing = payload["weight_load_missing_keys"]
    assert missing
    assert all(key.startswith("dynamics.attention_interactions.") for key in missing)
    pilot_state = pilot.state_dict()
    for name, value in control.state_dict().items():
        torch.testing.assert_close(pilot_state[name], value, rtol=0.0, atol=0.0)
    assert pilot.dynamics.attention_interactions is not None
    torch.testing.assert_close(
        pilot.dynamics.attention_interactions.node_decoder.weight,
        torch.zeros_like(pilot.dynamics.attention_interactions.node_decoder.weight),
    )


def test_weight_only_transfer_identity_grows_appended_attention_depth(tmp_path) -> None:
    control_config = _small_config()
    shallow_config = replace(
        control_config,
        model=replace(
            control_config.model,
            dynamics=replace(
                control_config.model.dynamics,
                attention_residual_enabled=True,
                attention_width=128,
                attention_heads=4,
                attention_layers=4,
                attention_feed_forward_width=512,
            ),
        ),
    )
    shallow = OnlineWorldModel.from_config(shallow_config, device="cpu")
    shallow_attention = shallow.dynamics.attention_interactions
    assert shallow_attention is not None
    with torch.no_grad():
        shallow_attention.node_decoder.weight.normal_(std=0.02)
        shallow_attention.relation_decoder.weight.normal_(std=0.02)
    checkpoint = save_checkpoint(
        tmp_path / "shallow_attention.pt",
        model=shallow,
        optimizer=None,
        config=shallow_config,
        step=0,
    )
    deeper_config = replace(
        shallow_config,
        model=replace(
            shallow_config.model,
            dynamics=replace(shallow_config.model.dynamics, attention_layers=6),
        ),
    )
    deeper = OnlineWorldModel.from_config(deeper_config, device="cpu")
    payload = load_model_weights(
        checkpoint,
        model=deeper,
        allowed_missing_prefixes=("dynamics.attention_interactions.",),
        architecture_growth_config=deeper_config,
    )

    assert payload["identity_grown_attention_blocks"] == (4, 5)
    deeper_attention = deeper.dynamics.attention_interactions
    assert deeper_attention is not None
    tokens = torch.randn(2, 7, 128)
    valid_mask = torch.tensor(
        [
            [True, True, True, True, True, True, True],
            [True, True, True, False, False, False, False],
        ]
    )
    shallow_tokens = tokens
    for block in shallow_attention.blocks:
        shallow_tokens = block(shallow_tokens, valid_mask)
    deeper_tokens = tokens
    for block in deeper_attention.blocks:
        deeper_tokens = block(deeper_tokens, valid_mask)
    torch.testing.assert_close(deeper_tokens, shallow_tokens, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        deeper_attention.node_decoder(deeper_attention.output_norm(deeper_tokens)),
        shallow_attention.node_decoder(shallow_attention.output_norm(shallow_tokens)),
        rtol=0.0,
        atol=0.0,
    )
    for index in (4, 5):
        block = deeper_attention.blocks[index]
        torch.testing.assert_close(
            block.attention.out_proj.weight,
            torch.zeros_like(block.attention.out_proj.weight),
        )
        torch.testing.assert_close(
            block.feed_forward.output.weight,
            torch.zeros_like(block.feed_forward.output.weight),
        )
    deeper_tokens.square().mean().backward()
    for index in (4, 5):
        block = deeper_attention.blocks[index]
        for output_weight in (
            block.attention.out_proj.weight,
            block.feed_forward.output.weight,
        ):
            assert output_weight.grad is not None
            assert torch.isfinite(output_weight.grad).all()
            assert torch.count_nonzero(output_weight.grad) > 0


def test_weight_only_transfer_rejects_malformed_attention_depth_growth(tmp_path) -> None:
    control_config = _small_config()
    shallow_config = replace(
        control_config,
        model=replace(
            control_config.model,
            dynamics=replace(
                control_config.model.dynamics,
                attention_residual_enabled=True,
                attention_width=128,
                attention_heads=4,
                attention_layers=4,
                attention_feed_forward_width=512,
            ),
        ),
    )
    shallow = OnlineWorldModel.from_config(shallow_config, device="cpu")
    checkpoint = save_checkpoint(
        tmp_path / "malformed_attention.pt",
        model=shallow,
        optimizer=None,
        config=shallow_config,
        step=0,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["model_state"].pop("dynamics.attention_interactions.blocks.2.attention.out_proj.weight")
    torch.save(payload, checkpoint)

    deeper_config = replace(
        shallow_config,
        model=replace(
            shallow_config.model,
            dynamics=replace(shallow_config.model.dynamics, attention_layers=6),
        ),
    )
    deeper = OnlineWorldModel.from_config(deeper_config, device="cpu")
    state_before = {name: value.detach().clone() for name, value in deeper.state_dict().items()}

    with pytest.raises(RuntimeError, match="partial architecture growth"):
        load_model_weights(
            checkpoint,
            model=deeper,
            allowed_missing_prefixes=("dynamics.attention_interactions.",),
            architecture_growth_config=deeper_config,
        )

    for name, value in deeper.state_dict().items():
        torch.testing.assert_close(value, state_before[name], rtol=0.0, atol=0.0)


def test_weight_only_attention_depth_growth_rejects_changed_head_semantics(tmp_path) -> None:
    control_config = _small_config()
    shallow_config = replace(
        control_config,
        model=replace(
            control_config.model,
            dynamics=replace(
                control_config.model.dynamics,
                attention_residual_enabled=True,
                attention_width=128,
                attention_heads=4,
                attention_layers=4,
                attention_feed_forward_width=512,
            ),
        ),
    )
    shallow = OnlineWorldModel.from_config(shallow_config, device="cpu")
    checkpoint = save_checkpoint(
        tmp_path / "four_heads.pt",
        model=shallow,
        optimizer=None,
        config=shallow_config,
        step=0,
    )
    changed_heads_config = replace(
        shallow_config,
        model=replace(
            shallow_config.model,
            dynamics=replace(
                shallow_config.model.dynamics,
                attention_heads=8,
                attention_layers=6,
            ),
        ),
    )
    changed_heads = OnlineWorldModel.from_config(changed_heads_config, device="cpu")
    state_before = {
        name: value.detach().clone() for name, value in changed_heads.state_dict().items()
    }

    with pytest.raises(ValueError, match="model except attention_layers"):
        load_model_weights(
            checkpoint,
            model=changed_heads,
            allowed_missing_prefixes=("dynamics.attention_interactions.",),
            architecture_growth_config=changed_heads_config,
        )

    for name, value in changed_heads.state_dict().items():
        torch.testing.assert_close(value, state_before[name], rtol=0.0, atol=0.0)


def test_checkpoint_save_rejects_nonfinite_model_and_optimizer_state(tmp_path) -> None:
    config = _small_config()
    model = OnlineWorldModel.from_config(config, device=torch.device("cpu"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    parameter = next(model.parameters())
    existing_target = save_checkpoint(
        tmp_path / "existing.pt",
        model=model,
        optimizer=optimizer,
        config=config,
        step=0,
    )
    existing_payload = existing_target.read_bytes()

    with torch.no_grad():
        parameter.flatten()[0] = float("nan")
    with pytest.raises(FloatingPointError, match="model_state.*NaN or Inf"):
        save_checkpoint(
            existing_target,
            model=model,
            optimizer=optimizer,
            config=config,
            step=1,
        )
    assert existing_target.read_bytes() == existing_payload
    model_target = tmp_path / "nonfinite-model.pt"
    with pytest.raises(FloatingPointError, match="model_state.*NaN or Inf"):
        save_checkpoint(
            model_target,
            model=model,
            optimizer=optimizer,
            config=config,
            step=1,
        )
    assert not model_target.exists()

    with torch.no_grad():
        parameter.flatten()[0] = 0.0
    buffer = next(
        buffer for buffer in model.buffers() if buffer.is_floating_point() and buffer.numel() > 0
    )
    original_buffer_value = buffer.flatten()[0].item()
    buffer.flatten()[0] = float("nan")
    buffer_target = tmp_path / "nonfinite-buffer.pt"
    with pytest.raises(FloatingPointError, match="model_state.*NaN or Inf"):
        save_checkpoint(
            buffer_target,
            model=model,
            optimizer=optimizer,
            config=config,
            step=1,
        )
    assert not buffer_target.exists()
    buffer.flatten()[0] = original_buffer_value

    optimizer.state[parameter] = {
        "step": torch.tensor(1.0),
        "exp_avg": torch.zeros_like(parameter),
        "exp_avg_sq": torch.zeros_like(parameter),
    }
    optimizer.state[parameter]["exp_avg"].flatten()[0] = float("inf")
    optimizer_target = tmp_path / "nonfinite-optimizer.pt"
    with pytest.raises(FloatingPointError, match="optimizer_state.*NaN or Inf"):
        save_checkpoint(
            optimizer_target,
            model=model,
            optimizer=optimizer,
            config=config,
            step=1,
        )
    assert not optimizer_target.exists()

    optimizer.state[parameter]["exp_avg"].zero_()
    optimizer.state[parameter]["step"].fill_(-1.0)
    step_target = tmp_path / "negative-optimizer-step.pt"
    with pytest.raises(FloatingPointError, match="optimizer step.*invalid"):
        save_checkpoint(
            step_target,
            model=model,
            optimizer=optimizer,
            config=config,
            step=1,
        )
    assert not step_target.exists()


def test_checkpoint_load_rejects_nonfinite_state_before_mutating_model(tmp_path) -> None:
    config = _small_config()
    source_model = OnlineWorldModel.from_config(config, device=torch.device("cpu"))
    payload = checkpoint_payload(
        model=source_model,
        optimizer=None,
        scheduler=None,
        config=config,
        step=1,
        metrics={},
        device="cpu",
    )
    tensor_name = next(
        name for name, tensor in payload["model_state"].items() if tensor.is_floating_point()
    )
    payload["model_state"][tensor_name].flatten()[0] = float("nan")
    source = tmp_path / "corrupt.pt"
    torch.save(payload, source)

    target_model = OnlineWorldModel.from_config(config, device=torch.device("cpu"))
    original = {name: tensor.detach().clone() for name, tensor in target_model.state_dict().items()}
    with pytest.raises(FloatingPointError, match="model_state.*NaN or Inf"):
        load_checkpoint(source, model=target_model)
    for name, tensor in target_model.state_dict().items():
        torch.testing.assert_close(tensor, original[name])


def test_checkpoint_rng_restore_uses_cpu_generator_state(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    config = _small_config()
    model = OnlineWorldModel.from_config(config, device="cpu")
    checkpoint = save_checkpoint(
        tmp_path / "rng.pt",
        model=model,
        optimizer=None,
        config=config,
        step=1,
        device="cpu",
    )
    observed_devices: list[torch.device] = []
    original = torch.set_rng_state

    def record_cpu_state(state: torch.Tensor) -> None:
        observed_devices.append(state.device)
        original(state)

    monkeypatch.setattr(torch, "set_rng_state", record_cpu_state)
    load_checkpoint(
        checkpoint,
        model=model,
        map_location="cpu",
        restore_rng=True,
        expected_config=config,
    )

    assert observed_devices == [torch.device("cpu")]


def test_checkpoint_copies_passed_source_provenance() -> None:
    config = _small_config()
    source_provenance = {
        "commit": "abc123",
        "dirty": True,
        "details": {"changed_paths": ["world_model/training/trainer.py"]},
    }
    payload = checkpoint_payload(
        model=torch.nn.Linear(1, 1),
        optimizer=None,
        scheduler=None,
        config=config,
        step=4,
        metrics={},
        device="cpu",
        source_provenance=source_provenance,
    )

    source_provenance["commit"] = "later-commit"
    source_provenance["details"]["changed_paths"].append("later-edit.py")

    assert payload["git"] == {
        "commit": "abc123",
        "dirty": True,
        "details": {"changed_paths": ["world_model/training/trainer.py"]},
    }
    # CPU-only runs should not initialize or persist an unrelated MPS stream.
    assert payload["rng"]["torch_mps"] is None


def test_training_resume_allows_only_non_numerical_operational_changes() -> None:
    config = _small_config()
    payload = {
        "config": config.to_dict(),
        "simulator_version": SIMULATOR_VERSION,
    }
    operational_change = replace(
        config,
        project=replace(
            config.project,
            name="renamed-run",
            output_dir="another-output-root",
        ),
        training=replace(
            config.training,
            steps=config.training.steps + 100,
            checkpoint_every=config.training.checkpoint_every + 1,
            log_every=config.training.log_every + 1,
            num_workers=config.training.num_workers + 1,
        ),
    )

    validate_training_resume_config(payload, operational_change)


def test_training_resume_rejects_closed_loop_validation_device_change() -> None:
    config = _small_config()
    payload = {
        "config": config.to_dict(),
        "simulator_version": SIMULATOR_VERSION,
    }
    changed = replace(
        config,
        device=replace(config.device, closed_loop_preference="cpu"),
    )

    with pytest.raises(
        ValueError,
        match=r"incompatible fields:.*device\.closed_loop_preference",
    ):
        validate_training_resume_config(payload, changed)


def test_training_resume_binds_state_roi_scope_and_late_transition() -> None:
    source = _small_config()
    state_roi = replace(
        source,
        training=replace(
            source.training,
            closed_loop_trainable_scope="state_roi",
            closed_loop_late_trainable_scope="state_dynamics_roi",
            closed_loop_scope_transition_steps=512,
        ),
    )
    state_roi.validate()
    payload = {
        "config": state_roi.to_dict(),
        "simulator_version": SIMULATOR_VERSION,
    }

    validate_training_resume_config(payload, state_roi)
    changed = replace(
        state_roi,
        training=replace(
            state_roi.training,
            closed_loop_trainable_scope="state_dynamics_roi",
        ),
    )
    with pytest.raises(
        ValueError,
        match=r"incompatible fields:.*training\.closed_loop_trainable_scope",
    ):
        validate_training_resume_config(payload, changed)


def test_training_resume_binds_state_relation_roi_late_scope() -> None:
    source = _small_config()
    relation_scope = replace(
        source,
        model=replace(
            source.model,
            dynamics=replace(
                source.model.dynamics,
                attention_residual_enabled=True,
            ),
        ),
        training=replace(
            source.training,
            closed_loop_trainable_scope="state_roi",
            closed_loop_late_trainable_scope="state_relation_roi",
            closed_loop_scope_transition_steps=512,
        ),
    )
    relation_scope.validate()
    payload = {
        "config": relation_scope.to_dict(),
        "simulator_version": SIMULATOR_VERSION,
    }

    validate_training_resume_config(payload, relation_scope)
    changed = replace(
        relation_scope,
        training=replace(
            relation_scope.training,
            closed_loop_late_trainable_scope="state_dynamics_roi",
        ),
    )
    with pytest.raises(
        ValueError,
        match=r"incompatible fields:.*training\.closed_loop_late_trainable_scope",
    ):
        validate_training_resume_config(payload, changed)


def test_training_resume_binds_scope_owned_event_weights_with_legacy_empty_default() -> None:
    config = _small_config()
    checkpoint_config = config.to_dict()
    checkpoint_config["training"].pop("closed_loop_event_loss_weights")
    legacy_payload = {
        "config": checkpoint_config,
        "simulator_version": SIMULATOR_VERSION,
    }

    # A checkpoint predating the field has the exact empty-map legacy
    # behavior and remains resumable under today's default.
    validate_training_resume_config(legacy_payload, config)

    changed = replace(
        config,
        training=replace(
            config.training,
            closed_loop_event_loss_weights={
                "state_roi": 0.0,
                "state_relation_roi": 0.05,
            },
        ),
    )
    changed.validate()
    with pytest.raises(
        ValueError,
        match=r"training\.closed_loop_event_loss_weights",
    ):
        validate_training_resume_config(legacy_payload, changed)


def test_training_resume_binds_prior_future_correction_with_legacy_true_default() -> None:
    config = _small_config()
    checkpoint_config = config.to_dict()
    checkpoint_config["training"].pop("closed_loop_prior_future_correction_enabled")
    legacy_payload = {
        "config": checkpoint_config,
        "simulator_version": SIMULATOR_VERSION,
    }

    validate_training_resume_config(legacy_payload, config)

    changed = replace(
        config,
        training=replace(
            config.training,
            closed_loop_prior_future_correction_enabled=False,
        ),
    )
    changed.validate()
    with pytest.raises(
        ValueError,
        match=r"training\.closed_loop_prior_future_correction_enabled",
    ):
        validate_training_resume_config(legacy_payload, changed)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("preference", "mps"),
        ("global_detector_cpu_on_mps", False),
        ("cuda_amp", True),
        ("mps_float32", False),
        ("compile", True),
    ],
)
def test_training_resume_rejects_execution_device_changes(
    field_name: str,
    value: object,
) -> None:
    config = _small_config()
    payload = {
        "config": config.to_dict(),
        "simulator_version": SIMULATOR_VERSION,
    }
    changed = replace(
        config,
        device=replace(config.device, **{field_name: value}),
    )

    with pytest.raises(
        ValueError,
        match=rf"incompatible fields:.*device\.{field_name}",
    ):
        validate_training_resume_config(payload, changed)


def test_training_resume_normalizes_explicit_legacy_defaults() -> None:
    config = _small_config()
    legacy_loss_weights = dict(config.training.loss_weights)
    legacy_loss_weights.pop("rollout_nll")
    legacy_config = replace(
        config,
        simulator=replace(
            config.simulator,
            ensured_pair_lateral_offset_range=(0.0, 0.0),
        ),
        training=replace(
            config.training,
            normalize_rollout_axes_over_configured_horizons=False,
            joint_collision_long_horizon_sampling=False,
            minimum_rollout_age_steps=0,
            loss_weights=legacy_loss_weights,
        ),
        device=replace(
            config.device,
            global_detector_cpu_on_mps=False,
        ),
    )
    checkpoint_config = legacy_config.to_dict()
    checkpoint_config["simulator"].pop("ensured_pair_lateral_offset_range")
    checkpoint_config["training"].pop("normalize_rollout_axes_over_configured_horizons")
    checkpoint_config["training"].pop("joint_collision_long_horizon_sampling")
    checkpoint_config["training"].pop("minimum_rollout_age_steps")
    for field_name in (
        "attention_node_grad_clip_norm",
        "attention_node_output_grad_clip_norm",
        "attention_collision_output_grad_clip_norm",
        "attention_force_output_grad_clip_norm",
        "attention_impulse_grad_clip_norm",
        "attention_impulse_output_grad_clip_norm",
        "minimum_interaction_gradient_retention",
        "closed_loop_learning_rate_schedule",
        "closed_loop_learning_rate_warmup_steps",
        "closed_loop_learning_rate_cosine_decay_steps",
        "closed_loop_learning_rate_minimum_scale",
    ):
        checkpoint_config["training"].pop(field_name)
    checkpoint_config["device"].pop("closed_loop_preference")
    checkpoint_config["device"].pop("global_detector_cpu_on_mps")
    payload = {
        "config": checkpoint_config,
        "simulator_version": SIMULATOR_VERSION,
    }

    validate_training_resume_config(payload, legacy_config)
    with pytest.raises(
        ValueError,
        match="training.joint_collision_long_horizon_sampling",
    ):
        validate_training_resume_config(payload, config)


@pytest.mark.parametrize(
    ("changed", "expected_path"),
    [
        (
            lambda config: replace(
                config,
                project=replace(config.project, seed=config.project.seed + 1),
            ),
            "project.seed",
        ),
        (
            lambda config: replace(
                config,
                training=replace(
                    config.training,
                    batch_size=config.training.batch_size + 1,
                ),
            ),
            "training.batch_size",
        ),
        (
            lambda config: replace(
                config,
                training=replace(
                    config.training,
                    closed_loop_learning_rate_schedule="warmup_cosine",
                    closed_loop_learning_rate_warmup_steps=4,
                    closed_loop_learning_rate_cosine_decay_steps=16,
                ),
            ),
            "training.closed_loop_learning_rate_cosine_decay_steps",
        ),
        (
            lambda config: replace(
                config,
                simulator=replace(
                    config.simulator,
                    scenario_mixture=("elastic_pairs",),
                ),
            ),
            "simulator.scenario_mixture",
        ),
        (
            lambda config: replace(
                config,
                model=replace(
                    config.model,
                    state=replace(
                        config.model.state,
                        appearance_dim=config.model.state.appearance_dim + 1,
                    ),
                ),
            ),
            "model.state.appearance_dim",
        ),
        (
            lambda config: replace(
                config,
                evaluation=replace(
                    config.evaluation,
                    confidence_level=0.8,
                ),
            ),
            "evaluation.confidence_level",
        ),
    ],
)
def test_training_resume_rejects_semantic_changes_with_field_diff(
    changed,
    expected_path: str,
) -> None:
    config = _small_config()
    payload = {
        "config": config.to_dict(),
        "simulator_version": SIMULATOR_VERSION,
    }

    with pytest.raises(ValueError, match=rf"incompatible fields:.*{expected_path}"):
        validate_training_resume_config(payload, changed(config))


def test_trainer_rejects_inexact_resume_before_overwriting_run_metadata(
    tmp_path,
) -> None:
    config = _small_config()
    run_directory = tmp_path / "existing-run"
    checkpoint = save_checkpoint(
        run_directory / "checkpoints" / "last.pt",
        model=torch.nn.Linear(1, 1),
        optimizer=None,
        config=config,
        step=1,
        device="cpu",
    )
    resolved_path = run_directory / "config.resolved.yaml"
    resolved_path.write_text("sentinel: original\n", encoding="utf-8")
    incompatible = replace(
        config,
        training=replace(
            config.training,
            batch_size=config.training.batch_size + 1,
        ),
    )

    with pytest.raises(ValueError, match="training.batch_size"):
        train_from_config(
            incompatible,
            resume_path=checkpoint,
            device_info=select_device("cpu"),
        )

    assert resolved_path.read_text(encoding="utf-8") == "sentinel: original\n"
    assert not (run_directory / "run_metadata.json").exists()


def test_trainer_rejects_prior_simulator_exact_resume_before_writing_metadata(
    tmp_path,
) -> None:
    config = _small_config()
    run_directory = tmp_path / "prior-simulator-run"
    checkpoint = save_checkpoint(
        run_directory / "checkpoints" / "last.pt",
        model=torch.nn.Linear(1, 1),
        optimizer=None,
        config=config,
        step=1,
        device="cpu",
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["simulator_version"] = "sphere_world_v4"
    torch.save(payload, checkpoint)

    with pytest.raises(
        ValueError,
        match="checkpoint simulator version differs from this exact resume",
    ):
        train_from_config(
            config,
            resume_path=checkpoint,
            device_info=select_device("cpu"),
        )

    assert not (run_directory / "run_metadata.json").exists()


@pytest.mark.parametrize("invalid_version", [None, 17, ""])
def test_trainer_rejects_invalid_simulator_provenance_without_mutating_artifacts(
    tmp_path,
    invalid_version,
) -> None:
    config = _small_config()
    run_directory = tmp_path / f"invalid-simulator-{invalid_version!r}"
    checkpoint = save_checkpoint(
        run_directory / "checkpoints" / "last.pt",
        model=torch.nn.Linear(1, 1),
        optimizer=None,
        config=config,
        step=1,
        device="cpu",
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if invalid_version is None:
        payload.pop("simulator_version")
    else:
        payload["simulator_version"] = invalid_version
    torch.save(payload, checkpoint)
    resolved_path = run_directory / "config.resolved.yaml"
    resolved_path.write_text("sentinel: original\n", encoding="utf-8")

    with pytest.raises(ValueError, match="valid top-level simulator_version"):
        train_from_config(
            config,
            resume_path=checkpoint,
            device_info=select_device("cpu"),
        )

    assert resolved_path.read_text(encoding="utf-8") == "sentinel: original\n"
    assert not (run_directory / "run_metadata.json").exists()


def test_trainer_rejects_checkpoint_from_different_execution_device(
    tmp_path,
) -> None:
    config = _small_config()
    run_directory = tmp_path / "device-mismatch"
    checkpoint = save_checkpoint(
        run_directory / "checkpoints" / "last.pt",
        model=torch.nn.Linear(1, 1),
        optimizer=None,
        config=config,
        step=1,
        device="mps",
    )

    with pytest.raises(
        ValueError,
        match="checkpoint execution device does not match this exact resume",
    ):
        train_from_config(
            config,
            resume_path=checkpoint,
        )

    assert not (run_directory / "run_metadata.json").exists()


def test_already_complete_resume_does_not_rewrite_historical_device_checkpoint(
    tmp_path,
) -> None:
    config = _small_config()
    config = replace(
        config,
        project=replace(config.project, output_dir=str(tmp_path / "runs")),
        device=replace(
            config.device,
            preference="mps",
            closed_loop_preference="same",
        ),
        training=replace(
            config.training,
            steps=1,
            rgb_pretrain_steps=0,
        ),
    )
    model = OnlineWorldModel.from_config(config, device="cpu")
    run_directory = tmp_path / "completed"
    checkpoint = save_checkpoint(
        run_directory / "checkpoints" / "last.pt",
        model=model,
        optimizer=None,
        config=config,
        step=1,
        metrics={"measurement_handoff_completed": 1.0},
        device="mps",
    )
    original_bytes = checkpoint.read_bytes()
    summary_path = run_directory / "train_summary.json"
    original_summary = (
        b'{\n  "completed_steps": 1,\n'
        b'  "elapsed_seconds": 987.5,\n'
        b'  "evidence": "original training segment",\n'
        b'  "last_metrics": {"loss_total": 4.25}\n'
        b"}\n"
    )
    summary_path.write_bytes(original_summary)

    result = train_from_config(
        config,
        resume_path=checkpoint,
    )

    assert result["completed_steps"] == 1
    assert result["elapsed_seconds"] == 987.5
    assert result["last_metrics"] == {"loss_total": 4.25}
    assert result["no_op_exact_resume"] is True
    assert result["optimizer_updates_this_invocation"] == 0
    assert result["resumed_from"] == str(checkpoint.resolve())
    assert result["resume_inspection_elapsed_seconds"] >= 0.0
    assert summary_path.read_bytes() == original_summary
    assert checkpoint.read_bytes() == original_bytes
    persisted = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert persisted["device"] == "mps"


def test_already_complete_resume_to_new_run_writes_truthful_no_op_summary(
    tmp_path,
) -> None:
    config = _small_config()
    config = replace(
        config,
        project=replace(config.project, output_dir=str(tmp_path / "runs")),
        device=replace(
            config.device,
            preference="mps",
            closed_loop_preference="same",
        ),
        training=replace(
            config.training,
            steps=1,
            rgb_pretrain_steps=0,
        ),
    )
    model = OnlineWorldModel.from_config(config, device="cpu")
    source_run = tmp_path / "source-completed"
    checkpoint = save_checkpoint(
        source_run / "checkpoints" / "last.pt",
        model=model,
        optimizer=None,
        config=config,
        step=1,
        metrics={
            "loss_total": 3.5,
            "measurement_handoff_completed": 1.0,
        },
        device="mps",
    )
    checkpoint_bytes = checkpoint.read_bytes()
    source_summary = source_run / "train_summary.json"
    source_summary.write_text('{"evidence": "source"}\n', encoding="utf-8")
    source_summary_bytes = source_summary.read_bytes()

    result = train_from_config(
        config,
        run_name="no-op-copy",
        resume_path=checkpoint,
    )

    destination = torch.load(
        result["last_checkpoint"],
        map_location="cpu",
        weights_only=False,
    )
    result_path = Path(result["run_directory"])
    summary = json.loads((result_path / "train_summary.json").read_text(encoding="utf-8"))
    assert result_path != source_run
    assert summary["no_op_exact_resume"] is True
    assert summary["optimizer_updates_this_invocation"] == 0
    assert summary["device"] == "mps"
    assert summary["last_metrics"]["loss_total"] == 3.5
    assert destination["device"] == "mps"
    assert Path(result["last_checkpoint"]).read_bytes() == checkpoint_bytes
    assert source_summary.read_bytes() == source_summary_bytes


@pytest.mark.skipif(
    not (
        torch.backends.mps.is_built()
        and hasattr(torch.mps, "get_rng_state")
        and hasattr(torch.mps, "set_rng_state")
    ),
    reason="PyTorch was built without MPS RNG APIs",
)
def test_checkpoint_restores_mps_rng_state_when_training_on_mps(tmp_path) -> None:
    config = _small_config()
    model = torch.nn.Linear(1, 1)
    try:
        original_state = torch.mps.get_rng_state().cpu()
    except RuntimeError as error:
        pytest.skip(f"MPS runtime is unavailable: {error}")
    try:
        torch.mps.manual_seed(9182)
        expected_state = torch.mps.get_rng_state().cpu()
        checkpoint = save_checkpoint(
            tmp_path / "mps-rng.pt",
            model=model,
            optimizer=None,
            config=config,
            step=1,
            device="mps",
        )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        torch.testing.assert_close(
            payload["rng"]["torch_mps"],
            expected_state,
            rtol=0,
            atol=0,
        )

        torch.mps.manual_seed(7721)
        load_checkpoint(
            checkpoint,
            model=model,
            map_location="cpu",
            restore_rng=True,
            expected_config=config,
        )
        torch.testing.assert_close(
            torch.mps.get_rng_state().cpu(),
            expected_state,
            rtol=0,
            atol=0,
        )
    finally:
        torch.mps.set_rng_state(original_state)


def test_rgb_runtime_controls_are_semantic_with_legacy_defaults() -> None:
    config = _small_config()
    payload = {"config": config.to_dict()}
    legacy_payload = deepcopy(payload)
    legacy_rgb = legacy_payload["config"]["model"]["rgb"]
    for field_name in (
        "temporal_velocity_enabled",
        "temporal_velocity_history_size",
        "temporal_velocity_min_dt",
        "temporal_velocity_variance_scale",
        "temporal_velocity_variance_floor",
        "temporal_velocity_variance_ceiling",
        "temporal_velocity_independent_raw_history_enabled",
        "temporal_velocity_continuous_gravity_axis_enabled",
        "structured_disc_center_enabled",
        "structured_disc_threshold",
        "structured_disc_min_pixels",
        "structured_disc_max_assignment_distance",
        "structured_disc_center_std_pixels",
    ):
        legacy_rgb.pop(field_name)

    enabled = replace(
        config,
        model=replace(
            config.model,
            rgb=replace(
                config.model.rgb,
                temporal_velocity_enabled=True,
                temporal_velocity_history_size=4,
                temporal_velocity_min_dt=0.002,
                temporal_velocity_variance_scale=2.0,
                temporal_velocity_variance_floor=0.5,
                temporal_velocity_variance_ceiling=4.0,
                temporal_velocity_independent_raw_history_enabled=True,
                temporal_velocity_continuous_gravity_axis_enabled=True,
                structured_disc_center_enabled=True,
                structured_disc_threshold=0.06,
                structured_disc_min_pixels=6,
                structured_disc_max_assignment_distance=0.5,
                structured_disc_center_std_pixels=1.0,
            ),
        ),
    )
    enabled.validate()
    validate_checkpoint_config(payload, config)
    legacy_compatible = replace(
        config,
        model=replace(
            config.model,
            rgb=replace(config.model.rgb, structured_disc_center_enabled=False),
        ),
    )
    legacy_compatible.validate()
    validate_checkpoint_config(legacy_payload, legacy_compatible)
    with pytest.raises(ValueError, match="model"):
        validate_checkpoint_config(legacy_payload, config)
    with pytest.raises(ValueError, match="model"):
        validate_checkpoint_config(payload, enabled)
    with pytest.raises(ValueError, match="model"):
        validate_checkpoint_config(legacy_payload, enabled)

    architecture_change = replace(
        enabled,
        model=replace(
            enabled.model,
            rgb=replace(enabled.model.rgb, roi_size=enabled.model.rgb.roi_size + 1),
        ),
    )
    architecture_change.validate()
    with pytest.raises(ValueError, match="model"):
        validate_checkpoint_config(payload, architecture_change)


def test_disabled_runtime_hypothesis_policy_migrates_but_enabled_policy_is_strict() -> None:
    config = _small_config()
    payload = {"config": config.to_dict()}
    legacy_payload = deepcopy(payload)
    legacy_runtime = legacy_payload["config"]["runtime"]
    for field_name in (
        "hypothesis_pool_enabled",
        "hypothesis_evidence_horizons_seconds",
        "hypothesis_axis_independent_axes",
        "hypothesis_axis_prior_strength",
        "hypothesis_evidence_decay",
        "hypothesis_timestamp_tolerance_seconds",
    ):
        legacy_runtime.pop(field_name)

    validate_checkpoint_config(legacy_payload, config)
    enabled = replace(
        config,
        runtime=replace(config.runtime, hypothesis_pool_enabled=True),
    )
    enabled.validate()
    with pytest.raises(ValueError, match="runtime"):
        validate_checkpoint_config(legacy_payload, enabled)


def test_innovation_anchored_correction_is_semantic_with_legacy_false() -> None:
    config = _small_config()
    payload = {"config": config.to_dict()}
    legacy_payload = deepcopy(payload)
    legacy_payload["config"]["model"]["filter"].pop("innovation_anchored_correction")

    validate_checkpoint_config(legacy_payload, config)
    corrected = replace(
        config,
        model=replace(
            config.model,
            filter=replace(
                config.model.filter,
                innovation_anchored_correction=True,
            ),
        ),
    )
    corrected.validate()
    with pytest.raises(ValueError, match="model"):
        validate_checkpoint_config(legacy_payload, corrected)
    with pytest.raises(ValueError, match="model"):
        validate_checkpoint_config(payload, corrected)


def test_rgbd_analytic_bridge_fields_are_semantic_with_legacy_disabled_defaults() -> None:
    config = _small_config()
    payload = {
        "config": config.to_dict(),
        "simulator_version": SIMULATOR_VERSION,
    }
    legacy_payload = deepcopy(payload)
    legacy_payload["config"]["model"].pop("rgbd")
    legacy_payload["config"]["model"]["dynamics"].pop("analytic_free_motion_only")
    legacy_payload["config"]["model"]["filter"].pop("enable_learned_corrector")
    legacy_payload["config"]["model"]["filter"].pop("direct_metric_position_update")
    existing_rgbd_legacy_payload = deepcopy(payload)
    for field_name in (
        "bounded_partial_visibility",
        "minimum_observed_support_fraction",
        "maximum_surface_residual_relative_rms",
        "maximum_full_silhouette_overlap_fraction",
        "max_missing_rows",
        "require_latest_valid",
    ):
        existing_rgbd_legacy_payload["config"]["model"]["rgbd"].pop(field_name)

    validate_checkpoint_config(legacy_payload, config)
    validate_training_resume_config(legacy_payload, config)
    validate_checkpoint_config(existing_rgbd_legacy_payload, config)
    validate_training_resume_config(existing_rgbd_legacy_payload, config)

    rgbd_enabled = replace(
        config,
        model=replace(
            config.model,
            max_objects=1,
            rgbd=replace(
                config.model.rgbd,
                enabled=True,
                world_radius=config.simulator.radius_range[0],
                linear_drag=config.simulator.drag_range[0],
            ),
        ),
        simulator=replace(
            config.simulator,
            min_objects=1,
            max_objects=1,
            radius_range=(
                config.simulator.radius_range[0],
                config.simulator.radius_range[0],
            ),
            drag_range=(
                config.simulator.drag_range[0],
                config.simulator.drag_range[0],
            ),
        ),
        runtime=replace(config.runtime, modality="rgbd"),
        evaluation=replace(config.evaluation, rgb_only=False),
    )
    rgbd_enabled.validate()
    analytic_enabled = replace(
        config,
        model=replace(
            config.model,
            dynamics=replace(config.model.dynamics, analytic_free_motion_only=True),
        ),
    )
    analytic_enabled.validate()
    learned_disabled = replace(
        config,
        model=replace(
            config.model,
            filter=replace(config.model.filter, enable_learned_corrector=False),
        ),
    )
    learned_disabled.validate()
    drag_changed = replace(
        config,
        model=replace(
            config.model,
            rgbd=replace(config.model.rgbd, linear_drag=0.07),
        ),
    )
    drag_changed.validate()
    direct_enabled = replace(
        rgbd_enabled,
        model=replace(
            rgbd_enabled.model,
            rgb=replace(rgbd_enabled.model.rgb, enabled=False),
            filter=replace(
                rgbd_enabled.model.filter,
                enable_learned_corrector=False,
                direct_metric_position_update=True,
            ),
        ),
    )
    direct_enabled.validate()
    partial_enabled = replace(
        rgbd_enabled,
        model=replace(
            rgbd_enabled.model,
            max_objects=2,
            state=replace(rgbd_enabled.model.state, appearance_dim=3),
            rgbd=replace(
                rgbd_enabled.model.rgbd,
                proposal_count=2,
                bounded_partial_visibility=True,
                max_missing_rows=1,
            ),
        ),
        simulator=replace(
            rgbd_enabled.simulator,
            min_objects=2,
            max_objects=2,
            ensure_collision=False,
        ),
    )
    partial_enabled.validate()
    two_object_default = replace(
        partial_enabled,
        model=replace(
            partial_enabled.model,
            rgbd=replace(
                partial_enabled.model.rgbd,
                bounded_partial_visibility=False,
                max_missing_rows=0,
            ),
        ),
    )
    two_object_default.validate()
    two_object_rgbd_legacy_payload = {
        "config": two_object_default.to_dict(),
        "simulator_version": SIMULATOR_VERSION,
    }
    for field_name in (
        "bounded_partial_visibility",
        "minimum_observed_support_fraction",
        "maximum_surface_residual_relative_rms",
        "maximum_full_silhouette_overlap_fraction",
        "max_missing_rows",
        "require_latest_valid",
    ):
        two_object_rgbd_legacy_payload["config"]["model"]["rgbd"].pop(field_name)
    validate_checkpoint_config(two_object_rgbd_legacy_payload, two_object_default)
    validate_training_resume_config(two_object_rgbd_legacy_payload, two_object_default)
    with pytest.raises(ValueError, match="model"):
        validate_checkpoint_config(two_object_rgbd_legacy_payload, partial_enabled)
    with pytest.raises(ValueError, match="model"):
        validate_training_resume_config(two_object_rgbd_legacy_payload, partial_enabled)

    for changed in (
        rgbd_enabled,
        analytic_enabled,
        learned_disabled,
        drag_changed,
        direct_enabled,
        partial_enabled,
    ):
        with pytest.raises(ValueError, match="model"):
            validate_checkpoint_config(legacy_payload, changed)
        with pytest.raises(ValueError, match="model"):
            validate_training_resume_config(legacy_payload, changed)


def test_attention_relation_endpoint_binding_is_semantic_with_legacy_false() -> None:
    config = _small_config()
    historical = replace(
        config,
        model=replace(
            config.model,
            dynamics=replace(
                config.model.dynamics,
                attention_residual_enabled=True,
            ),
        ),
    )
    historical.validate()
    payload = {
        "config": historical.to_dict(),
        "simulator_version": SIMULATOR_VERSION,
    }
    legacy_payload = deepcopy(payload)
    legacy_payload["config"]["model"]["dynamics"].pop("attention_relation_endpoint_binding_enabled")

    validate_checkpoint_config(legacy_payload, historical)
    validate_training_resume_config(legacy_payload, historical)

    enabled = replace(
        historical,
        model=replace(
            historical.model,
            dynamics=replace(
                historical.model.dynamics,
                attention_relation_endpoint_binding_enabled=True,
            ),
        ),
    )
    enabled.validate()
    enabled_model = OnlineWorldModel.from_config(enabled, device="cpu")
    enabled_attention = enabled_model.dynamics.attention_interactions
    assert enabled_attention is not None
    assert enabled_attention.relation_endpoint_binding_enabled
    with pytest.raises(ValueError, match="model"):
        validate_checkpoint_config(legacy_payload, enabled)
    with pytest.raises(
        ValueError,
        match=r"model\.dynamics\.attention_relation_endpoint_binding_enabled",
    ):
        validate_training_resume_config(legacy_payload, enabled)


def test_smooth_event_hazard_is_semantic_with_legacy_false() -> None:
    config = _small_config()
    payload = {
        "config": config.to_dict(),
        "simulator_version": SIMULATOR_VERSION,
    }
    legacy_payload = deepcopy(payload)
    for field_name in (
        "smooth_event_hazard_enabled",
        "event_hazard_gap_temperature_m",
        "event_hazard_velocity_temperature_mps",
        "event_hazard_resolved_logit_floor",
    ):
        legacy_payload["config"]["model"]["dynamics"].pop(field_name)

    validate_checkpoint_config(legacy_payload, config)
    validate_training_resume_config(legacy_payload, config)

    enabled = replace(
        config,
        model=replace(
            config.model,
            dynamics=replace(
                config.model.dynamics,
                smooth_event_hazard_enabled=True,
            ),
        ),
    )
    enabled.validate()
    enabled_model = OnlineWorldModel.from_config(enabled, device="cpu")
    assert enabled_model.dynamics.events.smooth_hazard_enabled
    with pytest.raises(ValueError, match="model"):
        validate_checkpoint_config(legacy_payload, enabled)
    with pytest.raises(
        ValueError,
        match=r"model\.dynamics\.smooth_event_hazard_enabled",
    ):
        validate_training_resume_config(legacy_payload, enabled)


def test_runtime_invariant_semantics_preserve_historical_contact_defaults() -> None:
    config = _small_config()
    historical = replace(
        config,
        model=replace(
            config.model,
            dynamics=replace(
                config.model.dynamics,
                contact_margin=1.0e-3,
                boundary_contact_tolerance=1.0e-3,
                contact_confidence_sigma=0.25,
                pair_collision_speed_epsilon=1.0e-4,
                boundary_collision_speed_epsilon=0.1,
            ),
        ),
    )
    historical.validate()
    payload = {"config": historical.to_dict()}
    legacy_payload = deepcopy(payload)
    for field_name in (
        "contact_margin",
        "boundary_contact_tolerance",
        "contact_confidence_sigma",
        "pair_collision_speed_epsilon",
        "boundary_collision_speed_epsilon",
    ):
        legacy_payload["config"]["model"]["dynamics"].pop(field_name)
    legacy_payload["config"]["model"]["association"].pop("minimum_measurement_confidence")
    legacy_payload["config"]["model"]["lifecycle"].pop("max_occluded_steps")

    validate_checkpoint_config(payload, historical)
    validate_checkpoint_config(legacy_payload, historical)
    with pytest.raises(ValueError, match="model"):
        validate_checkpoint_config(legacy_payload, config)

    changed = replace(
        historical,
        model=replace(
            historical.model,
            dynamics=replace(
                historical.model.dynamics,
                contact_confidence_sigma=0.5,
            ),
        ),
    )
    changed.validate()
    with pytest.raises(ValueError, match="model"):
        validate_checkpoint_config(payload, changed)
    with pytest.raises(ValueError, match="model"):
        validate_checkpoint_config(legacy_payload, changed)

    association_change = replace(
        historical,
        model=replace(
            historical.model,
            association=replace(
                historical.model.association,
                minimum_measurement_confidence=0.25,
            ),
        ),
    )
    association_change.validate()
    with pytest.raises(ValueError, match="model"):
        validate_checkpoint_config(payload, association_change)
    with pytest.raises(ValueError, match="model"):
        validate_checkpoint_config(legacy_payload, association_change)

    lifecycle_change = replace(
        config,
        model=replace(
            config.model,
            lifecycle=replace(
                config.model.lifecycle,
                max_occluded_steps=config.model.lifecycle.max_occluded_steps + 1,
            ),
        ),
    )
    lifecycle_change.validate()
    with pytest.raises(ValueError, match="model"):
        validate_checkpoint_config(payload, lifecycle_change)
    with pytest.raises(ValueError, match="model"):
        validate_checkpoint_config(legacy_payload, lifecycle_change)
