from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import world_model.training.trainer as training_trainer
from world_model.datasets import SyntheticSphereDataset, collate_episodes
from world_model.runtime import OnlineWorldModel
from world_model.training.checkpointing import (
    capture_checkpoint_snapshot,
    checkpoint_payload,
    load_checkpoint,
    load_model_weights,
    save_checkpoint,
    validate_checkpoint_config,
    validate_exact_resume_state,
    validate_training_resume_config,
)
from world_model.training.loop import (
    move_batch_to_device,
    pretrain_rgb_measurements,
)
from world_model.training.trainer import (
    _ROLLOUT_SELECTION_METRIC_VERSION,
    _validation_protocol_checkpoint_metrics,
    closed_loop_learning_rate_at_update,
    train_from_config,
)
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


def test_checkpoint_copies_artifact_metadata_and_marks_weight_only_resume_boundary() -> None:
    config = _small_config()
    artifact_metadata = {
        "role": "weight_only_initializer",
        "composition": {"module_prefixes": ["updater"]},
    }
    payload = checkpoint_payload(
        model=torch.nn.Linear(1, 1),
        optimizer=None,
        scheduler=None,
        config=config,
        step=0,
        metrics={"checkpoint_state_role": "weight_only_initializer"},
        device="cpu",
        artifact_metadata=artifact_metadata,
    )

    artifact_metadata["composition"]["module_prefixes"].append("identifier")
    assert payload["artifact_metadata"] == {
        "role": "weight_only_initializer",
        "composition": {"module_prefixes": ["updater"]},
    }
    with pytest.raises(ValueError, match="cannot be exactly resumed.*--initialize-from"):
        validate_exact_resume_state(payload)

    ordinary_weight_only = dict(payload)
    ordinary_weight_only["artifact_metadata"] = {"role": "diagnostic"}
    ordinary_weight_only["metrics"] = {}
    with pytest.raises(ValueError, match="requires checkpoint optimizer state"):
        validate_exact_resume_state(ordinary_weight_only)


def test_public_checkpoint_snapshot_hashes_the_same_immutable_byte_read(tmp_path: Path) -> None:
    source = tmp_path / "last.pt"
    original = b"first checkpoint bytes"
    source.write_bytes(original)

    with capture_checkpoint_snapshot(source) as captured:
        source.write_bytes(b"replacement checkpoint bytes")
        assert captured.source_path == source.resolve()
        assert captured.sha256 == hashlib.sha256(original).hexdigest()
        assert captured.byte_count == len(original)
        assert captured.snapshot_path.read_bytes() == original
        snapshot_path = captured.snapshot_path
    assert not snapshot_path.exists()


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


def test_training_resume_binds_rgb_pretrain_trainable_scope_with_legacy_all() -> None:
    source = _small_config()
    detector = replace(
        source,
        training=replace(
            source.training,
            rgb_pretrain_trainable_scope="global_detector",
        ),
    )
    detector.validate()
    payload = {
        "config": detector.to_dict(),
        "simulator_version": SIMULATOR_VERSION,
    }
    with pytest.raises(
        ValueError,
        match=r"incompatible fields:.*training\.rgb_pretrain_trainable_scope",
    ):
        validate_training_resume_config(payload, source)

    legacy_payload = {
        "config": source.to_dict(),
        "simulator_version": SIMULATOR_VERSION,
    }
    legacy_payload["config"]["training"].pop("rgb_pretrain_trainable_scope")
    validate_training_resume_config(legacy_payload, source)
    with pytest.raises(
        ValueError,
        match=r"incompatible fields:.*training\.rgb_pretrain_trainable_scope",
    ):
        validate_training_resume_config(legacy_payload, detector)


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


@pytest.mark.parametrize(
    "scope",
    (
        "updater_state_heads",
        "updater_state_heads_xy",
        "updater_state_heads_xy_collision",
        "updater_state_heads_xy_collision_node",
    ),
)
def test_training_resume_binds_updater_state_heads_scope(scope: str) -> None:
    source = _small_config()
    if scope in {
        "updater_state_heads_xy_collision",
        "updater_state_heads_xy_collision_node",
    }:
        source = replace(
            source,
            model=replace(
                source.model,
                dynamics=replace(source.model.dynamics, attention_residual_enabled=True),
            ),
            training=replace(
                source.training,
                closed_loop_event_loss_weights={scope: 0.0045},
            ),
        )
    updater_state_heads = replace(
        source,
        training=replace(
            source.training,
            closed_loop_trainable_scope=scope,
        ),
    )
    updater_state_heads.validate()
    payload = {
        "config": updater_state_heads.to_dict(),
        "simulator_version": SIMULATOR_VERSION,
    }

    validate_training_resume_config(payload, updater_state_heads)
    changed = replace(
        updater_state_heads,
        training=replace(
            updater_state_heads.training,
            closed_loop_trainable_scope=(
                "updater" if scope == "updater_state_heads" else "updater_state_heads"
            ),
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


def test_training_resume_binds_state_event_weights_with_legacy_empty_default() -> None:
    base = _small_config()
    scope = "updater_state_heads_xy_collision_node"
    config = replace(
        base,
        model=replace(
            base.model,
            dynamics=replace(base.model.dynamics, attention_residual_enabled=True),
        ),
        training=replace(
            base.training,
            closed_loop_trainable_scope=scope,
            closed_loop_event_loss_weights={scope: 0.0045},
        ),
    )
    config.validate()
    checkpoint_config = config.to_dict()
    checkpoint_config["training"].pop("closed_loop_state_event_loss_weights")
    legacy_payload = {
        "config": checkpoint_config,
        "simulator_version": SIMULATOR_VERSION,
    }

    validate_training_resume_config(legacy_payload, config)

    changed = replace(
        config,
        training=replace(
            config.training,
            closed_loop_state_event_loss_weights={scope: 0.04},
        ),
    )
    changed.validate()
    with pytest.raises(
        ValueError,
        match=r"training\.closed_loop_state_event_loss_weights",
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
    "field_name",
    [
        "closed_loop_batch_macro_physical_losses_enabled",
        "closed_loop_axiswise_correction_hinge_enabled",
        "closed_loop_modular_gradient_ownership_enabled",
        "closed_loop_scenario_tail_fraction",
        "closed_loop_uncertainty_standardized_error_gradient_cap",
        "closed_loop_protected_reference_nonregression_weight",
    ],
)
def test_training_resume_binds_physical_objective_repairs_with_legacy_default(
    field_name: str,
) -> None:
    config = _small_config()
    checkpoint_config = config.to_dict()
    checkpoint_config["training"].pop(field_name)
    legacy_payload = {
        "config": checkpoint_config,
        "simulator_version": SIMULATOR_VERSION,
    }

    validate_training_resume_config(legacy_payload, config)
    value = (
        0.25
        if field_name == "closed_loop_scenario_tail_fraction"
        else 25.0
        if field_name == "closed_loop_uncertainty_standardized_error_gradient_cap"
        else 1.0
        if field_name == "closed_loop_protected_reference_nonregression_weight"
        else True
    )
    changed_values: dict[str, object] = {
        "scenario_balanced_batches": field_name
        in {
            "closed_loop_scenario_tail_fraction",
            "closed_loop_protected_reference_nonregression_weight",
        },
        "batch_size": (
            len(config.simulator.scenario_mixture)
            if field_name
            in {
                "closed_loop_scenario_tail_fraction",
                "closed_loop_protected_reference_nonregression_weight",
            }
            else config.training.batch_size
        ),
        "train_episodes": (
            len(config.simulator.scenario_mixture)
            if field_name
            in {
                "closed_loop_scenario_tail_fraction",
                "closed_loop_protected_reference_nonregression_weight",
            }
            else config.training.train_episodes
        ),
        "closed_loop_batch_macro_physical_losses_enabled": (
            field_name
            in {
                "closed_loop_scenario_tail_fraction",
                "closed_loop_protected_reference_nonregression_weight",
            }
            or config.training.closed_loop_batch_macro_physical_losses_enabled
        ),
        "closed_loop_axiswise_correction_hinge_enabled": (
            field_name
            in {
                "closed_loop_scenario_tail_fraction",
                "closed_loop_protected_reference_nonregression_weight",
            }
            or config.training.closed_loop_axiswise_correction_hinge_enabled
        ),
        "closed_loop_modular_gradient_ownership_enabled": (
            field_name == "closed_loop_modular_gradient_ownership_enabled"
            or config.training.closed_loop_modular_gradient_ownership_enabled
        ),
        "closed_loop_trainable_scope": (
            "differentiable_state_estimator"
            if field_name == "closed_loop_modular_gradient_ownership_enabled"
            else config.training.closed_loop_trainable_scope
        ),
        "rgb_pretrain_steps": (
            0
            if field_name == "closed_loop_protected_reference_nonregression_weight"
            else config.training.rgb_pretrain_steps
        ),
        field_name: value,
    }
    changed_training = replace(config.training, **changed_values)
    changed = replace(config, training=changed_training)
    changed.validate()
    with pytest.raises(
        ValueError,
        match=rf"training\.{field_name}",
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
    checkpoint_config["simulator"].pop("ensured_pair_vertical_speed_range")
    checkpoint_config["simulator"].pop("ensured_pair_event_frame_range")
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
                simulator=replace(
                    config.simulator,
                    ensured_pair_vertical_speed_range=(4.7, 5.1),
                ),
            ),
            "simulator.ensured_pair_vertical_speed_range",
        ),
        (
            lambda config: replace(
                config,
                simulator=replace(
                    config.simulator,
                    ensured_pair_event_frame_range=(20, 24),
                ),
            ),
            "simulator.ensured_pair_event_frame_range",
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


def test_trainer_rejects_weight_only_initializer_as_exact_resume(tmp_path: Path) -> None:
    config = _small_config()
    run_directory = tmp_path / "weight-only-initializer"
    checkpoint = save_checkpoint(
        run_directory / "checkpoints" / "last.pt",
        model=torch.nn.Linear(1, 1),
        optimizer=None,
        config=config,
        step=0,
        metrics={"checkpoint_state_role": "weight_only_initializer"},
        artifact_metadata={"role": "weight_only_initializer"},
        device="cpu",
    )

    with pytest.raises(ValueError, match="cannot be exactly resumed.*--initialize-from"):
        train_from_config(
            config,
            resume_path=checkpoint,
            device_info=select_device("cpu"),
        )

    assert not (run_directory / "run_metadata.json").exists()


def test_trainer_rejects_missing_optimizer_state_as_exact_resume(tmp_path: Path) -> None:
    config = _small_config()
    run_directory = tmp_path / "missing-optimizer-state"
    model = OnlineWorldModel.from_config(config, device="cpu")
    checkpoint = save_checkpoint(
        run_directory / "checkpoints" / "last.pt",
        model=model,
        optimizer=None,
        config=config,
        step=1,
        metrics={
            "rollout_selection_metric_version": _ROLLOUT_SELECTION_METRIC_VERSION,
            **_validation_protocol_checkpoint_metrics(config),
        },
        device="cpu",
    )

    with pytest.raises(ValueError, match="requires checkpoint optimizer state.*--initialize-from"):
        train_from_config(
            config,
            resume_path=checkpoint,
            device_info=select_device("cpu"),
        )

    assert not (run_directory / "run_metadata.json").exists()


@pytest.mark.parametrize("missing_field", [None, "python", "numpy", "torch_cpu"])
def test_trainer_rejects_incomplete_rng_state_before_overwriting_metadata(
    tmp_path: Path,
    missing_field: str | None,
) -> None:
    config = _small_config()
    run_directory = tmp_path / f"missing-rng-{missing_field or 'mapping'}"
    model = OnlineWorldModel.from_config(config, device="cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    checkpoint = save_checkpoint(
        run_directory / "checkpoints" / "last.pt",
        model=model,
        optimizer=optimizer,
        config=config,
        step=1,
        metrics={
            "rollout_selection_metric_version": _ROLLOUT_SELECTION_METRIC_VERSION,
            **_validation_protocol_checkpoint_metrics(config),
        },
        device="cpu",
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if missing_field is None:
        payload.pop("rng")
    else:
        payload["rng"].pop(missing_field)
    torch.save(payload, checkpoint)
    resolved_path = run_directory / "config.resolved.yaml"
    resolved_path.write_text("sentinel: original\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint RNG state"):
        train_from_config(
            config,
            resume_path=checkpoint,
            device_info=select_device("cpu"),
        )

    assert resolved_path.read_text(encoding="utf-8") == "sentinel: original\n"
    assert not (run_directory / "run_metadata.json").exists()


def test_trainer_rejects_nonfinite_optimizer_state_before_overwriting_metadata(
    tmp_path: Path,
) -> None:
    config = _small_config()
    run_directory = tmp_path / "nonfinite-optimizer-state"
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    checkpoint = save_checkpoint(
        run_directory / "checkpoints" / "last.pt",
        model=model,
        optimizer=optimizer,
        config=config,
        step=1,
        device="cpu",
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    first_state = next(iter(payload["optimizer_state"]["state"].values()))
    first_state["exp_avg"].reshape(-1)[0] = float("nan")
    torch.save(payload, checkpoint)
    resolved_path = run_directory / "config.resolved.yaml"
    resolved_path.write_text("sentinel: original\n", encoding="utf-8")

    with pytest.raises(FloatingPointError, match="optimizer_state.*NaN or Inf"):
        train_from_config(
            config,
            resume_path=checkpoint,
            device_info=select_device("cpu"),
        )

    assert resolved_path.read_text(encoding="utf-8") == "sentinel: original\n"
    assert not (run_directory / "run_metadata.json").exists()


def _save_destination_compatible_resume_checkpoint(
    run_directory: Path,
    config,
    *,
    populate_optimizer: bool = False,
) -> Path:
    model = OnlineWorldModel.from_config(config, device="cpu")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    if config.training.rgb_pretrain_steps < 1:
        optimizer.param_groups[0]["lr"] = closed_loop_learning_rate_at_update(
            config,
            causal_update_index=0,
        )
    if populate_optimizer:
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return save_checkpoint(
        run_directory / "checkpoints" / "last.pt",
        model=model,
        optimizer=optimizer,
        config=config,
        step=1,
        metrics={
            "rollout_selection_metric_version": _ROLLOUT_SELECTION_METRIC_VERSION,
            **_validation_protocol_checkpoint_metrics(config),
        },
        device="cpu",
    )


@pytest.mark.parametrize(
    ("corruption", "expected_error"),
    [
        ("model_shape", "size mismatch"),
        ("optimizer_group_count", "different number of parameter groups"),
        ("optimizer_group_schema", "parameter group 0 schema"),
        ("optimizer_betas", "betas must contain two values"),
        ("optimizer_duplicate_id", "parameter IDs must be globally unique"),
        ("optimizer_orphan_state", "does not name a declared parameter ID"),
        ("optimizer_moment_shape", "optimizer moment.*shape"),
        ("optimizer_missing_step", "must contain a scalar Tensor 'step'"),
        ("optimizer_python_step", "must contain a scalar Tensor 'step'"),
        ("scheduler_state", "configured trainer has no scheduler"),
    ],
)
def test_resume_rejects_destination_incompatible_state_before_artifacts(
    tmp_path: Path,
    corruption: str,
    expected_error: str,
) -> None:
    config = _small_config()
    run_directory = tmp_path / f"incompatible-{corruption}"
    checkpoint = _save_destination_compatible_resume_checkpoint(
        run_directory,
        config,
        populate_optimizer=corruption
        in {
            "optimizer_moment_shape",
            "optimizer_missing_step",
            "optimizer_python_step",
        },
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if corruption == "model_shape":
        first_name = next(iter(payload["model_state"]))
        payload["model_state"][first_name] = torch.zeros(1)
    elif corruption == "optimizer_group_count":
        extra_group = deepcopy(payload["optimizer_state"]["param_groups"][0])
        parameter_ids = payload["optimizer_state"]["param_groups"][0]["params"]
        offset = max(parameter_ids) + 1
        extra_group["params"] = [offset + index for index in range(len(parameter_ids))]
        payload["optimizer_state"]["param_groups"].append(extra_group)
    elif corruption == "optimizer_group_schema":
        payload["optimizer_state"]["param_groups"][0].pop("lr")
    elif corruption == "optimizer_betas":
        payload["optimizer_state"]["param_groups"][0]["betas"] = (0.9,)
        disposable_model = OnlineWorldModel.from_config(config, device="cpu")
        disposable_optimizer = torch.optim.AdamW(
            disposable_model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        # PyTorch accepts this malformed group at load time and fails only on
        # the next update; exact-resume preflight must reject it sooner.
        disposable_optimizer.load_state_dict(payload["optimizer_state"])
        for parameter in disposable_model.parameters():
            parameter.grad = torch.ones_like(parameter)
        with pytest.raises((IndexError, RuntimeError, ValueError)):
            disposable_optimizer.step()
    elif corruption == "optimizer_duplicate_id":
        parameters = payload["optimizer_state"]["param_groups"][0]["params"]
        parameters[1] = parameters[0]
    elif corruption == "optimizer_orphan_state":
        payload["optimizer_state"]["state"][999_999] = {}
    elif corruption == "optimizer_moment_shape":
        first_state = next(iter(payload["optimizer_state"]["state"].values()))
        first_state["exp_avg"] = torch.zeros(1)
    elif corruption == "optimizer_missing_step":
        first_state = next(iter(payload["optimizer_state"]["state"].values()))
        first_state.pop("step")
    elif corruption == "optimizer_python_step":
        first_state = next(iter(payload["optimizer_state"]["state"].values()))
        first_state["step"] = float(first_state["step"].item())
    elif corruption == "scheduler_state":
        payload["scheduler_state"] = {"last_epoch": 1}
    else:
        raise AssertionError(f"unknown corruption: {corruption}")
    torch.save(payload, checkpoint)
    resolved_path = run_directory / "config.resolved.yaml"
    resolved_path.write_text("sentinel: original\n", encoding="utf-8")

    with pytest.raises((RuntimeError, ValueError), match=expected_error):
        train_from_config(
            config,
            resume_path=checkpoint,
            device_info=select_device("cpu"),
        )

    assert resolved_path.read_text(encoding="utf-8") == "sentinel: original\n"
    assert not (run_directory / "run_metadata.json").exists()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS private-generator state validation requires an available backend",
)
def test_resume_rejects_invalid_mps_rng_before_artifacts(tmp_path: Path) -> None:
    config = _small_config()
    run_directory = tmp_path / "invalid-mps-rng"
    checkpoint = _save_destination_compatible_resume_checkpoint(run_directory, config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["rng"]["torch_mps"] = torch.zeros(1, dtype=torch.uint8)
    torch.save(payload, checkpoint)
    resolved_path = run_directory / "config.resolved.yaml"
    resolved_path.write_text("sentinel: original\n", encoding="utf-8")

    with pytest.raises(ValueError, match="MPS RNG state is invalid"):
        train_from_config(
            config,
            resume_path=checkpoint,
            device_info=select_device("cpu"),
        )

    assert resolved_path.read_text(encoding="utf-8") == "sentinel: original\n"
    assert not (run_directory / "run_metadata.json").exists()


def test_exact_resume_validates_cuda_rng_device_count_without_global_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _small_config()
    checkpoint = _save_destination_compatible_resume_checkpoint(tmp_path / "cuda-count", config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["rng"]["torch_cuda"] = [payload["rng"]["torch_cpu"].clone()]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    with pytest.raises(ValueError, match="CUDA RNG state device count"):
        validate_exact_resume_state(payload)


@pytest.mark.parametrize(
    ("backend", "expected_error"),
    [("cuda", r"torch_cuda\[0\] is invalid"), ("mps", "MPS RNG state is invalid")],
)
def test_exact_resume_uses_private_generator_to_reject_backend_rng_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    expected_error: str,
) -> None:
    config = _small_config()
    checkpoint = _save_destination_compatible_resume_checkpoint(
        tmp_path / f"invalid-{backend}-generator-state",
        config,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["rng"]["torch_cpu"].clone()
    if backend == "cuda":
        payload["rng"]["torch_cuda"] = [state]
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    else:
        payload["rng"]["torch_mps"] = state
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    real_generator = torch.Generator

    class RejectingGenerator:
        def set_state(self, _state):
            raise RuntimeError("rejected test state")

    def private_generator(*args, **kwargs):
        device = kwargs.get("device", args[0] if args else "cpu")
        if str(device).startswith(backend):
            return RejectingGenerator()
        return real_generator(*args, **kwargs)

    cpu_rng_before = torch.get_rng_state().clone()
    monkeypatch.setattr(torch, "Generator", private_generator)

    with pytest.raises(ValueError, match=expected_error):
        validate_exact_resume_state(payload)

    torch.testing.assert_close(torch.get_rng_state(), cpu_rng_before, rtol=0, atol=0)


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
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    checkpoint = save_checkpoint(
        run_directory / "checkpoints" / "last.pt",
        model=model,
        optimizer=optimizer,
        config=config,
        step=1,
        metrics={
            "rollout_selection_metric_version": _ROLLOUT_SELECTION_METRIC_VERSION,
            **_validation_protocol_checkpoint_metrics(config),
        },
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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    optimizer.param_groups[0]["lr"] = closed_loop_learning_rate_at_update(
        config,
        causal_update_index=0,
    )
    run_directory = tmp_path / "completed"
    checkpoint = save_checkpoint(
        run_directory / "checkpoints" / "last.pt",
        model=model,
        optimizer=optimizer,
        config=config,
        step=1,
        metrics={
            "measurement_handoff_completed": 1.0,
            "rollout_selection_metric_version": _ROLLOUT_SELECTION_METRIC_VERSION,
            **_validation_protocol_checkpoint_metrics(config),
        },
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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    optimizer.param_groups[0]["lr"] = closed_loop_learning_rate_at_update(
        config,
        causal_update_index=0,
    )
    source_run = tmp_path / "source-completed"
    checkpoint = save_checkpoint(
        source_run / "checkpoints" / "last.pt",
        model=model,
        optimizer=optimizer,
        config=config,
        step=1,
        metrics={
            "loss_total": 3.5,
            "measurement_handoff_completed": 1.0,
            "rollout_selection_metric_version": _ROLLOUT_SELECTION_METRIC_VERSION,
            **_validation_protocol_checkpoint_metrics(config),
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


def test_branched_resume_rejects_occupied_destination_without_mutation(
    tmp_path: Path,
) -> None:
    config = _small_config()
    config = replace(
        config,
        project=replace(config.project, output_dir=str(tmp_path / "runs")),
        device=replace(config.device, preference="cpu", closed_loop_preference="same"),
        training=replace(config.training, steps=1, rgb_pretrain_steps=0),
    )
    checkpoint = _save_destination_compatible_resume_checkpoint(
        tmp_path / "source-run",
        config,
    )
    run_name = "20260821-120000-occupied-branch"
    destination = tmp_path / "runs" / run_name
    (destination / "checkpoints").mkdir(parents=True)
    sentinels = {
        destination / "metrics.jsonl": b'{"evidence":"metrics"}\n',
        destination / "checkpoints" / "foreign.pt": b"foreign checkpoint evidence",
        destination / "config.resolved.yaml": b"sentinel: config\n",
        destination / "run_metadata.json": b'{"evidence":"metadata"}\n',
        destination / "training_progress.json": b'{"evidence":"progress"}\n',
        destination / "training_state.json": b'{"state":"stale"}\n',
        destination / "train_summary.json": b'{"evidence":"summary"}\n',
        destination / ".training.lock": b"stale lock evidence\n",
        destination / "unknown-campaign-evidence.bin": b"unknown evidence",
    }
    for path, content in sentinels.items():
        path.write_bytes(content)

    with pytest.raises(
        FileExistsError,
        match="branched exact-resume destination must be absent or empty",
    ):
        train_from_config(
            config,
            run_name=run_name,
            resume_path=checkpoint,
            device_info=select_device("cpu"),
        )

    assert {path: path.read_bytes() for path in sentinels} == sentinels
    assert not (destination / "checkpoints" / "last.pt").exists()


def test_explicit_source_run_name_is_still_an_occupied_branch(tmp_path: Path) -> None:
    config = _small_config()
    config = replace(
        config,
        project=replace(config.project, output_dir=str(tmp_path / "runs")),
        device=replace(config.device, preference="cpu", closed_loop_preference="same"),
        training=replace(config.training, steps=1, rgb_pretrain_steps=0),
    )
    source_run = tmp_path / "runs" / "20260821-120000-source-run"
    checkpoint = _save_destination_compatible_resume_checkpoint(source_run, config)
    sentinels = {
        checkpoint: checkpoint.read_bytes(),
        source_run / "config.resolved.yaml": b"sentinel: source config\n",
        source_run / "metrics.jsonl": b'{"evidence":"source metrics"}\n',
    }
    for path, content in tuple(sentinels.items())[1:]:
        path.write_bytes(content)

    with pytest.raises(
        FileExistsError,
        match="branched exact-resume destination must be absent or empty",
    ):
        train_from_config(
            config,
            run_name=source_run.name,
            resume_path=checkpoint,
            device_info=select_device("cpu"),
        )

    assert {path: path.read_bytes() for path in sentinels} == sentinels


def test_branched_resume_reuses_one_timestamped_destination_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _small_config()
    config = replace(
        config,
        project=replace(config.project, output_dir=str(tmp_path / "runs")),
        device=replace(config.device, preference="cpu", closed_loop_preference="same"),
        training=replace(config.training, steps=1, rgb_pretrain_steps=0),
    )
    checkpoint = _save_destination_compatible_resume_checkpoint(
        tmp_path / "source-run",
        config,
    )
    run_name = "empty-branch-clock-boundary"
    resolution_count = 0

    def timestamp_crossing(path: str | Path) -> Path:
        nonlocal resolution_count
        target = Path(path)
        if target.name.startswith("20260821-"):
            return target
        resolution_count += 1
        return target.with_name(f"20260821-12000{resolution_count}-{target.name}")

    monkeypatch.setattr(
        training_trainer,
        "timestamped_artifact_path",
        timestamp_crossing,
    )
    destination = tmp_path / "runs" / f"20260821-120001-{run_name}"
    destination.mkdir(parents=True)

    result = train_from_config(
        config,
        run_name=run_name,
        resume_path=checkpoint,
        device_info=select_device("cpu"),
    )

    assert Path(result["run_directory"]) == destination
    assert resolution_count == 1
    assert Path(result["last_checkpoint"]).is_file()
    assert (destination / ".training.lock").read_text(encoding="utf-8") == (f"{os.getpid()}\n")


def test_direct_lock_creator_does_not_unlink_inode_when_flock_is_lost(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_directory = tmp_path / "lock-race"
    run_directory.mkdir()

    def lose_lock(_descriptor: int, _operation: int) -> None:
        raise BlockingIOError("another opener won the lock")

    monkeypatch.setattr(training_trainer.fcntl, "flock", lose_lock)

    with pytest.raises(BlockingIOError, match="another opener won"):
        training_trainer._acquire_direct_resume_lock(
            run_directory,
            require_empty_destination=False,
        )

    # The winner may still hold this inode. Unlinking it would let a third
    # writer create and lock a different inode concurrently.
    assert (run_directory / ".training.lock").is_file()


def test_public_in_place_resume_rejects_an_existing_direct_lock_without_mutation(
    tmp_path: Path,
) -> None:
    config = _small_config()
    config = replace(
        config,
        device=replace(config.device, preference="cpu", closed_loop_preference="same"),
        training=replace(config.training, steps=2, rgb_pretrain_steps=0),
    )
    source_run = tmp_path / "locked-in-place-source"
    checkpoint = _save_destination_compatible_resume_checkpoint(source_run, config)
    sentinels = {
        checkpoint: checkpoint.read_bytes(),
        source_run / "config.resolved.yaml": b"sentinel: locked config\n",
        source_run / "metrics.jsonl": b'{"evidence":"locked metrics"}\n',
    }
    for path, content in tuple(sentinels.items())[1:]:
        path.write_bytes(content)
    claim = training_trainer._acquire_direct_resume_lock(
        source_run,
        require_empty_destination=False,
    )
    try:
        with pytest.raises(BlockingIOError):
            train_from_config(
                config,
                resume_path=checkpoint,
                device_info=select_device("cpu"),
            )
        assert {path: path.read_bytes() for path in sentinels} == sentinels
    finally:
        training_trainer._restore_failed_direct_lock_claim(claim)
        claim.handle.close()


def test_branched_resume_rechecks_owned_destination_after_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _small_config()
    config = replace(
        config,
        project=replace(config.project, output_dir=str(tmp_path / "runs")),
        device=replace(config.device, preference="cpu", closed_loop_preference="same"),
        training=replace(config.training, steps=1, rgb_pretrain_steps=0),
    )
    checkpoint = _save_destination_compatible_resume_checkpoint(
        tmp_path / "source-run",
        config,
    )
    run_name = "20260821-120002-raced-branch"
    destination = tmp_path / "runs" / run_name
    sentinel = destination / "concurrent-sentinel.bin"
    sentinel_bytes = b"concurrent evidence must remain authoritative"
    real_validate = training_trainer.validate_exact_resume_state

    def add_evidence_after_early_destination_check(
        payload: Mapping[str, object],
        *,
        require_optimizer_state: bool = True,
    ) -> None:
        real_validate(payload, require_optimizer_state=require_optimizer_state)
        sentinel.write_bytes(sentinel_bytes)

    monkeypatch.setattr(
        training_trainer,
        "validate_exact_resume_state",
        add_evidence_after_early_destination_check,
    )

    with pytest.raises(
        FileExistsError,
        match="owned branched exact-resume destination contains unexpected entries",
    ):
        train_from_config(
            config,
            run_name=run_name,
            resume_path=checkpoint,
            device_info=select_device("cpu"),
        )

    assert sentinel.read_bytes() == sentinel_bytes
    assert {entry.name for entry in destination.iterdir()} == {sentinel.name}


def test_exact_resume_reuses_one_snapshot_when_source_is_replaced_after_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _small_config()
    config = replace(
        config,
        project=replace(config.project, output_dir=str(tmp_path / "runs")),
        device=replace(
            config.device,
            preference="cpu",
            closed_loop_preference="same",
        ),
        training=replace(
            config.training,
            steps=1,
            rgb_pretrain_steps=0,
        ),
    )
    original_model = OnlineWorldModel.from_config(config, device="cpu")
    original_optimizer = torch.optim.AdamW(
        original_model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    original_optimizer.param_groups[0]["lr"] = closed_loop_learning_rate_at_update(
        config,
        causal_update_index=0,
    )
    source_run = tmp_path / "mutable-source"
    checkpoint = save_checkpoint(
        source_run / "checkpoints" / "last.pt",
        model=original_model,
        optimizer=original_optimizer,
        config=config,
        step=1,
        metrics={
            "loss_total": 3.5,
            "measurement_handoff_completed": 1.0,
            "rollout_selection_metric_version": _ROLLOUT_SELECTION_METRIC_VERSION,
            **_validation_protocol_checkpoint_metrics(config),
        },
        device="cpu",
    )
    original_bytes = checkpoint.read_bytes()

    replacement_model = OnlineWorldModel.from_config(config, device="cpu")
    replacement_model.load_state_dict(original_model.state_dict())
    with torch.no_grad():
        next(replacement_model.parameters()).reshape(-1)[0].add_(0.125)
    replacement_optimizer = torch.optim.AdamW(
        replacement_model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    replacement_optimizer.param_groups[0]["lr"] = closed_loop_learning_rate_at_update(
        config,
        causal_update_index=0,
    )
    replacement = save_checkpoint(
        tmp_path / "replacement.pt",
        model=replacement_model,
        optimizer=replacement_optimizer,
        config=config,
        step=1,
        metrics={
            "loss_total": 99.0,
            "measurement_handoff_completed": 1.0,
            "rollout_selection_metric_version": _ROLLOUT_SELECTION_METRIC_VERSION,
            **_validation_protocol_checkpoint_metrics(config),
        },
        device="cpu",
    )
    replacement_bytes = replacement.read_bytes()
    real_validate = training_trainer.validate_exact_resume_state
    source_replaced = False

    def replace_source_after_preflight(
        payload: Mapping[str, object],
        *,
        require_optimizer_state: bool = True,
    ) -> None:
        nonlocal source_replaced
        real_validate(
            payload,
            require_optimizer_state=require_optimizer_state,
        )
        if not source_replaced:
            checkpoint.write_bytes(replacement_bytes)
            source_replaced = True

    monkeypatch.setattr(
        training_trainer,
        "validate_exact_resume_state",
        replace_source_after_preflight,
    )

    result = train_from_config(
        config,
        run_name="immutable-resume-copy",
        resume_path=checkpoint,
        device_info=select_device("cpu"),
    )

    assert source_replaced
    assert checkpoint.read_bytes() == replacement_bytes
    assert Path(result["last_checkpoint"]).read_bytes() == original_bytes
    assert result["last_metrics"]["loss_total"] == 3.5


def test_in_place_no_op_resume_rejects_source_replacement_without_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _small_config()
    config = replace(
        config,
        device=replace(config.device, preference="cpu", closed_loop_preference="same"),
        training=replace(config.training, steps=1, rgb_pretrain_steps=0),
    )
    source_run = tmp_path / "in-place-source"
    checkpoint = _save_destination_compatible_resume_checkpoint(source_run, config)
    original_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    original_payload["metrics"].update(
        {
            "measurement_handoff_completed": 1.0,
            "loss_total": 3.5,
        }
    )
    torch.save(original_payload, checkpoint)

    replacement_payload = deepcopy(original_payload)
    replacement_payload["metrics"]["loss_total"] = 99.0
    replacement_path = tmp_path / "replacement.pt"
    torch.save(replacement_payload, replacement_path)
    replacement_bytes = replacement_path.read_bytes()
    resolved_path = source_run / "config.resolved.yaml"
    resolved_path.write_text("sentinel: original\n", encoding="utf-8")
    real_validate = training_trainer.validate_exact_resume_state
    source_replaced = False

    def replace_source_after_preflight(
        payload: Mapping[str, object],
        *,
        require_optimizer_state: bool = True,
    ) -> None:
        nonlocal source_replaced
        real_validate(payload, require_optimizer_state=require_optimizer_state)
        if not source_replaced:
            checkpoint.write_bytes(replacement_bytes)
            source_replaced = True

    monkeypatch.setattr(
        training_trainer,
        "validate_exact_resume_state",
        replace_source_after_preflight,
    )

    with pytest.raises(ValueError, match="changed after immutable capture"):
        train_from_config(
            config,
            resume_path=checkpoint,
            device_info=select_device("cpu"),
        )

    assert source_replaced
    assert checkpoint.read_bytes() == replacement_bytes
    assert resolved_path.read_text(encoding="utf-8") == "sentinel: original\n"
    assert not (source_run / "run_metadata.json").exists()


def test_in_place_continuing_resume_rejects_source_replacement_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _small_config()
    config = replace(
        config,
        device=replace(config.device, preference="cpu", closed_loop_preference="same"),
        training=replace(config.training, steps=2, rgb_pretrain_steps=0),
    )
    source_run = tmp_path / "in-place-continuing-source"
    checkpoint = _save_destination_compatible_resume_checkpoint(source_run, config)
    original_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    original_payload["metrics"]["measurement_handoff_completed"] = 1.0
    torch.save(original_payload, checkpoint)

    replacement_payload = deepcopy(original_payload)
    replacement_payload["metrics"]["loss_total"] = 99.0
    replacement_path = tmp_path / "continuing-replacement.pt"
    torch.save(replacement_payload, replacement_path)
    replacement_bytes = replacement_path.read_bytes()
    sentinels = {
        source_run / "config.resolved.yaml": b"sentinel: original config\n",
        source_run / "run_metadata.json": b'{"evidence":"original metadata"}\n',
        source_run / "metrics.jsonl": b'{"evidence":"original metrics"}\n',
        source_run / "training_progress.json": b'{"evidence":"original progress"}\n',
    }
    for path, content in sentinels.items():
        path.write_bytes(content)
    real_validate = training_trainer.validate_exact_resume_state
    source_replaced = False

    def replace_source_after_preflight(
        payload: Mapping[str, object],
        *,
        require_optimizer_state: bool = True,
    ) -> None:
        nonlocal source_replaced
        real_validate(payload, require_optimizer_state=require_optimizer_state)
        if not source_replaced:
            checkpoint.write_bytes(replacement_bytes)
            source_replaced = True

    monkeypatch.setattr(
        training_trainer,
        "validate_exact_resume_state",
        replace_source_after_preflight,
    )

    with pytest.raises(ValueError, match="changed after immutable capture"):
        train_from_config(
            config,
            resume_path=checkpoint,
            device_info=select_device("cpu"),
        )

    assert source_replaced
    assert checkpoint.read_bytes() == replacement_bytes
    assert {path: path.read_bytes() for path in sentinels} == sentinels


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
        "fast_radius_derived_depth_enabled",
        "structured_disc_photometric_fast_depth_enabled",
        "structured_disc_photometric_maximum_fit_rms",
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
                structured_disc_fast_depth_enabled=True,
                structured_disc_photometric_fast_depth_enabled=True,
                structured_disc_photometric_maximum_fit_rms=0.03,
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

    derived_enabled = replace(
        config,
        model=replace(
            config.model,
            rgb=replace(config.model.rgb, fast_radius_derived_depth_enabled=True),
        ),
    )
    derived_enabled.validate()
    with pytest.raises(ValueError, match="model"):
        validate_checkpoint_config(payload, derived_enabled)
    with pytest.raises(ValueError, match="model"):
        validate_checkpoint_config(legacy_payload, derived_enabled)

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
        "hypothesis_local_applicability_enabled",
        "hypothesis_minimum_support_count",
        "hypothesis_maximum_evidence_age_seconds",
        "hypothesis_minimum_observability",
        "hypothesis_minimum_confidence_margin",
        "hypothesis_velocity_evidence_weight",
        "hypothesis_velocity_nonregression_gate_enabled",
        "hypothesis_residual_correction_gain_by_axis",
        "hypothesis_robust_influence_delta",
        "hypothesis_composition_step_seconds",
        "hypothesis_online_acceleration_enabled",
        "hypothesis_online_acceleration_minimum_support_count",
        "hypothesis_online_acceleration_maximum_mps2",
        "hypothesis_shared_horizon_rollout_enabled",
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

    online = replace(
        config,
        model=replace(
            config.model,
            rgb=replace(config.model.rgb, temporal_velocity_enabled=True),
        ),
        runtime=replace(
            config.runtime,
            hypothesis_pool_enabled=True,
            hypothesis_local_applicability_enabled=True,
            hypothesis_online_acceleration_enabled=True,
            hypothesis_online_acceleration_minimum_support_count=3,
            hypothesis_online_acceleration_maximum_mps2=12.0,
        ),
    )
    online.validate()
    validate_checkpoint_config({"config": online.to_dict()}, online)
    with pytest.raises(ValueError, match="runtime"):
        validate_checkpoint_config(payload, online)
    with pytest.raises(ValueError, match="runtime"):
        validate_checkpoint_config(legacy_payload, online)


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


def test_learned_correction_axis_support_is_semantic_with_legacy_false() -> None:
    config = _small_config()
    payload = {"config": config.to_dict()}
    legacy_payload = deepcopy(payload)
    legacy_payload["config"]["model"]["filter"].pop("learned_correction_independent_axis_support")

    validate_checkpoint_config(legacy_payload, config)
    corrected = replace(
        config,
        model=replace(
            config.model,
            filter=replace(
                config.model.filter,
                learned_correction_independent_axis_support=True,
            ),
        ),
    )
    corrected.validate()
    with pytest.raises(ValueError, match="model"):
        validate_checkpoint_config(legacy_payload, corrected)
    with pytest.raises(ValueError, match="model"):
        validate_checkpoint_config(payload, corrected)


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
