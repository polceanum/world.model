from __future__ import annotations

from dataclasses import asdict, replace

import pytest
import torch

from world_model.belief import BeliefFactory
from world_model.evaluation.neural_adaptive_physics import (
    NEURAL_ADAPTIVE_PHYSICS_SCHEMA,
    load_neural_physics_adapter,
    neural_adaptive_protocol_sha256,
    synthetic_physics_batch,
)
from world_model.identification import (
    EVIDENCE_FEATURE_DIM,
    EvidenceTransformerConfig,
    NeuralPhysicsAdapter,
    PhysicsEvidenceKind,
    event_transition_token,
    normalized_parameter_targets,
)


def test_compact_configuration_rejects_invalid_attention_shapes() -> None:
    with pytest.raises(ValueError, match="divide"):
        EvidenceTransformerConfig(width=47).validate()
    with pytest.raises(ValueError, match="below one"):
        EvidenceTransformerConfig(restitution_bounds=(0.2, 1.0)).validate()


def test_synthetic_public_evidence_is_deterministic_and_truth_is_separate() -> None:
    first = synthetic_physics_batch(
        8,
        generator=torch.Generator().manual_seed(17),
    )
    second = synthetic_physics_batch(
        8,
        generator=torch.Generator().manual_seed(17),
    )
    different = synthetic_physics_batch(
        8,
        generator=torch.Generator().manual_seed(18),
    )

    assert first.evidence.shape == (8, 5, EVIDENCE_FEATURE_DIM)
    assert first.target_belief_values.shape == (8, 4)
    assert torch.equal(first.evidence, second.evidence)
    assert torch.equal(first.target_belief_values, second.target_belief_values)
    assert not torch.equal(first.evidence, different.evidence)
    assert torch.isfinite(first.evidence).all()
    assert first.valid.all()

    assert neural_adaptive_protocol_sha256(training_steps=2048, batch_size=128) == (
        neural_adaptive_protocol_sha256(training_steps=2048, batch_size=128)
    )
    assert neural_adaptive_protocol_sha256(training_steps=2048, batch_size=128) != (
        neural_adaptive_protocol_sha256(training_steps=1024, batch_size=128)
    )


def test_event_tokens_ignore_absolute_scene_time() -> None:
    before = torch.tensor([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]])
    after = torch.tensor([[[0.2, 0.0, 0.0], [0.35, 0.0, 0.0], [0.5, 0.0, 0.0]]])
    before_time = torch.tensor([[-0.2, -0.1, 0.0]])
    after_time = torch.tensor([[0.0, 0.1, 0.2]])
    impulse = torch.tensor([[0.5, 0.0, 0.0]])

    origin = event_transition_token(
        PhysicsEvidenceKind.KNOWN_IMPULSE,
        before,
        before_time,
        after,
        after_time,
        known_impulse_world=impulse,
    )
    shifted = event_transition_token(
        PhysicsEvidenceKind.KNOWN_IMPULSE,
        before,
        before_time + 7.0,
        after,
        after_time + 7.0,
        known_impulse_world=impulse,
    )

    torch.testing.assert_close(origin, shifted, rtol=1.0e-6, atol=5.0e-7)


def test_transformer_is_compact_permutation_invariant_and_learnable() -> None:
    torch.manual_seed(31)
    model = NeuralPhysicsAdapter()
    batch = synthetic_physics_batch(
        16,
        generator=torch.Generator().manual_seed(32),
    )
    target = normalized_parameter_targets(batch.target_belief_values, model.config)
    order = torch.tensor([4, 1, 3, 0, 2])

    before = model(batch.evidence, batch.valid).normalized_mean
    permuted = model(batch.evidence[:, order], batch.valid[:, order]).normalized_mean
    torch.testing.assert_close(before, permuted, rtol=1.0e-5, atol=1.0e-6)
    assert model.parameter_count() == 48_920
    assert sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()) < (
        1 << 20
    )

    output_nonzero_before = int(torch.count_nonzero(model.mean_head.weight))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    loss = torch.nn.functional.smooth_l1_loss(before, target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    assert output_nonzero_before == 0
    assert int(torch.count_nonzero(model.mean_head.weight)) > 0
    assert model.input_projection.weight.grad is not None
    assert torch.isfinite(model.input_projection.weight.grad).all()


def test_runtime_adaptation_updates_only_active_object_beliefs() -> None:
    torch.manual_seed(41)
    model = NeuralPhysicsAdapter()
    batch = synthetic_physics_batch(
        1,
        generator=torch.Generator().manual_seed(42),
    )
    belief = BeliefFactory(max_objects=2).create(batch_size=1)
    objects = belief.objects.clone()
    objects.active[0, 0] = True
    objects.object_id[0, 0] = 7
    belief = replace(belief, objects=objects).validate()
    evidence = torch.zeros((1, 2, 5, EVIDENCE_FEATURE_DIM))
    valid = torch.zeros((1, 2, 5), dtype=torch.bool)
    evidence[0, 0] = batch.evidence[0]
    valid[0, 0] = True
    source = belief.clone()

    adapted = model.adapt_belief(belief, evidence, valid)

    assert torch.equal(belief.objects.log_mass, source.objects.log_mass)
    assert not torch.equal(adapted.objects.log_mass[0, 0], source.objects.log_mass[0, 0])
    torch.testing.assert_close(
        adapted.objects.log_mass[0, 1],
        source.objects.log_mass[0, 1],
    )
    torch.testing.assert_close(
        adapted.objects.slow_log_variance[0, 1],
        source.objects.slow_log_variance[0, 1],
    )


def test_weights_only_checkpoint_roundtrip_and_schema_validation(tmp_path) -> None:
    torch.manual_seed(51)
    model = NeuralPhysicsAdapter().eval()
    batch = synthetic_physics_batch(
        3,
        generator=torch.Generator().manual_seed(52),
    )
    checkpoint = tmp_path / "model.pt"
    payload = {
        "schema": NEURAL_ADAPTIVE_PHYSICS_SCHEMA,
        "model": "NeuralPhysicsAdapter",
        "config": asdict(model.config),
        "learned_parameter_count": model.parameter_count(),
        "state_dict": model.state_dict(),
    }
    torch.save(payload, checkpoint)

    restored = load_neural_physics_adapter(checkpoint)
    expected = model(batch.evidence, batch.valid)
    actual = restored(batch.evidence, batch.valid)
    torch.testing.assert_close(actual.belief_values, expected.belief_values)
    torch.testing.assert_close(actual.belief_log_variance, expected.belief_log_variance)

    payload["schema"] = "wrong_schema"
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="schema"):
        load_neural_physics_adapter(checkpoint)
