from __future__ import annotations

from dataclasses import asdict, replace

import pytest
import torch

from world_model.belief import BeliefFactory
from world_model.evaluation.neural_adaptive_physics import (
    NEURAL_ADAPTIVE_PHYSICS_SCHEMA,
    load_neural_physics_adapter,
    multihorizon_stability_loss,
    neural_adaptive_protocol_sha256,
    physical_rollout_signature,
    synthetic_physics_batch,
)
from world_model.evaluation.neural_long_horizon import (
    NEURAL_LONG_HORIZON_SCHEMA,
    load_long_horizon_physics_adapter,
    neural_long_horizon_protocol_sha256,
)
from world_model.identification import (
    EVIDENCE_FEATURE_DIM,
    EvidenceTransformerConfig,
    LongHorizonAdapterConfig,
    LongHorizonPhysicsAdapter,
    NeuralPhysicsAdapter,
    PhysicsEvidenceKind,
    causal_parameter_baseline,
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


def test_multihorizon_stability_signature_is_differentiable_and_exact_at_truth() -> None:
    target = torch.tensor(
        [[0.0, -2.0, torch.logit(torch.tensor(0.7)), torch.logit(torch.tensor(0.2))]]
    )
    predicted = (target + torch.tensor([[0.1, -0.1, 0.05, -0.05]])).requires_grad_()
    horizons = torch.tensor([0.5, 2.0, 4.0, 8.0, 12.0])

    signature = physical_rollout_signature(predicted, horizons)
    loss = multihorizon_stability_loss(predicted, target)
    exact = multihorizon_stability_loss(target, target)
    loss.backward()

    assert signature.shape == (1, 5, 6)
    assert float(loss.detach()) > 0.0
    assert float(exact.detach()) == 0.0
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()
    assert float(predicted.grad.abs().sum().detach()) > 0.0
    with pytest.raises(ValueError, match="positive vector"):
        physical_rollout_signature(predicted.detach(), torch.empty(0))


def test_long_horizon_adapter_starts_at_permutation_invariant_causal_baseline() -> None:
    batch = synthetic_physics_batch(
        8,
        generator=torch.Generator().manual_seed(62),
    )
    model = LongHorizonPhysicsAdapter()
    order = torch.tensor([4, 0, 3, 2, 1])

    baseline = causal_parameter_baseline(batch.evidence, batch.valid, model.config)
    prediction = model(batch.evidence, batch.valid)
    permuted = model(batch.evidence[:, order], batch.valid[:, order])

    assert model.parameter_count() == 48_920
    torch.testing.assert_close(prediction.normalized_mean, baseline)
    torch.testing.assert_close(prediction.normalized_mean, permuted.normalized_mean)
    with pytest.raises(ValueError, match="residual bound"):
        LongHorizonAdapterConfig(max_normalized_residual=0.0).validate()


def test_long_horizon_checkpoint_roundtrip_and_protocol_binding(tmp_path) -> None:
    torch.manual_seed(71)
    model = LongHorizonPhysicsAdapter().eval()
    batch = synthetic_physics_batch(
        3,
        generator=torch.Generator().manual_seed(72),
    )
    checkpoint = tmp_path / "long-horizon.pt"
    payload = {
        "schema": NEURAL_LONG_HORIZON_SCHEMA,
        "model": "LongHorizonPhysicsAdapter",
        "config": asdict(model.long_horizon_config),
        "learned_parameter_count": model.parameter_count(),
        "state_dict": model.state_dict(),
    }
    torch.save(payload, checkpoint)

    restored = load_long_horizon_physics_adapter(checkpoint)
    expected = model(batch.evidence, batch.valid)
    actual = restored(batch.evidence, batch.valid)
    torch.testing.assert_close(actual.belief_values, expected.belief_values)
    torch.testing.assert_close(actual.belief_log_variance, expected.belief_log_variance)

    first = neural_long_horizon_protocol_sha256(
        incumbent_checkpoint_sha256="a" * 64,
        training_steps=8_192,
        batch_size=128,
    )
    assert first == neural_long_horizon_protocol_sha256(
        incumbent_checkpoint_sha256="a" * 64,
        training_steps=8_192,
        batch_size=128,
    )
    assert first != neural_long_horizon_protocol_sha256(
        incumbent_checkpoint_sha256="b" * 64,
        training_steps=8_192,
        batch_size=128,
    )
    with pytest.raises(ValueError, match="positive"):
        neural_long_horizon_protocol_sha256(
            incumbent_checkpoint_sha256="a" * 64,
            training_steps=0,
            batch_size=128,
        )

    payload["schema"] = "wrong-schema"
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="schema"):
        load_long_horizon_physics_adapter(checkpoint)
