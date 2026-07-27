from __future__ import annotations

from dataclasses import fields, replace

import pytest
import torch

from world_model.abstractions import (
    AbstractionKind,
    AbstractionReason,
    PredictiveAbstractionRouter,
    PredictiveTokenType,
    WorldBeliefTokenizer,
)
from world_model.belief import BeliefFactory, MotionMode
from world_model.runtime import OnlineWorldModel
from world_model.utils.config import load_config


def _active_belief():
    belief = BeliefFactory(
        max_objects=3,
        geometry_dim=4,
        appearance_dim=5,
        residual_dynamics_dim=3,
        modal_count=2,
        modal_dim=2,
        parameter_memory_dim=6,
        global_code_dim=4,
    ).create(batch_size=2, timestamp=torch.tensor([0.25, 0.5]))
    objects = belief.objects.clone()
    objects.active[:, :2] = True
    objects.object_id[:, 0] = torch.tensor([3, 8])
    objects.object_id[:, 1] = torch.tensor([5, 9])
    objects.position[:, :2].normal_()
    objects.velocity[:, :2].normal_()
    objects.geometry[:, :2].uniform_(0.1, 0.5)
    objects.appearance[:, :2].normal_()
    objects.residual_dynamics[:, :2].normal_()
    objects.modal_state[:, :2].normal_()
    objects.modal_frequency[:, :2].uniform_(0.0, 2.0)
    objects.modal_decay_raw[:, :2].normal_()
    objects.parameter_memory[:, :2].normal_()
    objects.age_steps[:, :2] = torch.tensor([[4, 6], [7, 9]])
    objects.missed_steps[:, :2] = torch.tensor([[0, 1], [2, 0]])
    objects.motion_mode_logits.zero_()
    objects.motion_mode_logits[..., MotionMode.FREE] = 1.0
    objects.motion_mode_logits[0, 1].zero_()
    objects.motion_mode_logits[0, 1, MotionMode.COLLISION] = 3.0
    objects.motion_mode_logits[1, 0].zero_()
    objects.motion_mode_logits[1, 0, MotionMode.GROUND_CONTACT] = 3.0
    return replace(
        belief,
        objects=objects,
        gravity=torch.tensor([[0.0, -9.7, 0.0], [0.0, -9.9, 0.0]]),
        global_code=torch.randn_like(belief.global_code),
        global_log_variance=torch.randn_like(belief.global_log_variance),
        next_object_id=torch.tensor([6, 10]),
    ).validate()


def test_router_uses_point_trajectory_until_contact_requires_sphere() -> None:
    belief = _active_belief()
    assignment = PredictiveAbstractionRouter().route(belief)

    assert assignment.kind[0, 0] == AbstractionKind.POINT_TRAJECTORY
    assert assignment.reason[0, 0] == AbstractionReason.FREE_MOTION
    assert assignment.kind[0, 1] == AbstractionKind.RIGID_SPHERE
    assert assignment.reason[0, 1] == AbstractionReason.CONTACT_OR_EVENT
    assert assignment.kind[1, 0] == AbstractionKind.RIGID_SPHERE
    assert assignment.complexity_cost[0, 0] < assignment.complexity_cost[0, 1]
    assert assignment.confidence[~belief.objects.active].eq(0).all()


def test_belief_tokenization_is_typed_masked_and_reversible() -> None:
    belief = _active_belief()
    tokenizer = WorldBeliefTokenizer()
    tokens = tokenizer.encode(belief)
    restored = tokenizer.decode(tokens, belief)

    assert tokens.values.shape[:2] == (2, 11)
    assert tokens.token_type[:2].tolist() == [
        PredictiveTokenType.SCENE,
        PredictiveTokenType.CAMERA,
    ]
    assert tokens.valid_mask[:, :2].all()
    assert tokens.valid_mask[:, 2:8].all()
    assert not tokens.valid_mask[:, 8:].any()
    assert tokens.object_id[:, 2].tolist() == [3, 8]
    assert tokens.abstraction_kind[0, 5] == AbstractionKind.RIGID_SPHERE

    for item in fields(belief.objects):
        torch.testing.assert_close(
            getattr(restored.objects, item.name),
            getattr(belief.objects, item.name),
            rtol=0,
            atol=0,
        )
    for item in fields(belief.camera):
        torch.testing.assert_close(
            getattr(restored.camera, item.name),
            getattr(belief.camera, item.name),
            rtol=0,
            atol=0,
        )
    for name in (
        "timestamp",
        "gravity",
        "global_code",
        "global_log_variance",
        "next_object_id",
    ):
        torch.testing.assert_close(
            getattr(restored, name),
            getattr(belief, name),
            rtol=0,
            atol=0,
        )
    assert restored.active_modalities == belief.active_modalities
    assert restored.metadata == belief.metadata


def test_tokenizer_rejects_incompatible_belief_schema() -> None:
    belief = _active_belief()
    tokens = WorldBeliefTokenizer().encode(belief)
    incompatible = BeliefFactory(max_objects=2).create(batch_size=2)

    with pytest.raises(ValueError, match="schema"):
        WorldBeliefTokenizer().decode(tokens, incompatible)


def test_runtime_exposes_derived_tokens_without_state_dict_changes() -> None:
    config = load_config("configs/tiny_overfit.yaml")
    model = OnlineWorldModel.from_config(config, device="cpu")
    model.state.belief = model.belief_factory.create(active_modalities=("rgb",))

    keys_before = tuple(model.state_dict())
    assignments = model.predictive_abstractions()
    tokens = model.predictive_tokens()

    assert not assignments.active_mask.any()
    assert tokens.valid_mask[:, :2].all()
    assert tuple(model.state_dict()) == keys_before
