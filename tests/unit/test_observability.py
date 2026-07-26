from __future__ import annotations

import torch

from world_model.belief import BeliefFactory, MotionMode
from world_model.fusion import AssociationResult
from world_model.identification import (
    Observability,
    ObservabilityEstimator,
    ParameterUpdaterConfig,
    RecurrentParameterUpdater,
)
from world_model.observations import InnovationSet


def _case(mode: MotionMode, *, ambiguous: bool = False) -> tuple[object, object, object]:
    belief = BeliefFactory(max_objects=1).create()
    mode_logits = torch.full_like(belief.objects.motion_mode_logits, -5.0)
    mode_logits[..., mode] = 5.0
    objects = belief.objects.replace(
        active=torch.tensor([[True]]),
        object_id=torch.tensor([[0]]),
        velocity=torch.tensor([[[1.0, 0.0, 0.0]]]),
        visibility_logit=torch.tensor([[5.0]]),
        age_steps=torch.tensor([[10]]),
        motion_mode_logits=mode_logits,
    )
    belief = belief.replace(objects=objects)
    association = AssociationResult(
        belief_indices=torch.tensor([[0]]),
        measurement_indices=torch.tensor([[0]]),
        pair_mask=torch.tensor([[True]]),
        pair_cost=torch.tensor([[0.0]]),
        unmatched_beliefs=torch.tensor([[False]]),
        unmatched_measurements=torch.tensor([[False]]),
        ambiguous=torch.tensor([[ambiguous]]),
    )
    residual = torch.zeros(1, 1, 6)
    innovation = InnovationSet(
        modality="debug_oracle",
        residual=residual,
        whitened_residual=residual,
        innovation_norm=torch.zeros(1, 1),
        belief_indices=torch.tensor([[0]]),
        measurement_indices=torch.tensor([[0]]),
        pair_mask=torch.tensor([[True]]),
        log_likelihood=torch.zeros(1, 1),
        modality_index=torch.ones(1, 1, dtype=torch.int64),
        event_features=torch.zeros(1, 1, 5),
    )
    return belief, innovation, association


def test_free_motion_observes_drag_but_not_restitution_or_mass() -> None:
    belief, innovation, association = _case(MotionMode.FREE)
    observability = ObservabilityEstimator()(belief, innovation, association)
    assert observability.drag.item() > 0.5
    assert observability.restitution.item() < 1.0e-3
    assert observability.mass_ratio.item() < 1.0e-3


def test_collision_observes_restitution_and_mass_ratio() -> None:
    belief, innovation, association = _case(MotionMode.COLLISION)
    observability = ObservabilityEstimator()(belief, innovation, association)
    assert observability.restitution.item() > 0.5
    assert observability.mass_ratio.item() > 0.5
    assert observability.drag.item() < 1.0e-3


def test_ambiguous_association_gates_all_parameter_updates() -> None:
    belief, innovation, association = _case(MotionMode.COLLISION, ambiguous=True)
    observability = ObservabilityEstimator()(belief, innovation, association)
    assert torch.equal(observability.stacked(), torch.zeros_like(observability.stacked()))


def test_online_drag_update_is_bounded_and_requires_observability() -> None:
    factory = BeliefFactory(max_objects=1, parameter_memory_dim=8)
    belief, innovation, association = _case(MotionMode.FREE)
    # Use a dimensionally compatible belief while preserving the case state.
    compatible = factory.create()
    belief = compatible.replace(
        objects=compatible.objects.replace(
            active=belief.objects.active,
            object_id=belief.objects.object_id,
            velocity=belief.objects.velocity,
            visibility_logit=belief.objects.visibility_logit,
            age_steps=belief.objects.age_steps,
            motion_mode_logits=belief.objects.motion_mode_logits,
        )
    )
    # Measured velocity lags the prediction, supporting greater drag.
    innovation.residual[..., 3] = -0.5
    observable = Observability(
        mass_ratio=torch.zeros(1, 1),
        restitution=torch.zeros(1, 1),
        drag=torch.ones(1, 1),
        friction=torch.zeros(1, 1),
        geometry=torch.zeros(1, 1),
    )
    updater = RecurrentParameterUpdater(
        ParameterUpdaterConfig(hidden_dim=8, slow_learning_rate=0.1)
    )
    updated = updater.update(belief, innovation, association, observable)
    assert updated.objects.log_drag.item() > belief.objects.log_drag.item()
    assert torch.equal(
        updated.objects.restitution_logit,
        belief.objects.restitution_logit,
    )
    unobservable = Observability(*[torch.zeros(1, 1) for _ in range(5)])
    unchanged = updater.update(belief, innovation, association, unobservable)
    assert torch.equal(unchanged.objects.log_drag, belief.objects.log_drag)
    assert torch.equal(
        unchanged.objects.parameter_memory,
        belief.objects.parameter_memory,
    )


def test_online_restitution_update_is_bounded_and_collision_gated() -> None:
    factory = BeliefFactory(max_objects=1, parameter_memory_dim=8)
    belief, innovation, association = _case(MotionMode.COLLISION)
    compatible = factory.create()
    belief = compatible.replace(
        objects=compatible.objects.replace(
            active=belief.objects.active,
            object_id=belief.objects.object_id,
            velocity=belief.objects.velocity,
            visibility_logit=belief.objects.visibility_logit,
            age_steps=belief.objects.age_steps,
            motion_mode_logits=belief.objects.motion_mode_logits,
        )
    )
    # A post-impact measurement retaining more along-motion velocity supports
    # greater restitution.
    innovation.residual[..., 3] = 0.5
    observable = Observability(
        mass_ratio=torch.zeros(1, 1),
        restitution=torch.ones(1, 1),
        drag=torch.zeros(1, 1),
        friction=torch.zeros(1, 1),
        geometry=torch.zeros(1, 1),
    )
    updater = RecurrentParameterUpdater(
        ParameterUpdaterConfig(hidden_dim=8, slow_learning_rate=0.1)
    )
    updated = updater.update(belief, innovation, association, observable)
    assert updated.objects.restitution_logit.item() > belief.objects.restitution_logit.item()
    assert torch.equal(updated.objects.log_drag, belief.objects.log_drag)

    drag_only = Observability(
        mass_ratio=torch.zeros(1, 1),
        restitution=torch.zeros(1, 1),
        drag=torch.ones(1, 1),
        friction=torch.zeros(1, 1),
        geometry=torch.zeros(1, 1),
    )
    gated = updater.update(belief, innovation, association, drag_only)
    assert torch.equal(gated.objects.restitution_logit, belief.objects.restitution_logit)
