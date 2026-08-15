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
from world_model.observations import DirectVelocityEvidence, InnovationSet


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


def test_interval_collision_observes_restitution_after_endpoint_returns_free() -> None:
    belief, innovation, association = _case(MotionMode.FREE)
    observability = ObservabilityEstimator()(
        belief,
        innovation,
        association,
        interval_collision_mask=torch.tensor([[True]]),
    )

    assert observability.restitution.item() > 0.5
    assert observability.mass_ratio.item() > 0.5


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


def test_rgb_drag_signal_uses_causal_prior_innovation_not_corrected_posterior() -> None:
    factory = BeliefFactory(max_objects=1, parameter_memory_dim=8)
    belief, _, association = _case(MotionMode.FREE)
    compatible = factory.create()
    # This represents the post-correction state: the fast posterior has
    # already moved to the RGB measurement at x=0.5.
    belief = compatible.replace(
        objects=compatible.objects.replace(
            active=belief.objects.active,
            object_id=belief.objects.object_id,
            position=torch.tensor([[[0.5, 0.0, 0.0]]]),
            velocity=belief.objects.velocity,
            visibility_logit=belief.objects.visibility_logit,
            age_steps=belief.objects.age_steps,
            motion_mode_logits=belief.objects.motion_mode_logits,
        )
    )
    innovation = InnovationSet(
        modality="rgb",
        residual=torch.zeros(1, 1, 7),
        whitened_residual=torch.zeros(1, 1, 7),
        innovation_norm=torch.zeros(1, 1),
        belief_indices=torch.tensor([[0]]),
        measurement_indices=torch.tensor([[0]]),
        pair_mask=torch.tensor([[True]]),
        log_likelihood=torch.zeros(1, 1),
        modality_index=torch.zeros(1, 1, dtype=torch.int64),
        event_features=torch.zeros(1, 1, 5),
        auxiliary={
            "measured_world_position": torch.tensor([[[0.5, 0.0, 0.0]]]),
            "predicted_world_position": torch.tensor([[[1.0, 0.0, 0.0]]]),
        },
    )
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


@torch.no_grad()
def test_source_bound_roi_position_requires_independent_axis_for_drag_signal() -> None:
    belief, association = _compatible_parameter_belief(MotionMode.FREE)
    position_innovation = _rgb_position_innovation(
        measured_x=-1000.0,
        predicted_x=1000.0,
        measured_variance=1.0e-12,
    )
    position_innovation.auxiliary["measured_source_bound"] = torch.tensor([[True]])

    missing_provenance = RecurrentParameterUpdater._analytic_signals(
        belief,
        position_innovation,
        association,
        torch.zeros_like(belief.objects.velocity),
        elapsed_seconds=torch.tensor([0.05]),
        predicted_belief=belief,
    )
    torch.testing.assert_close(missing_provenance[..., 2], torch.zeros(1, 1))

    position_innovation.auxiliary["measured_world_position_independent_axis_mask"] = torch.tensor(
        [[[False, False, False]]]
    )
    explicitly_unsupported = RecurrentParameterUpdater._analytic_signals(
        belief,
        position_innovation,
        association,
        torch.zeros_like(belief.objects.velocity),
        elapsed_seconds=torch.tensor([0.05]),
        predicted_belief=belief,
    )
    torch.testing.assert_close(explicitly_unsupported[..., 2], torch.zeros(1, 1))

    position_innovation.auxiliary["measured_world_position_independent_axis_mask"] = torch.tensor(
        [[[True, False, False]]]
    )
    supported = RecurrentParameterUpdater._analytic_signals(
        belief,
        position_innovation,
        association,
        torch.zeros_like(belief.objects.velocity),
        elapsed_seconds=torch.tensor([0.05]),
        predicted_belief=belief,
    )
    assert supported[..., 2].item() > 0.0


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


def _rgb_position_innovation(
    *,
    measured_x: float,
    predicted_x: float,
    measured_variance: float,
) -> InnovationSet:
    return InnovationSet(
        modality="rgb",
        residual=torch.zeros(1, 1, 7),
        whitened_residual=torch.zeros(1, 1, 7),
        innovation_norm=torch.zeros(1, 1),
        belief_indices=torch.tensor([[0]]),
        measurement_indices=torch.tensor([[0]]),
        pair_mask=torch.tensor([[True]]),
        log_likelihood=torch.zeros(1, 1),
        modality_index=torch.zeros(1, 1, dtype=torch.int64),
        event_features=torch.zeros(1, 1, 5),
        auxiliary={
            "measured_world_position": torch.tensor([[[measured_x, 0.0, 0.0]]]),
            "predicted_world_position": torch.tensor([[[predicted_x, 0.0, 0.0]]]),
            "measured_world_position_log_variance": torch.tensor(
                [[[measured_variance, measured_variance, measured_variance]]]
            ).log(),
            "measured_position_confidence": torch.ones(1, 1),
        },
    )


def _compatible_parameter_belief(mode: MotionMode) -> tuple[object, object]:
    factory = BeliefFactory(max_objects=1, parameter_memory_dim=8)
    source, _, association = _case(mode)
    belief = factory.create()
    belief = belief.replace(
        objects=belief.objects.replace(
            active=source.objects.active,
            object_id=source.objects.object_id,
            velocity=source.objects.velocity,
            visibility_logit=source.objects.visibility_logit,
            age_steps=source.objects.age_steps,
            motion_mode_logits=source.objects.motion_mode_logits,
            fast_log_variance=torch.full_like(belief.objects.fast_log_variance, -20.0),
        )
    )
    return belief, association


def test_rgb_drag_signal_is_rate_normalized_for_asynchronous_packets() -> None:
    belief, association = _compatible_parameter_belief(MotionMode.FREE)
    observable = Observability(
        mass_ratio=torch.zeros(1, 1),
        restitution=torch.zeros(1, 1),
        drag=torch.ones(1, 1),
        friction=torch.zeros(1, 1),
        geometry=torch.zeros(1, 1),
    )
    updater = RecurrentParameterUpdater(
        ParameterUpdaterConfig(
            hidden_dim=8,
            slow_learning_rate=0.1,
            analytic_signal_scale=1.0,
        )
    )
    short = updater.update(
        belief,
        _rgb_position_innovation(
            measured_x=0.95,
            predicted_x=1.0,
            measured_variance=1.0e-12,
        ),
        association,
        observable,
        elapsed_seconds=torch.tensor([0.1]),
        predicted_belief=belief,
    )
    long = updater.update(
        belief,
        _rgb_position_innovation(
            measured_x=0.90,
            predicted_x=1.0,
            measured_variance=4.0e-12,
        ),
        association,
        observable,
        elapsed_seconds=torch.tensor([0.2]),
        predicted_belief=belief,
    )

    torch.testing.assert_close(short.objects.log_drag, long.objects.log_drag)


def test_rgb_drag_signal_respects_world_position_uncertainty() -> None:
    belief, association = _compatible_parameter_belief(MotionMode.FREE)
    observable = Observability(
        mass_ratio=torch.zeros(1, 1),
        restitution=torch.zeros(1, 1),
        drag=torch.ones(1, 1),
        friction=torch.zeros(1, 1),
        geometry=torch.zeros(1, 1),
    )
    updater = RecurrentParameterUpdater(
        ParameterUpdaterConfig(
            hidden_dim=8,
            slow_learning_rate=0.1,
            analytic_signal_scale=1.0,
        )
    )
    low_variance = updater.update(
        belief,
        _rgb_position_innovation(
            measured_x=0.95,
            predicted_x=1.0,
            measured_variance=1.0e-8,
        ),
        association,
        observable,
        elapsed_seconds=torch.tensor([0.1]),
        predicted_belief=belief,
    )
    high_variance = updater.update(
        belief,
        _rgb_position_innovation(
            measured_x=0.95,
            predicted_x=1.0,
            measured_variance=100.0,
        ),
        association,
        observable,
        elapsed_seconds=torch.tensor([0.1]),
        predicted_belief=belief,
    )

    low_delta = low_variance.objects.log_drag - belief.objects.log_drag
    high_delta = high_variance.objects.log_drag - belief.objects.log_drag
    assert low_delta.item() > 0.0
    assert high_delta.item() >= 0.0
    assert high_delta.item() < low_delta.item() * 1.0e-3


def test_rgb_position_displacement_does_not_fabricate_restitution_velocity() -> None:
    belief, association = _compatible_parameter_belief(MotionMode.COLLISION)
    observable = Observability(
        mass_ratio=torch.zeros(1, 1),
        restitution=torch.ones(1, 1),
        drag=torch.zeros(1, 1),
        friction=torch.zeros(1, 1),
        geometry=torch.zeros(1, 1),
    )
    updater = RecurrentParameterUpdater(
        ParameterUpdaterConfig(
            hidden_dim=8,
            slow_learning_rate=0.1,
            analytic_signal_scale=1.0,
        )
    )

    updated = updater.update(
        belief,
        _rgb_position_innovation(
            measured_x=1.1,
            predicted_x=1.0,
            measured_variance=1.0e-8,
        ),
        association,
        observable,
        elapsed_seconds=torch.tensor([0.1]),
        predicted_belief=belief,
    )

    torch.testing.assert_close(
        updated.objects.restitution_logit,
        belief.objects.restitution_logit,
    )


def test_direct_post_event_velocity_can_update_restitution() -> None:
    belief, association = _compatible_parameter_belief(MotionMode.COLLISION)
    observable = Observability(
        mass_ratio=torch.zeros(1, 1),
        restitution=torch.ones(1, 1),
        drag=torch.zeros(1, 1),
        friction=torch.zeros(1, 1),
        geometry=torch.zeros(1, 1),
    )
    evidence = DirectVelocityEvidence(
        velocity=torch.tensor([[[1.5, 0.0, 0.0]]]),
        log_variance=torch.full((1, 1, 3), -20.0),
        valid_mask=torch.tensor([[True]]),
        confidence=torch.ones(1, 1),
        axis_valid_mask=torch.tensor([[[True, False, False]]]),
    )
    updater = RecurrentParameterUpdater(
        ParameterUpdaterConfig(
            hidden_dim=8,
            slow_learning_rate=0.1,
            analytic_signal_scale=1.0,
        )
    )

    updated = updater.update(
        belief,
        _rgb_position_innovation(
            measured_x=1.0,
            predicted_x=1.0,
            measured_variance=1.0,
        ),
        association,
        observable,
        elapsed_seconds=torch.tensor([0.1]),
        predicted_belief=belief,
        direct_velocity_evidence=evidence,
    )

    assert updated.objects.restitution_logit.item() > belief.objects.restitution_logit.item()


@torch.no_grad()
def test_present_but_unsupported_direct_velocity_fails_closed_without_position_fallback() -> None:
    belief, association = _compatible_parameter_belief(MotionMode.FREE)
    innovation = _rgb_position_innovation(
        measured_x=0.0,
        predicted_x=1.0,
        measured_variance=1.0e-12,
    )
    evidence = DirectVelocityEvidence(
        velocity=torch.full_like(belief.objects.velocity, 1000.0),
        log_variance=torch.full_like(belief.objects.velocity, -20.0),
        valid_mask=torch.tensor([[False]]),
        confidence=torch.ones(1, 1),
        axis_valid_mask=torch.zeros(1, 1, 3, dtype=torch.bool),
    )

    position_fallback = RecurrentParameterUpdater._analytic_signals(
        belief,
        innovation,
        association,
        torch.zeros_like(belief.objects.velocity),
        elapsed_seconds=torch.tensor([0.1]),
        predicted_belief=belief,
    )
    unsupported_direct = RecurrentParameterUpdater._analytic_signals(
        belief,
        innovation,
        association,
        torch.zeros_like(belief.objects.velocity),
        elapsed_seconds=torch.tensor([0.1]),
        predicted_belief=belief,
        direct_velocity_evidence=evidence,
    )

    assert position_fallback[..., 2].item() > 0.0
    torch.testing.assert_close(unsupported_direct, torch.zeros_like(unsupported_direct))
