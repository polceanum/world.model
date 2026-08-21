from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import Tensor

import world_model.training.loop as training_loop
from world_model.belief import BeliefFactory, BeliefTrajectory
from world_model.datasets import collate_episodes
from world_model.observations import MeasurementSet
from world_model.runtime import OnlineWorldModel
from world_model.simulator import generate_episode
from world_model.training.loop import (
    TrainingBatchResult,
    _belief_state_losses,
    _current_correction_objective_support,
    _future_predictable_mask,
    _group_closed_loop_terms,
    _rollout_loss_result,
    _select_rollout_anchor_frames,
    _state_velocity_objective_axis_support,
    _update_geometric_identity_metrics,
    _weighted_closed_loop_total,
    future_scene_predictable_mask,
    gather_target_pairs,
    match_belief_to_targets,
    measurement_localization_metrics,
    pretrain_rgb_measurements,
    run_closed_loop_batch,
)
from world_model.training.losses import gaussian_nll
from world_model.training.trainer import (
    _aggregate_physical_validation_metrics,
    _mean_batch_results,
    _measurement_selection_from_checkpoint,
    _measurement_selection_guardrail_failures,
    _measurement_selection_improves,
    _measurement_selection_metrics,
    _rollout_selection_guardrail_failures,
    _rollout_selection_improves,
    _rollout_selection_metrics,
    set_closed_loop_trainable_scope,
)
from world_model.utils.config import OrpheusConfig, load_config


def _single_horizon_config(*, minimum_rollout_age_steps: int = 0) -> OrpheusConfig:
    config = load_config("configs/tiny_overfit.yaml")
    return replace(
        config,
        training=replace(
            config.training,
            horizon_weights=(1.0,),
            minimum_rollout_age_steps=minimum_rollout_age_steps,
        ),
        evaluation=replace(
            config.evaluation,
            horizons_seconds=(0.05,),
        ),
    )


def _active_belief(
    *,
    positions: Tensor,
    age_steps: Tensor,
):
    belief = BeliefFactory(max_objects=positions.shape[1]).create(batch_size=positions.shape[0])
    objects = belief.objects.clone()
    objects.active.fill_(True)
    objects.object_id.copy_(
        torch.arange(positions.shape[1], dtype=torch.int64)[None].expand_as(objects.object_id)
    )
    objects.position.copy_(positions)
    objects.age_steps.copy_(age_steps)
    return replace(belief, objects=objects)


def test_gather_target_pairs_aligns_both_axes_and_zeros_unmatched_slots() -> None:
    target = torch.tensor(
        [
            [
                [0, 1, 2],
                [3, 4, 5],
                [6, 7, 8],
            ]
        ]
    )
    indices = torch.tensor([[2, 0, -1]], dtype=torch.int64)

    aligned = gather_target_pairs(target, indices)

    torch.testing.assert_close(
        aligned,
        torch.tensor(
            [
                [
                    [8, 6, 0],
                    [2, 0, 0],
                    [0, 0, 0],
                ]
            ]
        ),
    )


def test_geometric_identity_metric_counts_track_swaps() -> None:
    target_position = torch.tensor(
        [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]],
    )
    target_active = torch.ones(1, 2, dtype=torch.bool)
    target_ids = torch.tensor([[100, 200]], dtype=torch.int64)
    belief = _active_belief(
        positions=target_position.clone(),
        age_steps=torch.ones(1, 2, dtype=torch.int64),
    )
    history: list[dict[int, int]] = [{}]

    indices, matched = match_belief_to_targets(
        belief,
        target_position,
        target_active,
    )
    switches, associations = _update_geometric_identity_metrics(
        belief,
        target_ids,
        indices,
        matched,
        history,
    )
    assert (switches, associations) == (0, 2)

    swapped_objects = belief.objects.clone()
    swapped_objects.position.copy_(target_position.flip(dims=(1,)))
    swapped_belief = replace(belief, objects=swapped_objects)
    indices, matched = match_belief_to_targets(
        swapped_belief,
        target_position,
        target_active,
    )
    switches, associations = _update_geometric_identity_metrics(
        swapped_belief,
        target_ids,
        indices,
        matched,
        history,
    )
    assert (switches, associations) == (2, 2)


def _measurement_batch() -> dict[str, Any]:
    return {
        "rgb": torch.zeros(1, 1, 3, 8, 8),
        "timestamps": torch.zeros(1, 1),
        "camera": {
            "world_from_camera": torch.eye(4)[None, None],
            "intrinsics": torch.eye(3)[None, None],
        },
        "labels": {
            "projected_center": torch.zeros(1, 1, 1, 2),
            "log_apparent_radius_normalized": torch.zeros(1, 1, 1),
            "inverse_depth": torch.zeros(1, 1, 1),
            "albedo": torch.zeros(1, 1, 1, 3),
            "existence": torch.ones(1, 1, 1, dtype=torch.bool),
            "projected_valid": torch.ones(1, 1, 1, dtype=torch.bool),
            "visible": torch.ones(1, 1, 1, dtype=torch.bool),
            "visible_fraction": torch.ones(1, 1, 1),
        },
        "objects": {
            "position": torch.zeros(1, 1, 1, 3),
            "albedo": torch.zeros(1, 1, 1, 3),
        },
    }


def _half_confident_measurement() -> MeasurementSet:
    return MeasurementSet(
        modality="rgb",
        sensor_id="camera0",
        timestamp=torch.zeros(1),
        values=torch.zeros(1, 1, 7),
        log_variance=torch.zeros(1, 1, 7),
        existence_logits=torch.zeros(1, 1),
        measurement_mask=torch.ones(1, 1, dtype=torch.bool),
        appearance=None,
        class_logits=None,
        frame_id="camera:camera0",
        supported_state_fields=("position",),
        auxiliary={"world_position": torch.zeros(1, 1, 3)},
    )


class _FixedMeasurementModule:
    def __init__(self, measurements: MeasurementSet) -> None:
        self.measurements = measurements

    def initialise_measurements(
        self,
        packets: Any,
        context: Any,
    ) -> MeasurementSet:
        del packets, context
        return self.measurements

    def training_losses(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor],
        masks: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        del targets, masks
        return {"existence": outputs["existence_logits"].square().mean()}


def test_measurement_discovery_metrics_use_runtime_birth_confidence() -> None:
    batch = _measurement_batch()
    measurements = _half_confident_measurement()

    detector_threshold = measurement_localization_metrics(
        measurements,
        batch,
        0,
        birth_confidence_threshold=0.45,
    )
    lifecycle_threshold = measurement_localization_metrics(
        measurements,
        batch,
        0,
        birth_confidence_threshold=0.55,
    )

    assert detector_threshold["rgb_runtime_birth_recall_at_0_5m"] == 1.0
    assert detector_threshold["rgb_runtime_birth_precision_at_0_5m"] == 1.0
    assert lifecycle_threshold["rgb_runtime_birth_recall_at_0_5m"] == 0.0
    assert lifecycle_threshold["rgb_runtime_birth_precision_at_0_5m"] == 0.0
    assert math.isinf(lifecycle_threshold["rgb_runtime_birth_world_position_mae_m"])


def test_runtime_discovery_matching_excludes_subthreshold_proposals() -> None:
    batch = _measurement_batch()
    values = torch.zeros(1, 2, 7)
    # The sub-threshold proposal is the closest all-proposal match. The only
    # runtime-eligible proposal remains within the physical acceptance gate.
    values[0, 1, 0] = 0.06
    world_position = torch.zeros(1, 2, 3)
    world_position[0, 1, 0] = 0.1
    measurements = MeasurementSet(
        modality="rgb",
        sensor_id="camera0",
        timestamp=torch.zeros(1),
        values=values,
        log_variance=torch.zeros_like(values),
        existence_logits=torch.tensor([[-10.0, 10.0]]),
        measurement_mask=torch.ones(1, 2, dtype=torch.bool),
        appearance=None,
        class_logits=None,
        frame_id="camera:camera0",
        supported_state_fields=("position",),
        auxiliary={"world_position": world_position},
    )

    metrics = measurement_localization_metrics(
        measurements,
        batch,
        0,
        birth_confidence_threshold=0.55,
    )

    assert metrics["rgb_world_position_mae_m"] == 0.0
    assert math.isclose(
        metrics["rgb_runtime_birth_world_position_mae_m"],
        0.1,
        abs_tol=1.0e-6,
    )
    assert metrics["rgb_runtime_birth_recall_at_0_5m"] == 1.0
    assert metrics["rgb_runtime_birth_precision_at_0_5m"] == 1.0
    assert metrics["rgb_runtime_birth_f1_at_0_5m"] == 1.0


def test_runtime_discovery_counts_false_positive_when_no_target_is_visible() -> None:
    batch = _measurement_batch()
    batch["labels"]["visible"].fill_(False)
    measurements = _half_confident_measurement()
    measurements.existence_logits.fill_(10.0)

    metrics = measurement_localization_metrics(
        measurements,
        batch,
        0,
        birth_confidence_threshold=0.55,
    )

    assert metrics["rgb_runtime_birth_target_count"] == 0.0
    assert metrics["rgb_runtime_birth_proposal_count"] == 1.0
    assert metrics["rgb_runtime_birth_true_positive_count_at_0_5m"] == 0.0
    assert metrics["rgb_runtime_birth_recall_at_0_5m"] == 0.0
    assert metrics["rgb_runtime_birth_precision_at_0_5m"] == 0.0
    assert metrics["rgb_runtime_birth_f1_at_0_5m"] == 0.0
    assert math.isinf(metrics["rgb_runtime_birth_world_position_mae_m"])


def test_pretraining_reports_proposals_at_lifecycle_not_detector_threshold() -> None:
    config = load_config("configs/tiny_overfit.yaml")
    config = replace(
        config,
        model=replace(
            config.model,
            rgb=replace(config.model.rgb, existence_threshold=0.45),
            lifecycle=replace(config.model.lifecycle, birth_confidence=0.55),
        ),
    )
    model = SimpleNamespace(
        observation_modules={"rgb": _FixedMeasurementModule(_half_confident_measurement())}
    )

    result = pretrain_rgb_measurements(
        model,
        _measurement_batch(),
        config,
        frame_index=0,
    )

    assert result.metrics["rgb_runtime_birth_confidence_threshold"] == 0.55
    assert result.metrics["proposals_above_birth_threshold"] == 0.0
    assert result.metrics["rgb_runtime_birth_recall_at_0_5m"] == 0.0


def test_measurement_validation_pools_birth_counts_and_errors_exactly() -> None:
    def result(
        *,
        true_positive: float,
        count: float,
        error_sum: float,
    ) -> TrainingBatchResult:
        return TrainingBatchResult(
            total_loss=torch.zeros(()),
            loss_terms={},
            metrics={
                # Deliberately misleading macro metrics: pooling must replace
                # them from the additive sufficient statistics below.
                "rgb_runtime_birth_recall_at_0_5m": true_positive / count,
                "rgb_runtime_birth_precision_at_0_5m": true_positive / count,
                "rgb_runtime_birth_f1_at_0_5m": true_positive / count,
                "rgb_runtime_birth_world_position_mae_m": error_sum / count,
                "rgb_world_position_mae_m": error_sum / count,
                "rgb_runtime_birth_true_positive_count_at_0_5m": true_positive,
                "rgb_runtime_birth_target_count": count,
                "rgb_runtime_birth_proposal_count": count,
                "rgb_runtime_birth_world_position_absolute_error_sum_m": error_sum,
                "rgb_runtime_birth_matched_proposal_count": count,
                "rgb_world_position_absolute_error_sum_m": error_sum,
                "rgb_world_position_matched_proposal_count": count,
            },
            phase="rgb_pretrain",
        )

    aggregate = _mean_batch_results(
        [
            result(true_positive=1.0, count=1.0, error_sum=0.1),
            result(true_positive=0.0, count=9.0, error_sum=9.0),
        ]
    )

    assert aggregate.metrics["rgb_runtime_birth_target_count"] == 10.0
    assert aggregate.metrics["rgb_runtime_birth_recall_at_0_5m"] == 0.1
    assert aggregate.metrics["rgb_runtime_birth_precision_at_0_5m"] == 0.1
    assert math.isclose(
        aggregate.metrics["rgb_runtime_birth_f1_at_0_5m"],
        0.1,
    )
    assert math.isclose(
        aggregate.metrics["rgb_runtime_birth_world_position_mae_m"],
        0.91,
    )
    assert math.isclose(aggregate.metrics["rgb_world_position_mae_m"], 0.91)


def test_measurement_validation_pools_fast_pair_support_exactly() -> None:
    def result(
        *,
        bootstrap_matched: float,
        target_count: float,
        supported: float,
        true_positive: float,
        error_sum: float,
        prior_error_sum: float,
    ) -> TrainingBatchResult:
        return TrainingBatchResult(
            total_loss=torch.zeros(()),
            loss_terms={},
            metrics={
                "rgb_fast_bootstrap_matched_target_count": bootstrap_matched,
                "rgb_fast_bootstrap_target_count": target_count,
                "rgb_fast_roi_supported_target_count": supported,
                "rgb_fast_roi_target_count": target_count,
                "rgb_fast_roi_eligible_proposal_count": supported,
                "rgb_fast_roi_world_position_absolute_error_sum_m": error_sum,
                "rgb_fast_roi_world_position_matched_count": supported,
                "rgb_fast_prior_world_position_absolute_error_sum_m": prior_error_sum,
                "rgb_fast_prior_world_position_matched_count": supported,
                "rgb_fast_roi_confident_proposal_count": supported,
                "rgb_fast_roi_true_positive_count_at_0_5m": true_positive,
            },
            phase="rgb_pretrain",
        )

    aggregate = _mean_batch_results(
        [
            result(
                bootstrap_matched=1.0,
                target_count=1.0,
                supported=1.0,
                true_positive=1.0,
                error_sum=0.1,
                prior_error_sum=0.3,
            ),
            result(
                bootstrap_matched=1.0,
                target_count=9.0,
                supported=2.0,
                true_positive=1.0,
                error_sum=0.8,
                prior_error_sum=1.2,
            ),
        ]
    )

    assert aggregate.metrics["rgb_fast_bootstrap_target_coverage"] == 0.2
    assert aggregate.metrics["rgb_fast_roi_target_coverage"] == 0.3
    assert aggregate.metrics["rgb_fast_roi_recall_at_0_5m"] == 0.2
    assert math.isclose(
        aggregate.metrics["rgb_fast_roi_world_position_mae_m"],
        0.3,
    )
    assert math.isclose(
        aggregate.metrics["rgb_fast_prior_world_position_mae_m"],
        0.5,
    )
    assert math.isclose(
        aggregate.metrics["rgb_fast_roi_improvement_m"],
        0.2,
    )


def test_state_uncertainty_nll_calibrates_variance_without_duplicate_mean_gradient() -> None:
    mean = torch.tensor([[[2.0, -1.0, 0.5]]], requires_grad=True)
    log_variance = torch.zeros_like(mean, requires_grad=True)
    loss = gaussian_nll(
        mean,
        torch.zeros_like(mean),
        log_variance,
        torch.ones(1, 1, dtype=torch.bool),
        detach_mean_error=True,
    )

    loss.backward()

    assert mean.grad is None
    assert log_variance.grad is not None
    assert torch.isfinite(log_variance.grad).all()


def test_unsupported_state_and_parameter_terms_are_omitted_not_zero_averaged() -> None:
    belief = BeliefFactory(max_objects=2).create(batch_size=1)
    batch = {
        "objects": {
            "active": torch.zeros(1, 1, 2, dtype=torch.bool),
            "position": torch.zeros(1, 1, 2, 3),
            "velocity": torch.zeros(1, 1, 2, 3),
            "drag": torch.zeros(1, 1, 2, 1),
            "restitution": torch.zeros(1, 1, 2, 1),
        },
        "events": {
            "collision": torch.zeros(1, 1, 2, dtype=torch.bool),
        },
    }

    losses, _, matched = _belief_state_losses(
        belief,
        batch,
        _single_horizon_config(),
        frame_index=0,
    )

    assert not matched.any()
    assert losses == {}


def test_birth_velocity_is_excluded_from_loss_numerator_and_denominator() -> None:
    belief = _active_belief(
        positions=torch.zeros(1, 2, 3),
        age_steps=torch.tensor([[0, 1]], dtype=torch.int64),
    )
    predicted_velocity = torch.zeros(1, 2, 3, requires_grad=True)
    belief = belief.replace(
        objects=belief.objects.replace(velocity=predicted_velocity),
    )
    target_velocity = torch.tensor(
        [[[[10.0, -10.0, 10.0], [0.5, -1.0, 2.0]]]],
    )
    batch = {
        "objects": {
            "active": torch.ones(1, 1, 2, dtype=torch.bool),
            "position": torch.zeros(1, 1, 2, 3),
            "velocity": target_velocity,
            "drag": torch.zeros(1, 1, 2, 1),
            "restitution": torch.zeros(1, 1, 2, 1),
        }
    }
    indices = torch.tensor([[0, 1]], dtype=torch.int64)
    matched = torch.ones(1, 2, dtype=torch.bool)

    support = _state_velocity_objective_axis_support(belief, matched)
    losses, _, _ = _belief_state_losses(
        belief,
        batch,
        _single_horizon_config(),
        frame_index=0,
        indices=indices,
        matched=matched,
        velocity_axis_support=support,
    )

    assert support.tolist() == [[[False, False, False], [True, True, True]]]
    assert support.sum().item() == 3
    expected = torch.nn.functional.smooth_l1_loss(
        predicted_velocity[:, 1],
        target_velocity[:, 0, 1],
    )
    torch.testing.assert_close(losses["state_velocity"], expected)

    losses["state_velocity"].backward()
    torch.testing.assert_close(
        predicted_velocity.grad[:, 0],
        torch.zeros(1, 3),
    )
    assert torch.count_nonzero(predicted_velocity.grad[:, 1]).item() == 3


def test_all_birth_velocity_support_omits_the_objective() -> None:
    belief = _active_belief(
        positions=torch.zeros(1, 2, 3),
        age_steps=torch.zeros(1, 2, dtype=torch.int64),
    )
    batch = {
        "objects": {
            "active": torch.ones(1, 1, 2, dtype=torch.bool),
            "position": torch.zeros(1, 1, 2, 3),
            "velocity": torch.full((1, 1, 2, 3), 100.0),
            "drag": torch.zeros(1, 1, 2, 1),
            "restitution": torch.zeros(1, 1, 2, 1),
        }
    }

    losses, _, _ = _belief_state_losses(
        belief,
        batch,
        _single_horizon_config(),
        frame_index=0,
        indices=torch.tensor([[0, 1]], dtype=torch.int64),
        matched=torch.ones(1, 2, dtype=torch.bool),
    )

    assert "state_position" in losses
    assert "state_velocity" not in losses


def test_velocity_correction_support_requires_a_surviving_incoming_prior() -> None:
    belief = _active_belief(
        positions=torch.zeros(1, 2, 3),
        age_steps=torch.tensor([[0, 1]], dtype=torch.int64),
    )
    prior = _active_belief(
        positions=torch.zeros(1, 2, 3),
        age_steps=torch.zeros(1, 2, dtype=torch.int64),
    )
    prior = prior.replace(
        objects=prior.objects.replace(active=torch.tensor([[False, True]])),
    )

    support = _current_correction_objective_support(
        belief,
        prior,
        torch.ones(1, 2, dtype=torch.bool),
    )

    assert support.tolist() == [[False, True]]


def test_existence_belief_supervises_only_causally_active_predictions() -> None:
    belief = BeliefFactory(max_objects=2).create(batch_size=1)
    existence_logit = torch.zeros((1, 2), requires_grad=True)
    belief = belief.replace(
        objects=belief.objects.replace(
            active=torch.tensor([[True, False]]),
            object_id=torch.tensor([[0, -1]], dtype=torch.int64),
            existence_logit=existence_logit,
        )
    )
    batch = {
        "objects": {
            "active": torch.zeros(1, 1, 2, dtype=torch.bool),
            "position": torch.zeros(1, 1, 2, 3),
            "velocity": torch.zeros(1, 1, 2, 3),
            "drag": torch.zeros(1, 1, 2, 1),
            "restitution": torch.zeros(1, 1, 2, 1),
        },
        "events": {
            "collision": torch.zeros(1, 1, 2, dtype=torch.bool),
        },
    }

    losses, _, matched = _belief_state_losses(
        belief,
        batch,
        _single_horizon_config(),
        frame_index=0,
    )
    losses["existence_belief"].backward()

    assert not matched.any()
    assert set(losses) == {"existence_belief"}
    torch.testing.assert_close(
        existence_logit.grad,
        torch.tensor([[0.5, 0.0]]),
    )


class _StaticRolloutDynamics:
    def __init__(self, position_log_variance: Tensor | None = None) -> None:
        self.position_log_variance = position_log_variance

    def rollout(
        self,
        belief: Any,
        query_seconds: list[float] | tuple[float, ...],
        *,
        return_events: bool,
        return_auxiliary: bool,
    ) -> BeliefTrajectory:
        assert not return_auxiliary
        count = len(query_seconds)
        objects = belief.objects
        timestamps = belief.timestamp[:, None] + belief.timestamp.new_tensor(query_seconds)[None]
        positions = objects.position[:, None].expand(-1, count, -1, -1)
        velocities = objects.velocity[:, None].expand(-1, count, -1, -1)
        orientations = objects.orientation[:, None].expand(-1, count, -1, -1)
        modes = objects.motion_mode_logits[:, None].expand(-1, count, -1, -1)
        active = objects.active[:, None].expand(-1, count, -1)
        base_log_variance = objects.fast_log_variance[:, None].expand(
            -1,
            count,
            -1,
            -1,
        )
        if self.position_log_variance is None:
            fast_log_variance = base_log_variance
        else:
            position_log_variance = self.position_log_variance.expand(
                *base_log_variance.shape[:-1],
                3,
            )
            fast_log_variance = torch.cat(
                (position_log_variance, base_log_variance[..., 3:]),
                dim=-1,
            )
        event_logits = modes.new_zeros(modes.shape) if return_events else None
        return BeliefTrajectory(
            timestamps=timestamps,
            positions=positions,
            velocities=velocities,
            orientations=orientations,
            motion_mode_logits=modes,
            fast_log_variance=fast_log_variance,
            active_mask=active,
            event_logits=event_logits,
        ).validate()


class _DifferentiableEventRolloutDynamics(_StaticRolloutDynamics):
    def __init__(self) -> None:
        super().__init__()
        self.event_logit = torch.tensor(0.25, requires_grad=True)

    def rollout(
        self,
        belief: Any,
        query_seconds: list[float] | tuple[float, ...],
        *,
        return_events: bool,
        return_auxiliary: bool,
    ) -> BeliefTrajectory:
        trajectory = super().rollout(
            belief,
            query_seconds,
            return_events=return_events,
            return_auxiliary=return_auxiliary,
        )
        if trajectory.event_logits is None:
            return trajectory
        return replace(
            trajectory,
            event_logits=trajectory.event_logits + self.event_logit,
        ).validate()


def _rollout_batch(*, externally_actuated: Tensor) -> dict[str, Any]:
    batch, frames, objects = externally_actuated.shape
    return {
        "rgb": torch.zeros(batch, frames, 3, 1, 1),
        "objects": {
            "active": torch.ones(batch, frames, objects, dtype=torch.bool),
            "position": torch.zeros(batch, frames, objects, 3),
            "velocity": torch.zeros(batch, frames, objects, 3),
        },
        "events": {
            "collision": torch.zeros(
                batch,
                frames,
                objects,
                dtype=torch.bool,
            ),
            "externally_actuated": externally_actuated,
        },
    }


def test_zero_effective_event_weight_omits_bce_graph_but_keeps_physical_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _single_horizon_config()
    config = replace(
        config,
        training=replace(
            config.training,
            closed_loop_event_loss_weights={"state_roi": 0.0},
        ),
    )
    config.validate()
    weights, _ = training_loop._closed_loop_loss_weights_for_scope(
        config,
        active_trainable_scope="state_roi",
    )
    assert weights["event"] == 0.0

    batch = _rollout_batch(externally_actuated=torch.zeros(1, 2, 2, dtype=torch.bool))
    batch["events"]["collision"][:, 1].fill_(True)
    belief = _active_belief(
        positions=torch.zeros(1, 2, 3),
        age_steps=torch.full((1, 2), 5, dtype=torch.int64),
    )
    belief.objects.position.requires_grad_()
    indices = torch.tensor([[0, 1]], dtype=torch.int64)
    matched = torch.ones(1, 2, dtype=torch.bool)

    event_owned_dynamics = _DifferentiableEventRolloutDynamics()
    event_owned = _rollout_loss_result(
        SimpleNamespace(dynamics=event_owned_dynamics),
        belief,
        batch,
        config,
        frame_index=0,
        indices=indices,
        matched=matched,
        compute_event_loss=True,
    )

    def unexpected_event_bce(*_args: Any, **_kwargs: Any) -> Tensor:
        raise AssertionError("zero-weight event objective constructed a BCE graph")

    monkeypatch.setattr(
        training_loop,
        "balanced_binary_cross_entropy",
        unexpected_event_bce,
    )
    suppressed_dynamics = _DifferentiableEventRolloutDynamics()
    suppressed = _rollout_loss_result(
        SimpleNamespace(dynamics=suppressed_dynamics),
        belief,
        batch,
        config,
        frame_index=0,
        indices=indices,
        matched=matched,
        compute_event_loss=weights["event"] != 0.0,
    )

    assert "event_collision" in event_owned.losses
    assert not any(name.startswith("event_collision") for name in suppressed.losses)
    assert event_owned.positions is not None and suppressed.positions is not None
    assert event_owned.velocities is not None and suppressed.velocities is not None
    torch.testing.assert_close(event_owned.positions, suppressed.positions)
    torch.testing.assert_close(event_owned.velocities, suppressed.velocities)
    assert event_owned.physical_metrics == suppressed.physical_metrics

    event_owned.losses["event_collision"].backward()
    assert event_owned_dynamics.event_logit.grad is not None
    assert event_owned_dynamics.event_logit.grad.abs().item() > 0.0
    sum(suppressed.losses.values()).backward()
    assert suppressed_dynamics.event_logit.grad is None


def test_unseen_actuation_censors_the_coupled_scene_but_keeps_distribution_nll() -> None:
    config = _single_horizon_config()
    actuation = torch.zeros(1, 2, 2, dtype=torch.bool)
    actuation[0, 1, 0] = True
    batch = _rollout_batch(externally_actuated=actuation)
    belief = _active_belief(
        positions=torch.full((1, 2, 3), 2.0),
        age_steps=torch.full((1, 2), 5, dtype=torch.int64),
    )
    belief.objects.position.requires_grad_()
    log_variance = torch.zeros((), requires_grad=True)
    model = SimpleNamespace(dynamics=_StaticRolloutDynamics(position_log_variance=log_variance))
    indices = torch.tensor([[0, 1]], dtype=torch.int64)
    matched = torch.ones(1, 2, dtype=torch.bool)

    predictable = _future_predictable_mask(
        batch,
        anchor_index=0,
        target_index=1,
        target_indices=indices,
    )
    result = _rollout_loss_result(
        model,
        belief,
        batch,
        config,
        frame_index=0,
        indices=indices,
        matched=matched,
    )

    assert predictable.tolist() == [[False, False]]
    assert "rollout_position" not in result.losses
    assert "rollout_velocity" not in result.losses
    assert "event_collision" not in result.losses
    assert result.losses["rollout_position_nll"].item() > 0.0
    assert "rollout_position@0.050s" not in result.losses
    assert "rollout_velocity@0.050s" not in result.losses
    assert "rollout_position_nll@0.050s" in result.losses
    assert (
        result.physical_metrics["physical_rollout_censored_external_actuation_count@0.050s"] == 2.0
    )
    assert result.physical_metrics["physical_forecast_predictable_target_count@0.050s"] == 0.0
    assert (
        result.physical_metrics["physical_rollout_position_coverage90@0.050s_coordinate_count"]
        == 6.0
    )
    assert result.physical_metrics["physical_rollout_position@0.050s_coordinate_count"] == 0.0

    result.losses["rollout_position_nll"].backward()
    assert belief.objects.position.grad is None
    assert log_variance.grad is not None
    # Gradient descent therefore increases variance for the hidden outcome.
    assert log_variance.grad.item() < 0.0


def test_scene_predictability_censors_only_batches_with_unseen_actuation() -> None:
    actuation = torch.zeros(2, 3, 2, dtype=torch.bool)
    actuation[0, 1, 0] = True
    batch = {
        "objects": {
            "active": torch.ones(2, 3, 2, dtype=torch.bool),
        },
        "events": {
            "externally_actuated": actuation,
        },
    }
    indices = torch.tensor([[0, 1], [0, 1]], dtype=torch.int64)

    scene = future_scene_predictable_mask(
        batch,
        anchor_index=0,
        target_index=2,
    )
    targets = _future_predictable_mask(
        batch,
        anchor_index=0,
        target_index=2,
        target_indices=indices,
    )

    assert scene.tolist() == [False, True]
    assert targets.tolist() == [[False, False], [True, True]]


def test_mature_mask_excludes_cold_deterministic_targets_but_reports_both() -> None:
    config = _single_horizon_config(minimum_rollout_age_steps=3)
    batch = _rollout_batch(externally_actuated=torch.zeros(1, 2, 2, dtype=torch.bool))
    belief = _active_belief(
        positions=torch.tensor([[[2.0, 2.0, 2.0], [0.0, 0.0, 0.0]]]),
        age_steps=torch.tensor([[0, 3]], dtype=torch.int64),
    )
    result = _rollout_loss_result(
        SimpleNamespace(dynamics=_StaticRolloutDynamics()),
        belief,
        batch,
        config,
        frame_index=0,
        indices=torch.tensor([[0, 1]], dtype=torch.int64),
        matched=torch.ones(1, 2, dtype=torch.bool),
    )

    assert result.losses["rollout_position"].item() == 0.0
    assert result.losses["rollout_position_nll"].item() > 0.0
    assert (
        result.physical_metrics["physical_rollout_mature_position@0.050s_coordinate_count"] == 3.0
    )
    assert (
        result.physical_metrics["physical_rollout_cold_start_position@0.050s_coordinate_count"]
        == 3.0
    )
    assert _select_rollout_anchor_frames(
        config,
        window_start=0,
        window_stop=5,
        total_frames=8,
        rollout_anchors_per_window=None,
    ) == (3, 4)


def test_optional_rollout_nll_never_falls_back_to_unit_weight() -> None:
    details = {
        "rollout_position_nll": torch.tensor(10.0),
        "frozen_global_measurement": torch.tensor(1000.0),
    }
    terms = _group_closed_loop_terms(details, torch.zeros(()))

    assert "measurement" not in terms
    without_explicit_weight = _weighted_closed_loop_total(
        terms,
        {"measurement": 1.0},
    )
    with_explicit_weight = _weighted_closed_loop_total(
        terms,
        {"measurement": 1.0, "rollout_nll": 0.02},
    )

    assert without_explicit_weight.item() == 0.0
    torch.testing.assert_close(with_explicit_weight, torch.tensor(0.2))


def test_frozen_global_measurement_is_diagnostic_only_with_fast_roi_scope() -> None:
    config = load_config("configs/tiny_overfit.yaml")
    batch = collate_episodes([generate_episode(config, seed=9)])
    model = OnlineWorldModel.from_config(config, device="cpu")
    set_closed_loop_trainable_scope(model, scope="state_dynamics_fast_roi")

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

    assert "frozen_global_measurement" in result.metrics
    assert "measurement_fast" in result.metrics
    assert "measurement_global" not in result.metrics
    assert (
        result.metrics["state_velocity_objective_supported_coordinate_count"]
        + result.metrics["state_velocity_objective_excluded_coordinate_count"]
        == result.metrics["physical_state_velocity_coordinate_count"]
    )
    torch.testing.assert_close(
        result.loss_terms["measurement"].detach(),
        torch.tensor(result.metrics["measurement_fast"])
        * config.training.fast_roi_pretrain_weight
        / (1.0 + config.training.fast_roi_pretrain_weight),
    )


def test_frozen_fast_measurement_cannot_train_state_dynamics_through_prior() -> None:
    config = load_config("configs/tiny_overfit.yaml")
    batch = collate_episodes([generate_episode(config, seed=9)])
    model = OnlineWorldModel.from_config(config, device="cpu")
    set_closed_loop_trainable_scope(model, scope="state_dynamics")

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

    assert "frozen_global_measurement" in result.metrics
    assert "frozen_fast_measurement" in result.metrics
    assert "measurement" not in result.loss_terms
    assert "measurement_fast" not in result.metrics
    assert not result.support_terms


def test_velocity_correction_is_a_distinct_weightable_objective() -> None:
    terms = _group_closed_loop_terms(
        {
            "correction_current": torch.tensor(1.0),
            "correction_future": torch.tensor(3.0),
            "correction_current_velocity": torch.tensor(2.0),
            "correction_future_velocity": torch.tensor(4.0),
            "correction_magnitude": torch.tensor(6.0),
        },
        torch.zeros(()),
    )

    torch.testing.assert_close(terms["correction_position"], torch.tensor(2.0))
    torch.testing.assert_close(terms["correction_velocity"], torch.tensor(3.0))
    torch.testing.assert_close(
        terms["correction_regularization"],
        torch.tensor(6.0),
    )
    torch.testing.assert_close(
        _weighted_closed_loop_total(
            terms,
            {
                "correction_position": 2.0,
                "correction_velocity": 3.0,
                "correction_regularization": 0.5,
            },
        ),
        torch.tensor(16.0),
    )


def _additive_result(scale: float) -> TrainingBatchResult:
    metrics = {
        "physical_state_position_sse": 3.0 * scale,
        "physical_state_position_coordinate_count": 3.0 * scale,
        "physical_state_position_x_sse": 1.0 * scale,
        "physical_state_position_x_coordinate_count": 1.0 * scale,
        "physical_state_position_y_sse": 1.0 * scale,
        "physical_state_position_y_coordinate_count": 1.0 * scale,
        "physical_state_position_z_sse": 1.0 * scale,
        "physical_state_position_z_coordinate_count": 1.0 * scale,
        "physical_state_velocity_sse": 3.0 * scale,
        "physical_state_velocity_coordinate_count": 3.0 * scale,
        "physical_distance_gated_matched_object_frames": 1.0 * scale,
        "physical_distance_gated_target_object_frames": 1.0 * scale,
        "physical_distance_gated_predicted_object_frames": 1.0 * scale,
        "physical_distance_gated_identity_switches": 0.0,
        "physical_distance_gated_object_frame_associations": 1.0 * scale,
        "physical_predicted_object_frames": 2.0 * scale,
        "physical_position_coverage90_hit_count": 12.0 * scale,
        "physical_position_coverage90_coordinate_count": 15.0 * scale,
        "physical_collision_true_positive_count": 1.0 * scale,
        "physical_collision_false_positive_count": 0.0,
        "physical_collision_false_negative_count": 0.0,
        "physical_rollout_position@0.050s_sse": 3.0 * scale,
        "physical_rollout_position@0.050s_coordinate_count": 3.0 * scale,
        "physical_rollout_position_x@0.050s_sse": 1.0 * scale,
        "physical_rollout_position_x@0.050s_coordinate_count": 1.0 * scale,
        "physical_rollout_position_y@0.050s_sse": 1.0 * scale,
        "physical_rollout_position_y@0.050s_coordinate_count": 1.0 * scale,
        "physical_rollout_position_z@0.050s_sse": 1.0 * scale,
        "physical_rollout_position_z@0.050s_coordinate_count": 1.0 * scale,
        "physical_rollout_position_coverage90@0.050s_hit_count": 12.0 * scale,
        "physical_rollout_position_coverage90@0.050s_coordinate_count": 15.0 * scale,
        "physical_forecast_active_count@0.050s": 1.0 * scale,
        "physical_forecast_tracked_count@0.050s": 5.0 * scale,
        "physical_forecast_target_count@0.050s": 5.0 * scale,
        "physical_forecast_predictable_target_count@0.050s": 1.0 * scale,
        "physical_rollout_predictable_target_count@0.050s": 1.0 * scale,
        "physical_collision_true_positive_count@0.050s": 2.0 * scale,
        "physical_rollout_censored_external_actuation_count@0.050s": (4.0 * scale),
    }
    return TrainingBatchResult(
        total_loss=torch.zeros(()),
        loss_terms={},
        metrics=metrics,
        phase="closed_loop_rgb",
    )


def test_physical_count_aggregation_preserves_exact_horizon_totals() -> None:
    aggregate = _aggregate_physical_validation_metrics(
        [_additive_result(1.0), _additive_result(2.0)],
        _single_horizon_config(),
    )

    assert aggregate["physical_collision_true_positive_count@0.050s"] == 6.0
    assert aggregate["physical_rollout_censored_external_actuation_count@0.050s"] == 12.0
    assert aggregate["physical_forecast_target_count@0.050s"] == 15.0
    assert aggregate["physical_predicted_object_frames"] == 6.0


def test_physical_validation_records_zero_horizon_support_without_fabricated_rmse() -> None:
    unsupported = _additive_result(1.0)
    unsupported.metrics["physical_rollout_position@0.050s_sse"] = 0.0
    unsupported.metrics["physical_rollout_position@0.050s_coordinate_count"] = 0.0

    aggregate = _aggregate_physical_validation_metrics(
        [unsupported],
        _single_horizon_config(),
    )

    assert aggregate["selection_metric_supported"] == 0.0
    assert aggregate["physical_rollout_position@0.050s_coordinate_count"] == 0.0
    assert "validation_position_rmse@0.050s" not in aggregate


def test_missing_physical_metric_schema_is_not_misreported_as_zero_support() -> None:
    malformed = _additive_result(1.0)
    del malformed.metrics["physical_state_position_sse"]

    with pytest.raises(
        RuntimeError,
        match="missing additive physical validation metric 'physical_state_position_sse'",
    ):
        _aggregate_physical_validation_metrics(
            [malformed],
            _single_horizon_config(),
        )


def test_zero_horizon_support_cannot_hide_an_unrelated_missing_schema_key() -> None:
    malformed = _additive_result(1.0)
    malformed.metrics["physical_forecast_predictable_target_count@0.050s"] = 0.0
    malformed.metrics["physical_rollout_position@0.050s_sse"] = 0.0
    malformed.metrics["physical_rollout_position@0.050s_coordinate_count"] = 0.0
    del malformed.metrics["physical_collision_false_negative_count"]

    with pytest.raises(
        RuntimeError,
        match="missing additive physical validation metric "
        "'physical_collision_false_negative_count'",
    ):
        _aggregate_physical_validation_metrics(
            [malformed],
            _single_horizon_config(),
        )


def test_negative_support_count_is_invalid_not_ordinary_unsupported() -> None:
    malformed = _additive_result(1.0)
    malformed.metrics["physical_forecast_predictable_target_count@0.050s"] = -1.0

    with pytest.raises(
        ValueError,
        match="physical_forecast_predictable_target_count@0.050s.*invalid",
    ):
        _aggregate_physical_validation_metrics(
            [malformed],
            _single_horizon_config(),
        )


def _selection_metrics(*, horizon: float, axis_x: float) -> dict[str, float]:
    metrics = {
        "validation_position_rmse_m": 0.5,
        "validation_velocity_rmse_mps": 1.0,
        "validation_target_coverage": 0.9,
        "validation_prediction_precision": 0.9,
        "validation_collision_f1": 0.6,
        "validation_id_switch_rate": 0.0,
        "validation_position_coverage90": 0.9,
        "validation_position_rmse@0.050s": horizon,
        "validation_forecast_target_coverage@0.050s": 0.9,
    }
    for axis, value in (("x", axis_x), ("y", 0.5), ("z", 0.5)):
        metrics[f"validation_position_rmse_{axis}_m"] = value
        metrics[f"validation_position_rmse_{axis}@0.050s"] = value
    return metrics


def _measurement_metrics(
    *,
    world_mae: float,
    recall: float,
    precision: float,
    fast_roi_coverage: float = 0.70,
    fast_roi_mae: float = 0.20,
) -> dict[str, float]:
    f1 = 0.0 if recall + precision == 0 else 2.0 * recall * precision / (recall + precision)
    return {
        "rgb_runtime_birth_world_position_mae_m": world_mae,
        "rgb_world_position_mae_m": world_mae,
        "rgb_runtime_birth_recall_at_0_5m": recall,
        "rgb_runtime_birth_precision_at_0_5m": precision,
        "rgb_runtime_birth_f1_at_0_5m": f1,
        "rgb_fast_bootstrap_target_coverage": 0.75,
        "rgb_fast_roi_target_coverage": fast_roi_coverage,
        "rgb_fast_roi_world_position_mae_m": fast_roi_mae,
        "rgb_fast_roi_recall_at_0_5m": 0.65,
        "rgb_fast_roi_precision_at_0_5m": 0.75,
        "rgb_fast_roi_f1_at_0_5m": 0.6964285714,
        "rgb_fast_roi_improvement_m": 0.05,
    }


def test_measurement_selector_rejects_mae_win_that_loses_runtime_recall() -> None:
    incumbent = _measurement_selection_metrics(
        _measurement_metrics(world_mae=0.25, recall=0.8, precision=0.8)
    )
    candidate = _measurement_selection_metrics(
        _measurement_metrics(world_mae=0.20, recall=0.7, precision=0.85)
    )
    assert incumbent is not None
    assert candidate is not None

    failures = _measurement_selection_guardrail_failures(candidate, incumbent)

    assert candidate.score < incumbent.score
    assert not _measurement_selection_improves(candidate, incumbent)
    assert {failure["metric"] for failure in failures} == {"runtime_birth_recall_at_0_5m"}


def test_measurement_selector_can_promote_runtime_recall_not_only_mae() -> None:
    incumbent = _measurement_selection_metrics(
        _measurement_metrics(world_mae=0.25, recall=0.7, precision=0.8)
    )
    candidate = _measurement_selection_metrics(
        _measurement_metrics(world_mae=0.25, recall=0.8, precision=0.8)
    )
    assert incumbent is not None
    assert candidate is not None

    assert candidate.score < incumbent.score
    assert _measurement_selection_improves(candidate, incumbent)


def test_measurement_selector_rejects_global_gain_with_fast_roi_collapse() -> None:
    incumbent = _measurement_selection_metrics(
        _measurement_metrics(world_mae=0.25, recall=0.75, precision=0.75)
    )
    candidate = _measurement_selection_metrics(
        _measurement_metrics(
            world_mae=0.10,
            recall=0.90,
            precision=0.90,
            fast_roi_coverage=0.20,
            fast_roi_mae=0.40,
        )
    )
    assert incumbent is not None
    assert candidate is not None

    failures = _measurement_selection_guardrail_failures(candidate, incumbent)

    assert not _measurement_selection_improves(candidate, incumbent)
    assert {failure["metric"] for failure in failures} >= {
        "fast_roi_target_coverage",
        "fast_roi_world_position_mae_m",
    }


def test_measurement_selector_resume_requires_broad_versioned_metrics() -> None:
    selection = _measurement_selection_metrics(
        _measurement_metrics(world_mae=0.25, recall=0.8, precision=0.75)
    )
    assert selection is not None
    checkpoint_metrics = selection.checkpoint_metrics()

    assert _measurement_selection_from_checkpoint(checkpoint_metrics) == selection
    assert (
        _measurement_selection_from_checkpoint(
            {
                "best_measurement_validated": 1.0,
                "best_measurement_world_position_mae_m": 0.2,
            }
        )
        is None
    )
    unusable = _measurement_metrics(world_mae=math.inf, recall=0.0, precision=0.0)
    assert _measurement_selection_metrics(unusable) is None


def test_axis_guardrail_blocks_hidden_regression_despite_better_score() -> None:
    config = _single_horizon_config()
    incumbent = _rollout_selection_metrics(
        _selection_metrics(horizon=0.5, axis_x=0.5),
        config,
    )
    candidate = _rollout_selection_metrics(
        _selection_metrics(horizon=0.4, axis_x=0.516),
        config,
    )

    failures = _rollout_selection_guardrail_failures(candidate, incumbent)

    assert candidate.score < incumbent.score
    assert not _rollout_selection_improves(candidate, incumbent)
    assert {failure["metric"] for failure in failures} >= {
        "position_rmse_x_m",
        "position_rmse_x@0.050s",
    }


def _prior_future_correction_test_config() -> OrpheusConfig:
    config = load_config("configs/tiny_overfit.yaml")
    config = replace(
        config,
        simulator=replace(
            config.simulator,
            image_size=(32, 32),
            sequence_frames=4,
            min_objects=1,
            max_objects=1,
            camera_motion="fixed",
            render_noise_std=0.0,
        ),
        model=replace(
            config.model,
            rgb=replace(
                config.model.rgb,
                global_every_steps=100,
                global_uncertainty_threshold=1.0e6,
                surprise_threshold=1.0e6,
            ),
            lifecycle=replace(config.model.lifecycle, birth_confidence=0.0),
        ),
        training=replace(
            config.training,
            batch_size=1,
            tbptt_steps=4,
            horizon_weights=(1.0,),
        ),
        evaluation=replace(
            config.evaluation,
            horizons_seconds=(0.05,),
            episodes=1,
        ),
    )
    config.validate()
    return config


def test_prior_and_posterior_correction_rollouts_use_identical_query_partitions(
    monkeypatch: Any,
) -> None:
    # This test isolates rollout graph/partition reuse from target-bootstrap
    # localization. The randomly initialized RGB model is not expected to
    # satisfy the production 0.5 m supervision gate.
    monkeypatch.setattr(
        training_loop,
        "_PHYSICAL_SELECTION_DISTANCE_THRESHOLD_M",
        10.0,
    )
    config = _prior_future_correction_test_config()
    batch = collate_episodes([generate_episode(config, seed=9)])
    model = OnlineWorldModel.from_config(config, device="cpu")
    original_rollout = model.dynamics.rollout
    query_partitions: list[tuple[float, ...]] = []
    rollout_grad_modes: list[tuple[bool, bool, bool]] = []

    def recording_rollout(
        belief: Any,
        query_seconds: Any,
        *,
        return_events: bool,
        return_auxiliary: bool,
    ) -> BeliefTrajectory:
        assert not return_auxiliary
        query_partitions.append(tuple(float(value) for value in query_seconds))
        trajectory = original_rollout(
            belief,
            query_seconds,
            return_events=return_events,
            return_auxiliary=return_auxiliary,
        )
        rollout_grad_modes.append(
            (
                return_events,
                torch.is_grad_enabled(),
                trajectory.positions.requires_grad,
            )
        )
        if not return_events:
            # The no-grad guard changes graph retention, not the trajectory.
            with torch.enable_grad():
                gradient_reference = original_rollout(
                    belief,
                    query_seconds,
                    return_events=False,
                    return_auxiliary=False,
                )
            torch.testing.assert_close(
                trajectory.positions,
                gradient_reference.positions,
            )
            torch.testing.assert_close(
                trajectory.velocities,
                gradient_reference.velocities,
            )
        return trajectory

    monkeypatch.setattr(model.dynamics, "rollout", recording_rollout)
    result = run_closed_loop_batch(
        model,
        batch,
        config,
        window_steps=4,
        apply_perturbations=False,
        include_measurement_supervision=False,
        rollout_anchors_per_window=2,
    )

    # Frame zero has only a posterior rollout. The later anchor has both
    # prior and posterior rollouts; every path must include the preceding
    # observation boundary [0, 0.05] rather than comparing different
    # numerical/event partitions.
    assert query_partitions == [
        (0.0, 0.05),
        (0.0, 0.05),
        (0.0, 0.05),
    ]
    prior_modes = [mode for mode in rollout_grad_modes if not mode[0]]
    posterior_modes = [mode for mode in rollout_grad_modes if mode[0]]
    assert prior_modes
    assert all(
        not grad_enabled and not output_requires_grad
        for _, grad_enabled, output_requires_grad in prior_modes
    )
    assert posterior_modes
    assert all(
        grad_enabled and output_requires_grad
        for _, grad_enabled, output_requires_grad in posterior_modes
    )

    posterior_rollout_loss = (
        result.loss_terms["rollout_position"] + result.loss_terms["rollout_velocity"]
    )
    assert posterior_rollout_loss.requires_grad
    model.zero_grad(set_to_none=True)
    posterior_rollout_loss.backward()
    assert any(
        parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad
    )


def test_disabling_prior_future_correction_removes_only_the_extra_rollout_and_losses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        training_loop,
        "_PHYSICAL_SELECTION_DISTANCE_THRESHOLD_M",
        10.0,
    )
    enabled_config = _prior_future_correction_test_config()
    disabled_config = replace(
        enabled_config,
        training=replace(
            enabled_config.training,
            closed_loop_prior_future_correction_enabled=False,
        ),
    )
    disabled_config.validate()
    batch = collate_episodes([generate_episode(enabled_config, seed=9)])

    torch.manual_seed(321)
    enabled_model = OnlineWorldModel.from_config(enabled_config, device="cpu")
    disabled_model = OnlineWorldModel.from_config(disabled_config, device="cpu")
    disabled_model.load_state_dict(enabled_model.state_dict())
    enabled_calls: list[bool] = []
    disabled_calls: list[bool] = []

    def record_rollouts(model: OnlineWorldModel, calls: list[bool]) -> None:
        original_rollout = model.dynamics.rollout

        def recording_rollout(
            belief: Any,
            query_seconds: Any,
            *,
            return_events: bool,
            return_auxiliary: bool,
            **kwargs: Any,
        ) -> BeliefTrajectory:
            calls.append(return_events)
            return original_rollout(
                belief,
                query_seconds,
                return_events=return_events,
                return_auxiliary=return_auxiliary,
                **kwargs,
            )

        monkeypatch.setattr(model.dynamics, "rollout", recording_rollout)

    record_rollouts(enabled_model, enabled_calls)
    record_rollouts(disabled_model, disabled_calls)
    common = {
        "window_steps": 4,
        "apply_perturbations": False,
        "include_measurement_supervision": False,
        "rollout_anchors_per_window": 2,
    }
    torch.manual_seed(987)
    enabled = run_closed_loop_batch(enabled_model, batch, enabled_config, **common)
    torch.manual_seed(987)
    disabled = run_closed_loop_batch(disabled_model, batch, disabled_config, **common)

    assert enabled_calls.count(False) == 1
    assert enabled_calls.count(True) == 2
    assert disabled_calls.count(False) == 0
    assert disabled_calls.count(True) == 2
    assert enabled.metrics["prior_future_correction_rollout_enabled"] == 1.0
    assert disabled.metrics["prior_future_correction_rollout_enabled"] == 0.0

    def is_future_correction_loss_detail(name: str) -> bool:
        return name in {"correction_future", "correction_future_velocity"} or name.startswith(
            ("correction_future@", "correction_future_velocity@")
        )

    assert any(is_future_correction_loss_detail(name) for name in enabled.metrics)
    assert not any(is_future_correction_loss_detail(name) for name in disabled.metrics)

    comparable_details = {
        name
        for name in disabled.metrics
        if name in enabled.metrics
        and name.startswith(
            (
                "rollout_position",
                "rollout_velocity",
                "event_collision",
                "correction_current",
            )
        )
    }
    assert comparable_details
    for name in comparable_details:
        assert disabled.metrics[name] == pytest.approx(enabled.metrics[name], abs=1.0e-7)
    for name, value in disabled.loss_terms.items():
        if name.startswith("rollout_") or name == "event":
            torch.testing.assert_close(value, enabled.loss_terms[name])

    enabled_physical = {
        name: value for name, value in enabled.metrics.items() if name.startswith("physical_")
    }
    disabled_physical = {
        name: value for name, value in disabled.metrics.items() if name.startswith("physical_")
    }
    assert enabled_physical.keys() == disabled_physical.keys()
    for name, value in disabled_physical.items():
        assert value == pytest.approx(enabled_physical[name], nan_ok=True)
