from __future__ import annotations

import pytest
import torch

from world_model.evaluation.occlusion_metrics import (
    OcclusionTransitionAccumulator,
)


def _update(
    accumulator: OcclusionTransitionAccumulator,
    *,
    prediction_id: int,
    position_std_m: float,
    visible_fraction: float,
    reliable_match: bool,
    prediction_active: bool = True,
) -> None:
    accumulator.update_frame(
        predicted_ids=torch.tensor([[prediction_id]]),
        predicted_active=torch.tensor([[prediction_active]]),
        position_std_m=torch.tensor([[position_std_m]]),
        target_ids=torch.tensor([[42]]),
        target_active=torch.tensor([[True]]),
        target_visible_fraction=torch.tensor([[visible_fraction]]),
        matched_target_indices=torch.tensor([[0 if reliable_match else -1]]),
        reliable_visible_matches=torch.tensor([[reliable_match]]),
        episode_offset=0,
    )


def test_occlusion_metrics_follow_persistent_id_without_occluded_rematching() -> None:
    accumulator = OcclusionTransitionAccumulator()
    _update(
        accumulator,
        prediction_id=7,
        position_std_m=0.10,
        visible_fraction=1.0,
        reliable_match=True,
    )
    # There is deliberately no current association during either fully hidden
    # frame. The established persistent ID is still sufficient to measure its
    # uncertainty, regardless of current localization quality.
    _update(
        accumulator,
        prediction_id=7,
        position_std_m=0.20,
        visible_fraction=0.0,
        reliable_match=False,
    )
    _update(
        accumulator,
        prediction_id=7,
        position_std_m=0.35,
        visible_fraction=0.0,
        reliable_match=False,
    )
    _update(
        accumulator,
        prediction_id=7,
        position_std_m=0.15,
        visible_fraction=0.9,
        reliable_match=True,
    )

    metrics = accumulator.metrics()
    assert metrics["occlusion_qualifying_sequence_count"] == 1.0
    assert metrics["occlusion_identity_survival_count"] == 1.0
    assert metrics["occlusion_identity_survival_rate"] == 1.0
    assert metrics["occlusion_growth_evaluated_sequence_count"] == 1.0
    assert metrics["occlusion_pre_position_std_mean_m"] == pytest.approx(0.10)
    assert metrics["occlusion_peak_position_std_mean_m"] == pytest.approx(0.35)
    assert metrics["occlusion_position_std_growth_mean_m"] == pytest.approx(0.25)
    assert metrics["occlusion_position_std_growth_positive_rate"] == 1.0
    assert metrics["occlusion_reobservation_contraction_evaluated_sequence_count"] == 1.0
    assert metrics["occlusion_reobservation_position_std_mean_m"] == pytest.approx(0.15)
    assert metrics["occlusion_reobservation_std_contraction_mean_m"] == pytest.approx(0.20)
    assert metrics["occlusion_reobservation_std_contraction_positive_rate"] == 1.0


def test_reappearance_with_new_prediction_id_is_a_survival_failure() -> None:
    accumulator = OcclusionTransitionAccumulator()
    _update(
        accumulator,
        prediction_id=7,
        position_std_m=0.10,
        visible_fraction=1.0,
        reliable_match=True,
    )
    _update(
        accumulator,
        prediction_id=7,
        position_std_m=0.25,
        visible_fraction=0.0,
        reliable_match=False,
    )
    _update(
        accumulator,
        prediction_id=8,
        position_std_m=0.12,
        visible_fraction=1.0,
        reliable_match=True,
    )

    metrics = accumulator.metrics()
    assert metrics["occlusion_qualifying_sequence_count"] == 1.0
    assert metrics["occlusion_identity_survival_count"] == 0.0
    assert metrics["occlusion_identity_survival_rate"] == 0.0
    assert metrics["occlusion_position_std_growth_mean_m"] == pytest.approx(0.15)
    assert metrics["occlusion_reobservation_contraction_evaluated_sequence_count"] == 0.0
    assert metrics["occlusion_reobservation_position_std_mean_m"] is None
    assert metrics["occlusion_reobservation_std_contraction_mean_m"] is None
    assert metrics["occlusion_reobservation_std_contraction_positive_rate"] is None


def test_no_complete_occlusion_sequence_reports_zero_counts_and_null_values() -> None:
    accumulator = OcclusionTransitionAccumulator()
    _update(
        accumulator,
        prediction_id=7,
        position_std_m=0.10,
        visible_fraction=1.0,
        reliable_match=True,
    )
    _update(
        accumulator,
        prediction_id=7,
        position_std_m=0.30,
        visible_fraction=0.0,
        reliable_match=False,
    )

    metrics = accumulator.metrics()
    assert metrics["occlusion_qualifying_sequence_count"] == 0.0
    assert metrics["occlusion_identity_survival_count"] == 0.0
    assert metrics["occlusion_identity_survival_rate"] is None
    assert metrics["occlusion_growth_evaluated_sequence_count"] == 0.0
    assert metrics["occlusion_pre_position_std_mean_m"] is None
    assert metrics["occlusion_peak_position_std_mean_m"] is None
    assert metrics["occlusion_position_std_growth_mean_m"] is None
    assert metrics["occlusion_position_std_growth_positive_rate"] is None
    assert metrics["occlusion_reobservation_contraction_evaluated_sequence_count"] == 0.0
    assert metrics["occlusion_reobservation_position_std_mean_m"] is None
    assert metrics["occlusion_reobservation_std_contraction_mean_m"] is None
    assert metrics["occlusion_reobservation_std_contraction_positive_rate"] is None
