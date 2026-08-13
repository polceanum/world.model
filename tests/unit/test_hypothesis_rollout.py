from __future__ import annotations

import pytest
import torch

from world_model.belief import BeliefTrajectory
from world_model.dynamics import HypothesisRolloutEngine


def _trajectory(position: float, *, variance: float = 1.0) -> BeliefTrajectory:
    return BeliefTrajectory(
        timestamps=torch.tensor([[0.1, 0.2]], dtype=torch.float32),
        positions=torch.full((1, 2, 1, 3), position),
        velocities=torch.zeros(1, 2, 1, 3),
        orientations=torch.tensor([[[[1.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]]]]),
        motion_mode_logits=torch.zeros(1, 2, 1, 2),
        fast_log_variance=torch.full((1, 2, 1, 13), variance),
        active_mask=torch.ones(1, 2, 1, dtype=torch.bool),
    ).validate()


def test_selector_chooses_best_candidate_per_batch() -> None:
    target = torch.zeros(1, 2, 1, 3)
    selection = HypothesisRolloutEngine.score(
        [_trajectory(0.5), _trajectory(0.0)],
        target,
        torch.ones(1, 2, 1, dtype=torch.bool),
        uncertainty_aware=False,
    )
    assert selection.selected_index.tolist() == [1]
    assert selection.scores[0, 1] < selection.scores[0, 0]
    assert torch.allclose(selection.posterior_weights.sum(-1), torch.ones(1))


def test_selector_ignores_occluded_frames_and_scores_uncertainty() -> None:
    target = torch.zeros(1, 2, 1, 3)
    mask = torch.tensor([[[True], [False]]])
    narrow = _trajectory(0.2, variance=0.0)
    wide = _trajectory(0.3, variance=2.0)
    selection = HypothesisRolloutEngine.score(
        [narrow, wide], target, mask, uncertainty_aware=True
    )
    # The log-variance penalty prevents a deliberately broad forecast from
    # winning just because its normalized residual is smaller.
    assert selection.selected_index.item() == 0


def test_selector_rejects_empty_or_bad_mask() -> None:
    target = torch.zeros(1, 2, 1, 3)
    mask = torch.ones(1, 2, 1, dtype=torch.bool)
    with pytest.raises(ValueError, match="at least one"):
        HypothesisRolloutEngine.score([], target, mask)
    with pytest.raises(TypeError, match="torch.bool"):
        HypothesisRolloutEngine.score([_trajectory(0.0)], target, mask.to(torch.float32))
