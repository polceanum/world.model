"""Focused correctness tests for heterogeneous-pool report accounting."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

_SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "evaluate_hypothesis_pool.py"
_SPEC = importlib.util.spec_from_file_location("evaluate_hypothesis_pool", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_aggregate_episode_reports = _MODULE._aggregate_episode_reports
_selected_lifecycle_counts = _MODULE._selected_lifecycle_counts


def test_selected_lifecycle_counts_use_the_selected_horizon_mask() -> None:
    # The first horizon is fully active while the later selected horizon has a
    # disappearance. Reusing the first slice would falsely report zero error.
    later_prediction = torch.tensor([[True, False, True]])
    target = torch.tensor([[True, True, False]])

    mismatch, coverage = _selected_lifecycle_counts(later_prediction, target)

    assert mismatch == 2
    assert coverage == 1


@pytest.mark.parametrize(
    ("prediction", "target", "exception"),
    [
        (torch.tensor([True]), torch.tensor([[True]]), ValueError),
        (torch.tensor([1]), torch.tensor([1]), TypeError),
    ],
)
def test_selected_lifecycle_counts_validate_masks(
    prediction: torch.Tensor,
    target: torch.Tensor,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        _selected_lifecycle_counts(prediction, target)


def test_aggregate_episode_reports_pools_sse_before_root() -> None:
    def result(selected_sse: float, selected_count: int) -> dict[str, object]:
        return {
            "selection_counts": [1, 0],
            "selected_position_sse_m2": {"0.1": [selected_sse, 0.0, 0.0]},
            "selected_position_coordinate_count": {"0.1": [selected_count, 1, 1]},
            "candidate_position_sse_m2": {"0.1": [[selected_sse, 0.0, 0.0], [9.0, 0.0, 0.0]]},
            "candidate_position_coordinate_count": {"0.1": [[selected_count, 1, 1], [1, 1, 1]]},
            "selected_lifecycle_mismatch": {"0.1": 2},
            "selected_identity_coverage": {"0.1": 3},
            "selected_event_metrics": {
                "0.1": {
                    "collision_true_positive": 1.0,
                    "collision_false_positive": 2.0,
                    "collision_false_negative": 3.0,
                }
            },
        }

    aggregate = _aggregate_episode_reports(
        [result(1.0, 1), result(9.0, 9)],
        (0.1,),
        candidate_count=2,
    )

    assert aggregate["selected_rmse_m"]["0.1"][0] == pytest.approx(1.0)
    assert aggregate["candidate_rmse_m"]["0.1"][1][0] == pytest.approx(3.0)
    assert aggregate["selection_counts"] == [2, 0]
    assert aggregate["selected_lifecycle_mismatch"]["0.1"] == 4
    assert aggregate["selected_identity_coverage"]["0.1"] == 6
