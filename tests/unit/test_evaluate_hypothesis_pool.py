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
