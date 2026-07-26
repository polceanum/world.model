from __future__ import annotations

import pytest
import torch

from world_model.evaluation.evaluator import (
    _distance_gate_matches,
    _IdentifierAccumulator,
    _ParameterAccumulator,
)
from world_model.identification import ParameterUpdateDiagnostics


def test_parameter_metrics_follow_runtime_identifier_masks() -> None:
    active = torch.tensor([[True, True, False]])
    observability = torch.zeros(1, 3, 5)
    gate = torch.zeros_like(observability)
    update_count = torch.zeros_like(observability, dtype=torch.int64)
    observability[0, 0, 1] = 0.5
    observability[0, :2, 2] = torch.tensor([0.25, 0.75])
    gate[0, 0, 1] = 0.05
    gate[0, :2, 2] = torch.tensor([0.0005, 0.0009])
    update_count[0, 0, 1] = 1
    diagnostics = ParameterUpdateDiagnostics(
        observability=observability,
        gate=gate,
        delta=torch.zeros_like(observability),
        update_count=update_count,
    )

    identifier = _IdentifierAccumulator()
    identifier.update(diagnostics, active)
    identifier_results = identifier.metrics()
    assert identifier_results["identifier_restitution_update_count"] == 1.0
    assert identifier_results["identifier_drag_update_count"] == 0.0
    assert identifier_results["identifier_drag_observability_mean"] == 0.5
    assert identifier_results["identifier_drag_gate_max"] == pytest.approx(0.0009)

    prediction = torch.tensor([[[0.4], [0.8], [0.1]]])
    target = torch.tensor([[[0.5], [0.5], [0.5]]])
    matched = torch.tensor([[True, True, True]])
    parameter = _ParameterAccumulator()
    parameter.update(
        "observable",
        "drag",
        prediction,
        target,
        matched & (observability[..., 2] > 0),
    )
    parameter.update(
        "updated",
        "drag",
        prediction,
        target,
        matched & update_count[..., 2].bool(),
    )
    parameter_results = parameter.metrics()
    assert parameter_results["observable_drag_count"] == 2.0
    assert parameter_results["observable_drag_mae"] == pytest.approx(0.2)
    assert parameter_results["updated_drag_count"] == 0.0
    assert parameter_results["updated_drag_mae"] is None


def test_detection_distance_gate_rejects_far_hungarian_assignments() -> None:
    prediction = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]])
    aligned_target = torch.tensor([[[0.3, 0.0, 0.0], [1.6, 1.0, 1.0], [0.0, 0.0, 0.0]]])
    assignment = torch.tensor([[True, True, False]])

    gated = _distance_gate_matches(
        prediction,
        aligned_target,
        assignment,
        threshold_m=0.5,
    )

    torch.testing.assert_close(gated, torch.tensor([[True, False, False]]))
