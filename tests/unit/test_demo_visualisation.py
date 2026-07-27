from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from world_model.visualisation.animation import (
    _add_image_legend,
    _add_world_legend,
    _configure_world_axis,
    _ForecastTrace,
    _future_query_seconds,
    _history_alpha,
    _match_positions,
    _plot_historical_forecasts,
)


def test_demo_endpoint_matching_respects_masks_and_original_slots() -> None:
    prediction = torch.tensor(
        [
            [20.0, 20.0, 0.0],
            [1.1, 1.0, 0.0],
            [9.9, 0.0, 0.0],
            [-20.0, -20.0, 0.0],
        ]
    )
    target = torch.tensor(
        [
            [10.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [50.0, 50.0, 0.0],
        ]
    )

    matches = _match_positions(
        prediction,
        target,
        torch.tensor([False, True, True, False]),
        torch.tensor([True, True, False]),
    )

    assert matches.prediction_indices.tolist() == [1, 2]
    assert matches.target_indices.tolist() == [1, 0]
    np.testing.assert_allclose(matches.distances, [0.1, 0.1], atol=1.0e-6)
    assert matches.mean_error == pytest.approx(0.1, abs=1.0e-6)


def test_demo_dense_queries_follow_observation_timestamps() -> None:
    timestamps = torch.tensor([0.0, 0.05, 0.10, 0.15])

    assert _future_query_seconds(timestamps, 1, 3) == pytest.approx([0.05, 0.10])
    assert _future_query_seconds(timestamps, 3, 3) == []
    with pytest.raises(IndexError):
        _future_query_seconds(timestamps, 2, 4)


def test_historical_forecasts_keep_absolute_anchors_and_fade_by_age() -> None:
    figure, axis = plt.subplots()
    forecasts = [
        _ForecastTrace(
            anchor_index=1,
            anchor_timestamp=0.05,
            positions=np.asarray([[[1.0, 2.0, 0.0]], [[1.5, 2.5, 0.0]]]),
            active=np.ones((2, 1), dtype=bool),
        ),
        _ForecastTrace(
            anchor_index=3,
            anchor_timestamp=0.15,
            positions=np.asarray([[[3.0, 1.0, 0.0]], [[3.5, 1.5, 0.0]]]),
            active=np.ones((2, 1), dtype=bool),
        ),
    ]

    lines = _plot_historical_forecasts(axis, forecasts)

    assert len(lines) == 2
    np.testing.assert_allclose(lines[0].get_xdata(), [1.0, 1.5])
    np.testing.assert_allclose(lines[1].get_xdata(), [3.0, 3.5])
    assert lines[0].get_alpha() == pytest.approx(_history_alpha(0, 2))
    assert lines[1].get_alpha() == pytest.approx(_history_alpha(1, 2))
    assert lines[0].get_alpha() < lines[1].get_alpha()
    plt.close(figure)


def test_demo_axes_and_legends_have_stable_geometry_and_entries() -> None:
    figure, axes = plt.subplots(1, 2)
    bounds = ((-2.0, 2.0), (0.0, 3.0), (-1.0, 1.0))

    _add_image_legend(axes[0])
    _configure_world_axis(axes[1], bounds)
    _add_world_legend(axes[1])

    assert axes[1].get_xlim() == pytest.approx((-2.16, 2.16))
    assert axes[1].get_ylim() == pytest.approx((-0.12, 3.12))
    assert [text.get_text() for text in axes[0].get_legend().get_texts()] == [
        "scheduled RGB measurement",
        "prior",
        "posterior",
        "ground truth overlay",
    ]
    assert [text.get_text() for text in axes[1].get_legend().get_texts()] == [
        "GT trajectory overlay",
        "historical posterior forecasts",
        "latest prior forecast",
        "latest posterior forecast",
        "posterior endpoint ↔ matched GT",
    ]
    plt.close(figure)
