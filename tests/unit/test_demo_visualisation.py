from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from world_model.utils.config import OrpheusConfig
from world_model.visualisation.animation import (
    _add_image_legend,
    _add_world_legend,
    _configure_world_axis,
    _demo_generation_config,
    _ForecastTrace,
    _future_query_seconds,
    _history_alpha,
    _match_positions,
    _plot_ground_truth_window,
    _plot_historical_forecasts,
    _project_world_uncertainty,
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


def test_demo_reserves_label_only_lookahead_without_mutating_runtime_config() -> None:
    config = OrpheusConfig()

    generation_config, display_count = _demo_generation_config(config)

    assert display_count == config.demo.max_frames
    assert generation_config.simulator.sequence_frames == 78
    assert config.simulator.sequence_frames == 72
    assert generation_config.model is config.model


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


def test_ground_truth_plot_separates_past_from_current_horizon_by_object() -> None:
    figure, axis = plt.subplots()
    positions = np.asarray(
        [
            [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            [[0.1, 0.9, 0.0], [0.9, 0.9, 0.0]],
            [[0.2, 0.7, 0.0], [0.8, 0.7, 0.0]],
            [[0.3, 0.4, 0.0], [0.7, 0.4, 0.0]],
        ]
    )
    active = np.ones((4, 2), dtype=bool)

    lines = _plot_ground_truth_window(
        axis,
        positions,
        active,
        np.asarray([10, 11]),
        current_index=1,
        future_index=3,
    )

    assert len(lines) == 4
    np.testing.assert_allclose(lines[0].get_xdata(), [0.0, 0.1])
    np.testing.assert_allclose(lines[1].get_xdata(), [0.1, 0.2, 0.3])
    np.testing.assert_allclose(lines[2].get_xdata(), [1.0, 0.9])
    np.testing.assert_allclose(lines[3].get_xdata(), [0.9, 0.8, 0.7])
    assert lines[0].get_linestyle() == ":"
    assert lines[1].get_linestyle() == "-"
    assert lines[1].get_color() != lines[3].get_color()
    assert [text.get_text() for text in axis.texts if text.get_text()] == [
        "GT 10",
        "GT 11",
    ]
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
        "posterior 90% position uncertainty",
        "ground truth overlay",
    ]
    assert [text.get_text() for text in axes[1].get_legend().get_texts()] == [
        "GT past (through now)",
        "GT current horizon (object colours)",
        "historical posterior forecasts",
        "latest prior forecast",
        "latest posterior forecast",
        "posterior endpoint ↔ matched GT",
    ]
    assert axes[1].get_legend()._loc == 1
    plt.close(figure)


def test_world_covariance_is_projected_through_camera_jacobian() -> None:
    position = torch.tensor([[0.0, 0.0, 2.0]])
    log_variance = torch.tensor([[0.01, 0.04, 0.25]]).log()
    active = torch.tensor([True])
    world_from_camera = torch.eye(4)
    intrinsics = torch.tensor(
        [
            [100.0, 0.0, 20.0],
            [0.0, 100.0, 20.0],
            [0.0, 0.0, 1.0],
        ]
    )

    sigma, angle, valid = _project_world_uncertainty(
        position,
        log_variance,
        active,
        world_from_camera,
        intrinsics,
    )

    assert valid.tolist() == [True]
    np.testing.assert_allclose(sigma[0], [10.0, 5.0], atol=1.0e-5)
    assert abs(abs(float(angle[0])) - 90.0) < 1.0e-5
