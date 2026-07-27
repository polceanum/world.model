from __future__ import annotations

import math

import pytest
import torch

from world_model.evaluation.velocity_metrics import (
    MaskedVelocityErrorAccumulator,
    OrdinaryVelocityCorrectionAccumulator,
    TemporalVelocityMeasurementAccumulator,
)
from world_model.observations import MeasurementSet


def _measurements(
    *,
    auxiliary: dict[str, torch.Tensor],
    measurement_mask: torch.Tensor | None = None,
) -> MeasurementSet:
    mask = torch.tensor([[True, True, False]]) if measurement_mask is None else measurement_mask
    return MeasurementSet(
        modality="rgb",
        sensor_id="camera0",
        timestamp=torch.tensor([0.1]),
        values=torch.zeros(1, 3, 8),
        log_variance=torch.zeros(1, 3, 8),
        existence_logits=torch.zeros(1, 3),
        measurement_mask=mask,
        appearance=None,
        class_logits=None,
        frame_id="camera:camera0",
        supported_state_fields=("position", "velocity"),
        auxiliary=auxiliary,
    )


def test_current_velocity_error_is_coordinate_wise_under_exact_mask() -> None:
    accumulator = MaskedVelocityErrorAccumulator()
    accumulator.update(
        prediction=torch.tensor([[[1.0, -2.0, 3.0], [100.0, 100.0, 100.0]]]),
        target=torch.zeros(1, 2, 3),
        mask=torch.tensor([[True, False]]),
    )

    metrics = accumulator.metrics("posterior_current")
    assert metrics["posterior_current_velocity_rmse_mps"] == pytest.approx(math.sqrt(14.0 / 3.0))
    assert metrics["posterior_current_velocity_mae_mps"] == pytest.approx(2.0)
    assert metrics["posterior_current_velocity_coordinate_count"] == 3.0
    assert metrics["posterior_current_velocity_object_frame_count"] == 1.0
    assert metrics["posterior_current_velocity_x_rmse_mps"] == 1.0
    assert metrics["posterior_current_velocity_y_rmse_mps"] == 2.0
    assert metrics["posterior_current_velocity_z_rmse_mps"] == 3.0
    assert metrics["posterior_current_velocity_x_count"] == 1.0


def test_ordinary_velocity_correction_reports_norm_error_improvement() -> None:
    accumulator = OrdinaryVelocityCorrectionAccumulator()
    accumulator.update(
        prior=torch.tensor([[[3.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]),
        posterior=torch.tensor([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]),
        target=torch.zeros(1, 2, 3),
        mask=torch.tensor([[True, True]]),
    )

    metrics = accumulator.metrics()
    assert metrics["ordinary_velocity_prior_norm_error_mean_mps"] == 2.0
    assert metrics["ordinary_velocity_posterior_norm_error_mean_mps"] == 1.5
    assert metrics["ordinary_velocity_prior_to_posterior_norm_error_improvement_mean_mps"] == 0.5
    assert metrics["ordinary_velocity_prior_to_posterior_norm_error_improvement_fraction"] == 0.25
    assert metrics["ordinary_velocity_prior_to_posterior_positive_rate"] == 0.5
    assert metrics["ordinary_velocity_evaluated_object_updates"] == 2.0
    assert metrics["ordinary_velocity_x_prior_mae_mps"] == 2.0
    assert metrics["ordinary_velocity_x_posterior_mae_mps"] == 1.5
    assert metrics["ordinary_velocity_x_improvement_mps"] == 0.5
    assert metrics["ordinary_velocity_x_positive_rate"] == 0.5
    assert metrics["ordinary_velocity_y_improvement_mps"] == 0.0


def test_velocity_accumulators_report_null_without_eligible_objects() -> None:
    current = MaskedVelocityErrorAccumulator().metrics("posterior_current")
    correction = OrdinaryVelocityCorrectionAccumulator().metrics()

    assert current["posterior_current_velocity_rmse_mps"] is None
    assert current["posterior_current_velocity_object_frame_count"] == 0.0
    assert correction["ordinary_velocity_prior_norm_error_mean_mps"] is None
    assert correction["ordinary_velocity_prior_to_posterior_positive_rate"] is None
    assert correction["ordinary_velocity_evaluated_object_updates"] == 0.0


def test_temporal_velocity_measurement_availability_and_variance() -> None:
    variance = torch.tensor([1.0, 4.0, 9.0])
    log_variance = variance.log().view(1, 1, 3).expand(1, 3, 3).clone()
    accumulator = TemporalVelocityMeasurementAccumulator()
    accumulator.update(None)
    accumulator.update(
        _measurements(
            auxiliary={
                "world_velocity": torch.zeros(1, 3, 3),
                "world_velocity_log_variance": log_variance,
                "world_velocity_valid_mask": torch.tensor([[True, False, True]]),
            }
        )
    )

    metrics = accumulator.metrics()
    assert metrics["temporal_velocity_measurement_inspected_update_count"] == 2.0
    assert metrics["temporal_velocity_measurement_explicit_field_update_count"] == 1.0
    assert metrics["temporal_velocity_measurement_explicit_field_update_fraction"] == 0.5
    assert metrics["temporal_velocity_measurement_valid_update_count"] == 1.0
    assert metrics["temporal_velocity_measurement_candidate_object_count"] == 2.0
    assert metrics["temporal_velocity_measurement_valid_object_count"] == 1.0
    assert metrics["temporal_velocity_measurement_valid_object_fraction"] == 0.5
    assert metrics["temporal_velocity_measurement_reported_variance_mean_mps2"] == pytest.approx(
        14.0 / 3.0
    )
    assert metrics["temporal_velocity_measurement_reported_variance_coordinate_count"] == 3.0


def test_temporal_velocity_measurement_rejects_partial_explicit_fields() -> None:
    accumulator = TemporalVelocityMeasurementAccumulator()
    measurements = _measurements(
        auxiliary={
            "world_velocity_valid_mask": torch.tensor([[True, False, False]]),
        }
    )

    with pytest.raises(ValueError, match="require all auxiliary fields"):
        accumulator.update(measurements)


def test_velocity_metrics_validate_shapes_and_mask_types() -> None:
    with pytest.raises(TypeError, match="torch.bool"):
        MaskedVelocityErrorAccumulator().update(
            prediction=torch.zeros(1, 2, 3),
            target=torch.zeros(1, 2, 3),
            mask=torch.ones(1, 2),
        )

    measurements = _measurements(
        auxiliary={
            "world_velocity": torch.zeros(1, 3, 3),
            "world_velocity_log_variance": torch.zeros(1, 3, 3),
            "world_velocity_valid_mask": torch.ones(1, 3),
        }
    )
    with pytest.raises(TypeError, match="valid_mask"):
        TemporalVelocityMeasurementAccumulator().update(measurements)
