"""Evaluator-only evidence for posterior and measured world velocity."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor

from world_model.observations import DirectVelocityEvidence, MeasurementSet

_TEMPORAL_VELOCITY_KEYS = (
    "world_velocity",
    "world_velocity_log_variance",
    "world_velocity_valid_mask",
)
_TEMPORAL_VELOCITY_AXIS_MASK_KEY = "world_velocity_axis_valid_mask"


def _validate_velocity_inputs(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    prediction_name: str,
) -> None:
    if prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ValueError(f"{prediction_name} must have shape [B,N,3]")
    if target.shape != prediction.shape:
        raise ValueError("target velocity must match prediction shape [B,N,3]")
    if mask.shape != prediction.shape[:2]:
        raise ValueError("velocity evaluation mask must have shape [B,N]")
    if mask.dtype != torch.bool:
        raise TypeError("velocity evaluation mask must be torch.bool")
    expanded = mask.unsqueeze(-1).expand_as(prediction)
    if not torch.isfinite(prediction.masked_select(expanded)).all():
        raise ValueError(f"{prediction_name} contains NaN or Inf under the evaluation mask")
    if not torch.isfinite(target.masked_select(expanded)).all():
        raise ValueError("target velocity contains NaN or Inf under the evaluation mask")


@dataclass
class MaskedVelocityErrorAccumulator:
    """Coordinate-wise velocity error under an externally defined object mask."""

    squared_sum: float = 0.0
    absolute_sum: float = 0.0
    coordinate_count: int = 0
    object_frame_count: int = 0
    axis_squared_sum: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    axis_absolute_sum: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    axis_count: list[int] = field(default_factory=lambda: [0, 0, 0])

    def update(self, prediction: Tensor, target: Tensor, mask: Tensor) -> None:
        _validate_velocity_inputs(
            prediction,
            target,
            mask,
            prediction_name="predicted velocity",
        )
        residual = prediction - target
        values = residual.masked_select(mask.unsqueeze(-1).expand_as(prediction))
        if values.numel() == 0:
            return
        detached = values.detach().float().cpu()
        self.squared_sum += float(detached.square().sum())
        self.absolute_sum += float(detached.abs().sum())
        self.coordinate_count += int(detached.numel())
        self.object_frame_count += int(mask.sum().detach().cpu())
        for axis in range(3):
            axis_values = residual[..., axis].masked_select(mask).detach().float().cpu()
            self.axis_squared_sum[axis] += float(axis_values.square().sum())
            self.axis_absolute_sum[axis] += float(axis_values.abs().sum())
            self.axis_count[axis] += int(axis_values.numel())

    def metrics(self, prefix: str) -> dict[str, float | None]:
        metrics: dict[str, float | None] = {}
        for axis, label in enumerate(("x", "y", "z")):
            count = self.axis_count[axis]
            metrics.update(
                {
                    f"{prefix}_velocity_{label}_rmse_mps": (
                        math.sqrt(self.axis_squared_sum[axis] / count) if count else None
                    ),
                    f"{prefix}_velocity_{label}_mae_mps": (
                        self.axis_absolute_sum[axis] / count if count else None
                    ),
                    f"{prefix}_velocity_{label}_count": float(count),
                }
            )
        if self.coordinate_count == 0:
            metrics.update(
                {
                    f"{prefix}_velocity_rmse_mps": None,
                    f"{prefix}_velocity_mae_mps": None,
                    f"{prefix}_velocity_coordinate_count": 0.0,
                    f"{prefix}_velocity_object_frame_count": 0.0,
                }
            )
            return metrics
        metrics.update(
            {
                f"{prefix}_velocity_rmse_mps": math.sqrt(self.squared_sum / self.coordinate_count),
                f"{prefix}_velocity_mae_mps": self.absolute_sum / self.coordinate_count,
                f"{prefix}_velocity_coordinate_count": float(self.coordinate_count),
                f"{prefix}_velocity_object_frame_count": float(self.object_frame_count),
            }
        )
        return metrics


@dataclass
class OrdinaryVelocityCorrectionAccumulator:
    """Prior-to-posterior velocity accuracy changes on ordinary observations."""

    prior_norm_error_sum: float = 0.0
    posterior_norm_error_sum: float = 0.0
    positive_count: int = 0
    object_update_count: int = 0
    axis_prior_absolute_sum: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    axis_posterior_absolute_sum: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    axis_positive_count: list[int] = field(default_factory=lambda: [0, 0, 0])
    axis_count: list[int] = field(default_factory=lambda: [0, 0, 0])

    def update(
        self,
        prior: Tensor,
        posterior: Tensor,
        target: Tensor,
        mask: Tensor,
    ) -> None:
        _validate_velocity_inputs(
            prior,
            target,
            mask,
            prediction_name="prior velocity",
        )
        _validate_velocity_inputs(
            posterior,
            target,
            mask,
            prediction_name="posterior velocity",
        )
        prior_error = torch.linalg.vector_norm(prior - target, dim=-1)
        posterior_error = torch.linalg.vector_norm(posterior - target, dim=-1)
        prior_values = prior_error.masked_select(mask).detach().float().cpu()
        posterior_values = posterior_error.masked_select(mask).detach().float().cpu()
        if prior_values.numel() == 0:
            return
        improvement = prior_values - posterior_values
        self.prior_norm_error_sum += float(prior_values.sum())
        self.posterior_norm_error_sum += float(posterior_values.sum())
        self.positive_count += int((improvement > 0.0).sum())
        self.object_update_count += int(improvement.numel())
        for axis in range(3):
            prior_axis = (prior[..., axis] - target[..., axis]).abs().masked_select(mask)
            posterior_axis = (posterior[..., axis] - target[..., axis]).abs().masked_select(mask)
            prior_axis = prior_axis.detach().float().cpu()
            posterior_axis = posterior_axis.detach().float().cpu()
            self.axis_prior_absolute_sum[axis] += float(prior_axis.sum())
            self.axis_posterior_absolute_sum[axis] += float(posterior_axis.sum())
            self.axis_positive_count[axis] += int((posterior_axis < prior_axis).sum())
            self.axis_count[axis] += int(prior_axis.numel())

    def metrics(self) -> dict[str, float | None]:
        metrics: dict[str, float | None] = {}
        for axis, label in enumerate(("x", "y", "z")):
            count = self.axis_count[axis]
            prior_axis = self.axis_prior_absolute_sum[axis] / count if count else None
            posterior_axis = self.axis_posterior_absolute_sum[axis] / count if count else None
            metrics.update(
                {
                    f"ordinary_velocity_{label}_prior_mae_mps": prior_axis,
                    f"ordinary_velocity_{label}_posterior_mae_mps": posterior_axis,
                    f"ordinary_velocity_{label}_improvement_mps": (
                        prior_axis - posterior_axis
                        if prior_axis is not None and posterior_axis is not None
                        else None
                    ),
                    f"ordinary_velocity_{label}_positive_rate": (
                        self.axis_positive_count[axis] / count if count else None
                    ),
                    f"ordinary_velocity_{label}_evaluated_updates": float(count),
                }
            )
        if self.object_update_count == 0:
            metrics.update(
                {
                    "ordinary_velocity_prior_norm_error_mean_mps": None,
                    "ordinary_velocity_posterior_norm_error_mean_mps": None,
                    ("ordinary_velocity_prior_to_posterior_norm_error_improvement_mean_mps"): None,
                    ("ordinary_velocity_prior_to_posterior_norm_error_improvement_fraction"): None,
                    "ordinary_velocity_prior_to_posterior_positive_rate": None,
                    "ordinary_velocity_evaluated_object_updates": 0.0,
                }
            )
            return metrics
        prior = self.prior_norm_error_sum / self.object_update_count
        posterior = self.posterior_norm_error_sum / self.object_update_count
        improvement = prior - posterior
        metrics.update(
            {
                "ordinary_velocity_prior_norm_error_mean_mps": prior,
                "ordinary_velocity_posterior_norm_error_mean_mps": posterior,
                (
                    "ordinary_velocity_prior_to_posterior_norm_error_improvement_mean_mps"
                ): improvement,
                (
                    "ordinary_velocity_prior_to_posterior_norm_error_improvement_fraction"
                ): improvement / max(prior, 1.0e-8),
                "ordinary_velocity_prior_to_posterior_positive_rate": (
                    self.positive_count / self.object_update_count
                ),
                "ordinary_velocity_evaluated_object_updates": float(self.object_update_count),
            }
        )
        return metrics


@dataclass
class TemporalVelocityMeasurementAccumulator:
    """Availability and uncertainty of explicit temporal velocity measurements."""

    inspected_update_count: int = 0
    explicit_field_update_count: int = 0
    valid_update_count: int = 0
    candidate_object_count: int = 0
    valid_object_count: int = 0
    reported_variance_sum: float = 0.0
    reported_variance_coordinate_count: int = 0
    axis_valid_coordinate_count: list[int] = field(default_factory=lambda: [0, 0, 0])

    def update(self, measurements: MeasurementSet | None) -> None:
        """Inspect one update; ``None`` denotes no fresh measurement at its timestamp."""

        self.inspected_update_count += 1
        if measurements is None:
            return
        present = tuple(key in measurements.auxiliary for key in _TEMPORAL_VELOCITY_KEYS)
        if not any(present):
            return
        if not all(present):
            missing = [
                key
                for key, is_present in zip(_TEMPORAL_VELOCITY_KEYS, present, strict=True)
                if not is_present
            ]
            raise ValueError(
                "explicit temporal velocity diagnostics require all auxiliary fields; "
                f"missing {', '.join(missing)}"
            )

        velocity = measurements.auxiliary["world_velocity"]
        log_variance = measurements.auxiliary["world_velocity_log_variance"]
        valid_mask = measurements.auxiliary["world_velocity_valid_mask"]
        expected_vector_shape = (*measurements.measurement_mask.shape, 3)
        if velocity.shape != expected_vector_shape:
            raise ValueError("auxiliary.world_velocity must have shape [B,M,3]")
        if log_variance.shape != expected_vector_shape:
            raise ValueError("auxiliary.world_velocity_log_variance must have shape [B,M,3]")
        if valid_mask.shape != measurements.measurement_mask.shape:
            raise ValueError("auxiliary.world_velocity_valid_mask must have shape [B,M]")
        if valid_mask.dtype != torch.bool:
            raise TypeError("auxiliary.world_velocity_valid_mask must be torch.bool")
        if measurements.measurement_mask.dtype != torch.bool:
            raise TypeError("measurement_mask must be torch.bool")
        axis_valid_mask = measurements.auxiliary.get(_TEMPORAL_VELOCITY_AXIS_MASK_KEY)
        if axis_valid_mask is None:
            # Checkpoints predating component-local temporal velocity support
            # used one object-valid flag to mean all three coordinates.
            axis_valid_mask = valid_mask.unsqueeze(-1).expand_as(velocity)
        else:
            if axis_valid_mask.shape != expected_vector_shape:
                raise ValueError("auxiliary.world_velocity_axis_valid_mask must have shape [B,M,3]")
            if axis_valid_mask.dtype != torch.bool:
                raise TypeError("auxiliary.world_velocity_axis_valid_mask must be torch.bool")

        self.explicit_field_update_count += 1
        self.candidate_object_count += int(measurements.measurement_mask.sum().detach().cpu())
        valid = valid_mask & measurements.measurement_mask
        valid_axes = axis_valid_mask & valid.unsqueeze(-1)
        valid_count = int(valid_axes.any(dim=-1).sum().detach().cpu())
        self.valid_object_count += valid_count
        if valid_count == 0:
            return
        selected_velocity = velocity.masked_select(valid_axes)
        selected_log_variance = log_variance.masked_select(valid_axes)
        if not torch.isfinite(selected_velocity).all():
            raise ValueError("explicit temporal velocity contains NaN or Inf where valid")
        if not torch.isfinite(selected_log_variance).all():
            raise ValueError(
                "explicit temporal velocity log variance contains NaN or Inf where valid"
            )
        variance = selected_log_variance.detach().float().cpu().clamp(-30.0, 30.0).exp()
        self.valid_update_count += 1
        self.reported_variance_sum += float(variance.sum())
        self.reported_variance_coordinate_count += int(variance.numel())
        for axis in range(3):
            self.axis_valid_coordinate_count[axis] += int(
                valid_axes[..., axis].sum().detach().cpu()
            )

    def update_direct(self, evidence: DirectVelocityEvidence | None) -> None:
        """Inspect post-association evidence aligned to persistent belief slots."""

        if evidence is None:
            return
        evidence.validate()
        axis_valid = evidence.resolved_axis_valid_mask()
        self.explicit_field_update_count += 1
        self.candidate_object_count += int(evidence.valid_mask.numel())
        valid_count = int(axis_valid.any(dim=-1).sum().detach().cpu())
        self.valid_object_count += valid_count
        if valid_count == 0:
            return
        selected_velocity = evidence.velocity.masked_select(axis_valid)
        selected_log_variance = evidence.log_variance.masked_select(axis_valid)
        if not torch.isfinite(selected_velocity).all():
            raise ValueError("direct temporal velocity contains NaN or Inf where valid")
        if not torch.isfinite(selected_log_variance).all():
            raise ValueError(
                "direct temporal velocity log variance contains NaN or Inf where valid"
            )
        variance = selected_log_variance.detach().float().cpu().clamp(-30.0, 30.0).exp()
        self.valid_update_count += 1
        self.reported_variance_sum += float(variance.sum())
        self.reported_variance_coordinate_count += int(variance.numel())
        for axis in range(3):
            self.axis_valid_coordinate_count[axis] += int(
                axis_valid[..., axis].sum().detach().cpu()
            )

    def metrics(self) -> dict[str, float | None]:
        metrics = {
            "temporal_velocity_measurement_inspected_update_count": float(
                self.inspected_update_count
            ),
            "temporal_velocity_measurement_explicit_field_update_count": float(
                self.explicit_field_update_count
            ),
            "temporal_velocity_measurement_explicit_field_update_fraction": (
                self.explicit_field_update_count / self.inspected_update_count
                if self.inspected_update_count
                else None
            ),
            "temporal_velocity_measurement_valid_update_count": float(self.valid_update_count),
            "temporal_velocity_measurement_candidate_object_count": float(
                self.candidate_object_count
            ),
            "temporal_velocity_measurement_valid_object_count": float(self.valid_object_count),
            "temporal_velocity_measurement_valid_object_fraction": (
                self.valid_object_count / self.candidate_object_count
                if self.candidate_object_count
                else None
            ),
            "temporal_velocity_measurement_reported_variance_mean_mps2": (
                self.reported_variance_sum / self.reported_variance_coordinate_count
                if self.reported_variance_coordinate_count
                else None
            ),
            "temporal_velocity_measurement_reported_variance_coordinate_count": float(
                self.reported_variance_coordinate_count
            ),
        }
        for axis, label in enumerate(("x", "y", "z")):
            count = self.axis_valid_coordinate_count[axis]
            metrics[f"temporal_velocity_measurement_{label}_valid_coordinate_count"] = float(count)
            metrics[f"temporal_velocity_measurement_{label}_valid_object_fraction"] = (
                count / self.candidate_object_count if self.candidate_object_count else None
            )
        return metrics


__all__ = [
    "MaskedVelocityErrorAccumulator",
    "OrdinaryVelocityCorrectionAccumulator",
    "TemporalVelocityMeasurementAccumulator",
]
