"""Differentiable RGB-D measurements for known calibrated geometry."""

from world_model.observations.rgbd.module import (
    RGBDObservationConfig,
    RGBDObservationModule,
)
from world_model.observations.rgbd.sphere_centres import (
    MetricSphereCentreOutput,
    RGBDSphereCentreMeasurement,
    RGBDSphereCentreMeasurementModule,
    metric_sphere_centres_from_surface_depth,
)
from world_model.observations.rgbd.temporal import RGBDTemporalPositionHistory

__all__ = [
    "MetricSphereCentreOutput",
    "RGBDObservationConfig",
    "RGBDObservationModule",
    "RGBDSphereCentreMeasurement",
    "RGBDSphereCentreMeasurementModule",
    "RGBDTemporalPositionHistory",
    "metric_sphere_centres_from_surface_depth",
]
