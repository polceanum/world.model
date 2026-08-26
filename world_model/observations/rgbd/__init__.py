"""Differentiable RGB-D measurements for known calibrated geometry."""

from world_model.observations.rgbd.sphere_centres import (
    MetricSphereCentreOutput,
    RGBDSphereCentreMeasurement,
    RGBDSphereCentreMeasurementModule,
    metric_sphere_centres_from_surface_depth,
)

__all__ = [
    "MetricSphereCentreOutput",
    "RGBDSphereCentreMeasurement",
    "RGBDSphereCentreMeasurementModule",
    "metric_sphere_centres_from_surface_depth",
]
