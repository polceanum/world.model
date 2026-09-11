"""Differentiable RGB-D measurements for known calibrated geometry."""

from world_model.observations.rgbd.module import (
    RGBDObservationConfig,
    RGBDObservationModule,
)
from world_model.observations.rgbd.open_world import (
    DiscoveredRigidObject,
    OpenWorldRigidFrame,
    OpenWorldRigidTracker,
    TrackedRigidObject,
    discover_rigid_objects_from_rgbd,
    tracked_objects_to_belief,
)
from world_model.observations.rgbd.rigid_geometry import (
    ObservableRigidGeometry,
    fit_rigid_geometry_from_rgbd,
)
from world_model.observations.rgbd.set_proposer import (
    SET_APPEARANCE_DIM,
    SET_MAX_OBJECTS,
    SET_PROPOSAL_COUNT,
    RGBDSetProposalOutput,
    RGBDSetProposer,
)
from world_model.observations.rgbd.sphere_centres import (
    MetricSphereCentreOutput,
    RGBDSphereCentreMeasurement,
    RGBDSphereCentreMeasurementModule,
    metric_sphere_centres_from_surface_depth,
)
from world_model.observations.rgbd.temporal import RGBDTemporalPositionHistory
from world_model.observations.rgbd.two_disc_geometry import (
    TwoDiscRGBDGeometryOutput,
    fit_visible_sphere_surfaces,
    two_disc_geometry_from_rgbd,
)

__all__ = [
    "MetricSphereCentreOutput",
    "ObservableRigidGeometry",
    "DiscoveredRigidObject",
    "OpenWorldRigidFrame",
    "OpenWorldRigidTracker",
    "RGBDObservationConfig",
    "RGBDObservationModule",
    "RGBDSetProposalOutput",
    "RGBDSetProposer",
    "RGBDSphereCentreMeasurement",
    "RGBDSphereCentreMeasurementModule",
    "RGBDTemporalPositionHistory",
    "SET_APPEARANCE_DIM",
    "SET_MAX_OBJECTS",
    "SET_PROPOSAL_COUNT",
    "TwoDiscRGBDGeometryOutput",
    "TrackedRigidObject",
    "discover_rigid_objects_from_rgbd",
    "fit_visible_sphere_surfaces",
    "fit_rigid_geometry_from_rgbd",
    "metric_sphere_centres_from_surface_depth",
    "two_disc_geometry_from_rgbd",
    "tracked_objects_to_belief",
]
