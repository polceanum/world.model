"""RGB global discovery, projection, and residual ROI measurement path."""

from world_model.observations.rgb.backbone import RGBBackbone
from world_model.observations.rgb.cache import RGBModalityCache
from world_model.observations.rgb.global_detector import (
    GlobalDetectorOutput,
    GlobalObjectDetector,
)
from world_model.observations.rgb.module import RGBObservationConfig, RGBObservationModule
from world_model.observations.rgb.projector import (
    RGBMeasurementProjector,
    RGBProjectorConfig,
    backproject_rgb_log_variance,
    backproject_rgb_measurements,
    depth_ordered_circle_occlusion,
    project_world_points,
)
from world_model.observations.rgb.roi_updater import (
    FastROIUpdater,
    ROIUpdateOutput,
    make_roi_grid,
    sample_rois,
)

__all__ = [
    "FastROIUpdater",
    "GlobalDetectorOutput",
    "GlobalObjectDetector",
    "RGBBackbone",
    "RGBMeasurementProjector",
    "RGBModalityCache",
    "RGBObservationConfig",
    "RGBObservationModule",
    "RGBProjectorConfig",
    "ROIUpdateOutput",
    "backproject_rgb_log_variance",
    "backproject_rgb_measurements",
    "depth_ordered_circle_occlusion",
    "make_roi_grid",
    "project_world_points",
    "sample_rois",
]
