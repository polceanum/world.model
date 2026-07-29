"""Complete RGB observation module with global and fast residual paths."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor

from world_model.belief import MotionMode, fast_packing_map
from world_model.fusion.innovation import build_innovation
from world_model.observations.base import (
    ModalityCache,
    ModalityHistory,
    ObservationModule,
)
from world_model.observations.context import ObservationContext, SensorContext
from world_model.observations.measurements import (
    DirectVelocityEvidence,
    InnovationSet,
    MeasurementSet,
    PredictedMeasurements,
)
from world_model.observations.packets import ObservationPacket
from world_model.observations.registry import register_observation_module
from world_model.observations.rgb.backbone import RGBBackbone
from world_model.observations.rgb.cache import RGBModalityCache
from world_model.observations.rgb.global_detector import GlobalObjectDetector
from world_model.observations.rgb.losses import rgb_measurement_losses
from world_model.observations.rgb.projector import (
    RGBMeasurementProjector,
    RGBProjectorConfig,
    backproject_rgb_log_variance,
    backproject_rgb_measurements,
    calibration_tensors,
)
from world_model.observations.rgb.roi_updater import FastROIUpdater
from world_model.observations.rgb.structured_centres import (
    structured_disc_centres,
    structured_disc_centres_in_rois,
)
from world_model.observations.rgb.temporal import RGBTemporalPositionHistory

if TYPE_CHECKING:
    from world_model.belief.world_belief import WorldBelief
    from world_model.fusion.association import AssociationResult


@dataclass(frozen=True)
class RGBObservationConfig:
    max_objects: int = 8
    birth_extra_queries: int = 2
    backbone_channels: tuple[int, int, int, int] = (32, 64, 96, 128)
    feature_dim: int = 64
    appearance_dim: int = 32
    roi_size: int = 20
    roi_hidden_dim: int = 96
    fast_depth_residual_enabled: bool = False
    temporal_velocity_enabled: bool = False
    temporal_velocity_history_size: int = 3
    temporal_velocity_min_samples: int = 3
    temporal_velocity_min_dt: float = 1.0e-3
    temporal_velocity_variance_scale: float = 1.0
    temporal_velocity_variance_floor: float = 0.25
    temporal_velocity_variance_ceiling: float | None = None
    temporal_velocity_lateral_only: bool = False
    temporal_velocity_unobserved_variance: float = 1.0e4
    temporal_velocity_reset_on_collision: bool = False
    temporal_velocity_max_age_steps: int | None = None
    temporal_velocity_post_event_max_samples: int | None = None
    temporal_velocity_measurement_position_blend: float = 0.0
    temporal_velocity_position_innovation_coupling: bool = False
    temporal_position_enabled: bool = False
    temporal_position_min_samples: int = 3
    temporal_position_robust_threshold: float = 2.5
    temporal_position_variance_scale: float = 4.0
    temporal_position_variance_floor: float = 0.01
    temporal_position_variance_ceiling: float | None = None
    temporal_position_depth_only: bool = True
    structured_disc_center_enabled: bool = False
    structured_disc_threshold: float = 0.04
    structured_disc_min_pixels: int = 4
    structured_disc_max_assignment_distance: float = 0.75
    structured_disc_center_std_pixels: float = 0.75
    structured_disc_fast_depth_enabled: bool = False
    structured_disc_depth_relative_std: float | None = None
    structured_disc_depth_outlier_relative_threshold: float | None = None
    structured_disc_depth_outlier_variance_scale: float = 9.0
    structured_disc_position_confidence: float | None = None
    roi_uncertainty_scale: float = 2.5
    default_world_radius: float = 0.15
    proposal_threshold: float = 0.25
    measurement_log_variance_min: float = -8.0
    measurement_log_variance_max: float = 3.0

    def __post_init__(self) -> None:
        if self.temporal_velocity_history_size < 3:
            raise ValueError("temporal_velocity_history_size must be at least three")
        if not 2 <= self.temporal_velocity_min_samples <= self.temporal_velocity_history_size:
            raise ValueError("temporal_velocity_min_samples must lie between two and history_size")
        if not math.isfinite(self.temporal_velocity_min_dt) or (self.temporal_velocity_min_dt <= 0):
            raise ValueError("temporal_velocity_min_dt must be finite and positive")
        if not math.isfinite(self.temporal_velocity_variance_scale) or (
            self.temporal_velocity_variance_scale < 1
        ):
            raise ValueError("temporal_velocity_variance_scale must be finite and at least one")
        if not math.isfinite(self.temporal_velocity_variance_floor) or (
            self.temporal_velocity_variance_floor <= 0
        ):
            raise ValueError("temporal_velocity_variance_floor must be finite and positive")
        if self.temporal_velocity_variance_ceiling is not None and (
            not math.isfinite(self.temporal_velocity_variance_ceiling)
            or self.temporal_velocity_variance_ceiling < self.temporal_velocity_variance_floor
        ):
            raise ValueError(
                "temporal_velocity_variance_ceiling must be finite and no "
                "smaller than temporal_velocity_variance_floor"
            )
        if not math.isfinite(self.temporal_velocity_unobserved_variance) or (
            self.temporal_velocity_unobserved_variance < self.temporal_velocity_variance_floor
        ):
            raise ValueError(
                "temporal_velocity_unobserved_variance must be finite and no "
                "smaller than temporal_velocity_variance_floor"
            )
        if (
            self.temporal_velocity_max_age_steps is not None
            and self.temporal_velocity_max_age_steps < self.temporal_velocity_min_samples
        ):
            raise ValueError(
                "temporal_velocity_max_age_steps must be no smaller than "
                "temporal_velocity_min_samples"
            )
        if (
            self.temporal_velocity_post_event_max_samples is not None
            and self.temporal_velocity_post_event_max_samples < self.temporal_velocity_min_samples
        ):
            raise ValueError(
                "temporal_velocity_post_event_max_samples must be no smaller than "
                "temporal_velocity_min_samples"
            )
        if (
            not math.isfinite(self.temporal_velocity_measurement_position_blend)
            or not 0.0 <= self.temporal_velocity_measurement_position_blend <= 1.0
        ):
            raise ValueError("temporal_velocity_measurement_position_blend must lie in [0, 1]")
        if not 2 <= self.temporal_position_min_samples <= self.temporal_velocity_history_size:
            raise ValueError("temporal_position_min_samples must lie between two and history_size")
        if (
            not math.isfinite(self.temporal_position_robust_threshold)
            or self.temporal_position_robust_threshold <= 0
        ):
            raise ValueError("temporal_position_robust_threshold must be finite and positive")
        if (
            not math.isfinite(self.temporal_position_variance_scale)
            or self.temporal_position_variance_scale < 1
        ):
            raise ValueError("temporal_position_variance_scale must be finite and at least one")
        if (
            not math.isfinite(self.temporal_position_variance_floor)
            or self.temporal_position_variance_floor <= 0
        ):
            raise ValueError("temporal_position_variance_floor must be finite and positive")
        if self.temporal_position_variance_ceiling is not None and (
            not math.isfinite(self.temporal_position_variance_ceiling)
            or self.temporal_position_variance_ceiling < self.temporal_position_variance_floor
        ):
            raise ValueError(
                "temporal_position_variance_ceiling must be finite and no "
                "smaller than temporal_position_variance_floor"
            )
        if not math.isfinite(self.structured_disc_threshold) or not (
            0 < self.structured_disc_threshold < 2
        ):
            raise ValueError("structured_disc_threshold must lie in (0, 2)")
        if self.structured_disc_min_pixels <= 0:
            raise ValueError("structured_disc_min_pixels must be positive")
        if (
            not math.isfinite(self.structured_disc_max_assignment_distance)
            or self.structured_disc_max_assignment_distance <= 0
        ):
            raise ValueError("structured_disc_max_assignment_distance must be finite and positive")
        if (
            not math.isfinite(self.structured_disc_center_std_pixels)
            or self.structured_disc_center_std_pixels <= 0
        ):
            raise ValueError("structured_disc_center_std_pixels must be finite and positive")
        if self.structured_disc_depth_relative_std is not None and (
            not math.isfinite(self.structured_disc_depth_relative_std)
            or not 0.0 < self.structured_disc_depth_relative_std <= 1.0
        ):
            raise ValueError("structured_disc_depth_relative_std must lie in (0, 1]")
        if self.structured_disc_depth_outlier_relative_threshold is not None and (
            not math.isfinite(self.structured_disc_depth_outlier_relative_threshold)
            or self.structured_disc_depth_outlier_relative_threshold <= 0.0
        ):
            raise ValueError(
                "structured_disc_depth_outlier_relative_threshold must be finite and positive"
            )
        if (
            not math.isfinite(self.structured_disc_depth_outlier_variance_scale)
            or self.structured_disc_depth_outlier_variance_scale < 1.0
        ):
            raise ValueError(
                "structured_disc_depth_outlier_variance_scale must be finite and at least one"
            )
        if self.structured_disc_position_confidence is not None and (
            not math.isfinite(self.structured_disc_position_confidence)
            or not 0.0 < self.structured_disc_position_confidence <= 1.0
        ):
            raise ValueError("structured_disc_position_confidence must lie in (0, 1]")


def _packet_batch(packets: Sequence[ObservationPacket]) -> tuple[Tensor, float]:
    if not packets:
        raise ValueError("RGB observation requires at least one packet")
    timestamp = packets[0].timestamp
    if any(abs(packet.timestamp - timestamp) > 1.0e-9 for packet in packets):
        raise ValueError("batched RGB packets must share a timestamp")
    if len(packets) == 1 and isinstance(packets[0].payload, Tensor):
        image = packets[0].payload
        if image.ndim == 3:
            image = image.unsqueeze(0)
        elif image.ndim != 4:
            raise ValueError("RGB payload must be [3,H,W] or [B,3,H,W]")
    else:
        images = []
        for packet in packets:
            if not isinstance(packet.payload, Tensor) or packet.payload.ndim != 3:
                raise ValueError("multiple RGB packets must each contain a [3,H,W] Tensor")
            images.append(packet.payload)
        image = torch.stack(images, dim=0)
    return image, timestamp


@register_observation_module("rgb")
class RGBObservationModule(ObservationModule):
    """Synthetic-scene RGB measurements without simulator-state input."""

    modality_name = "rgb"
    modality_index = 0

    def __init__(self, config: RGBObservationConfig | None = None) -> None:
        super().__init__()
        self.config = config or RGBObservationConfig()
        self.backbone = RGBBackbone(
            self.config.backbone_channels,
            self.config.feature_dim,
        )
        self.global_detector = GlobalObjectDetector(
            feature_dim=self.config.feature_dim,
            query_count=self.config.max_objects + self.config.birth_extra_queries,
            appearance_dim=self.config.appearance_dim,
        )
        self.roi_updater = FastROIUpdater(
            feature_dim=self.config.feature_dim,
            appearance_dim=self.config.appearance_dim,
            roi_size=self.config.roi_size,
            hidden_dim=self.config.roi_hidden_dim,
        )
        self.projector = RGBMeasurementProjector(
            RGBProjectorConfig(
                default_radius=self.config.default_world_radius,
                uncertainty_roi_scale=self.config.roi_uncertainty_scale,
            )
        )

    def validate_packet(self, packet: ObservationPacket) -> None:
        if packet.modality != self.modality_name:
            raise ValueError(f"RGB module received modality {packet.modality!r}, expected 'rgb'")
        if not isinstance(packet.payload, Tensor):
            raise TypeError("RGB packet payload must be a torch.Tensor")
        if packet.payload.ndim not in {3, 4}:
            raise ValueError("RGB payload must be [3,H,W] or [B,3,H,W]")
        channel_dimension = 0 if packet.payload.ndim == 3 else 1
        if packet.payload.shape[channel_dimension] != 3:
            raise ValueError("RGB payload must have three channels")
        if "intrinsics" not in packet.calibration:
            raise ValueError("RGB packet calibration requires intrinsics")
        if "world_from_camera" not in packet.calibration:
            raise ValueError("RGB packet calibration requires world_from_camera")
        if not packet.payload.is_floating_point():
            raise TypeError("RGB payload must be floating point")
        if not torch.isfinite(packet.payload).all():
            raise ValueError("RGB payload contains NaN or Inf")

    @staticmethod
    def _structured_inverse_depth(
        log_radius: Tensor,
        intrinsics: Tensor,
        image_size: tuple[int, int],
        world_radius: float,
        residual: Tensor,
    ) -> Tensor:
        height, width = image_size
        radius_pixels = log_radius.exp().squeeze(-1) * (0.5 * min(height, width))
        focal = 0.5 * (intrinsics[:, None, 0, 0] + intrinsics[:, None, 1, 1])
        analytic = radius_pixels / (focal * world_radius)
        return (analytic + residual.squeeze(-1)).clamp(1.0e-3, 20.0)

    def _global_measurements(
        self,
        image: Tensor,
        packet: ObservationPacket,
    ) -> MeasurementSet:
        feature_pyramid = self.backbone(image)
        output = self.global_detector(feature_pyramid["full"])
        batch = image.shape[0]
        image_size = (image.shape[-2], image.shape[-1])
        world_from_camera, intrinsics = calibration_tensors(
            packet.calibration,
            batch=batch,
            device=image.device,
            dtype=image.dtype,
        )
        centre = output.centre
        raw_centre = output.centre
        structured_valid = torch.zeros(
            output.centre.shape[:2],
            device=image.device,
            dtype=torch.bool,
        )
        structured_depth_valid = torch.zeros_like(structured_valid)
        structured_count = torch.zeros(
            batch,
            device=image.device,
            dtype=torch.int64,
        )
        log_radius = output.log_radius
        if self.config.structured_disc_center_enabled:
            structured = structured_disc_centres(
                image,
                output.centre,
                threshold=self.config.structured_disc_threshold,
                minimum_pixels=self.config.structured_disc_min_pixels,
                maximum_assignment_distance=(self.config.structured_disc_max_assignment_distance),
            )
            # The RGB component operation is detached.  This straight-through
            # residual retains gradients for the learned discovery head while
            # using the directly observed pixel centroid in the forward pass.
            centre = output.centre + (structured.centres - output.centre).detach()
            normalized_radius = (
                structured.radius_pixels / (0.5 * min(image.shape[-2], image.shape[-1]))
            ).clamp_min(1.0e-4)
            structured_log_radius = normalized_radius.log().unsqueeze(-1)
            log_radius = torch.where(
                structured.valid_mask.unsqueeze(-1),
                output.log_radius + (structured_log_radius - output.log_radius).detach(),
                output.log_radius,
            )
            structured_valid = structured.valid_mask
            structured_depth_valid = structured.depth_valid_mask
            structured_count = structured.component_count
        inverse_depth = self._structured_inverse_depth(
            log_radius,
            intrinsics,
            image_size,
            self.config.default_world_radius,
            output.inverse_depth_residual,
        )
        if self.config.structured_disc_depth_relative_std is not None:
            analytic_inverse_depth = self._structured_inverse_depth(
                log_radius,
                intrinsics,
                image_size,
                self.config.default_world_radius,
                torch.zeros_like(output.inverse_depth_residual),
            )
            inverse_depth = torch.where(
                structured_valid,
                analytic_inverse_depth,
                inverse_depth,
            )
        values = torch.cat(
            (
                centre,
                log_radius,
                inverse_depth.unsqueeze(-1),
                output.colour,
            ),
            dim=-1,
        )
        measurement_log_variance = output.log_variance.clamp(
            self.config.measurement_log_variance_min,
            self.config.measurement_log_variance_max,
        )
        if self.config.structured_disc_center_enabled:
            height, width = image_size
            pixel_std = self.config.structured_disc_center_std_pixels
            centre_variance = image.new_tensor(
                (
                    (2.0 * pixel_std / max(width - 1, 1)) ** 2,
                    (2.0 * pixel_std / max(height - 1, 1)) ** 2,
                )
            )
            structured_log_variance = centre_variance.log().view(1, 1, 2)
            centre_log_variance = torch.where(
                structured_valid.unsqueeze(-1),
                structured_log_variance,
                measurement_log_variance[..., :2],
            )
            measurement_log_variance = torch.cat(
                (centre_log_variance, measurement_log_variance[..., 2:]),
                dim=-1,
            )
            if self.config.structured_disc_depth_relative_std is not None:
                relative_variance = self.config.structured_disc_depth_relative_std**2
                radius_log_variance = image.new_full(
                    log_radius.shape,
                    math.log(relative_variance),
                )
                inverse_depth_log_variance = (
                    (inverse_depth.square() * relative_variance)
                    .clamp_min(1.0e-10)
                    .log()
                    .unsqueeze(-1)
                )
                measurement_log_variance = measurement_log_variance.clone()
                measurement_log_variance[..., 2:3] = torch.where(
                    structured_valid.unsqueeze(-1),
                    radius_log_variance,
                    measurement_log_variance[..., 2:3],
                )
                measurement_log_variance[..., 3:4] = torch.where(
                    structured_valid.unsqueeze(-1),
                    inverse_depth_log_variance,
                    measurement_log_variance[..., 3:4],
                )
        world_position = backproject_rgb_measurements(
            values,
            world_from_camera,
            intrinsics,
            image_size,
        )
        world_position_log_variance = backproject_rgb_log_variance(
            values,
            measurement_log_variance,
            world_from_camera,
            intrinsics,
            image_size,
        )
        confidence_bias = torch.log(
            torch.tensor(
                packet.confidence,
                device=image.device,
                dtype=image.dtype,
            ).clamp_min(1.0e-4)
        )
        existence_logits = output.existence_logits + confidence_bias
        position_confidence = existence_logits.sigmoid()
        if self.config.structured_disc_position_confidence is not None:
            position_confidence = torch.where(
                structured_valid,
                torch.full_like(
                    position_confidence,
                    self.config.structured_disc_position_confidence,
                ),
                position_confidence,
            )
        # Keep all proposals available to Hungarian matching; lifecycle applies
        # the confidence threshold for births.
        measurement_mask = torch.ones_like(existence_logits, dtype=torch.bool)
        timestamp = image.new_full((batch,), packet.timestamp)
        result = MeasurementSet(
            modality=self.modality_name,
            sensor_id=packet.sensor_id,
            timestamp=timestamp,
            values=values,
            log_variance=measurement_log_variance,
            existence_logits=existence_logits,
            measurement_mask=measurement_mask,
            appearance=output.appearance,
            class_logits=None,
            frame_id=packet.frame_id,
            supported_state_fields=(
                (
                    "position",
                    "velocity_from_position",
                    "geometry",
                    "appearance",
                )
                if (
                    not self.config.temporal_velocity_enabled
                    or self.config.temporal_velocity_position_innovation_coupling
                )
                else ("position", "geometry", "appearance")
            ),
            auxiliary={
                "world_position": world_position,
                "world_radius": image.new_full(
                    (*world_position.shape[:2], 1),
                    self.config.default_world_radius,
                ),
                "world_log_variance": world_position_log_variance,
                "world_position_log_variance": world_position_log_variance,
                "position_confidence": position_confidence,
                "visibility_logit": output.visibility_logits,
                "visibility_logits": output.visibility_logits,
                "query_features": output.query_features,
                "attention": output.attention,
                "raw_centre": raw_centre,
                "structured_centre_valid": structured_valid,
                "structured_depth_valid": structured_depth_valid,
                "structured_component_count": structured_count,
            },
        )
        result.validate()
        return result

    def initialise_measurements(
        self,
        packets: Sequence[ObservationPacket],
        context: ObservationContext,
    ) -> MeasurementSet:
        del context
        for packet in packets:
            self.validate_packet(packet)
        image, _ = _packet_batch(packets)
        return self._global_measurements(image, packets[0])

    def encode_measurements(
        self,
        packets: Sequence[ObservationPacket],
        prior: WorldBelief,
        predicted: PredictedMeasurements,
        cache: ModalityCache | None,
    ) -> tuple[MeasurementSet, ModalityCache]:
        del prior
        for packet in packets:
            self.validate_packet(packet)
        image, timestamp = _packet_batch(packets)
        feature_map = self.backbone.forward_fast(image)["stage2"]
        if predicted.rois is None:
            raise ValueError("RGB fast path requires projected ROIs")
        previous_features = None
        if isinstance(cache, RGBModalityCache):
            cached_features = cache.object_features
            if (
                cached_features.shape[:2] == predicted.object_ids.shape
                and cached_features.device == feature_map.device
                and cached_features.dtype == feature_map.dtype
                and cache.object_ids.shape == predicted.object_ids.shape
                and cache.object_ids.device == predicted.object_ids.device
            ):
                same_identity = (predicted.object_ids >= 0) & (
                    cache.object_ids == predicted.object_ids
                )
                previous_features = torch.where(
                    same_identity.unsqueeze(-1),
                    cached_features,
                    torch.zeros_like(cached_features),
                )
        output = self.roi_updater(
            feature_map,
            predicted.rois,
            predicted.values,
            previous_object_features=previous_features,
            valid_mask=predicted.valid_mask,
        )
        values = output.values
        raw_centre = output.values[..., :2]
        if not self.config.fast_depth_residual_enabled:
            # Depth is substantially less observable from a small residual
            # crop than centre offset.  Keep the analytic predicted depth until
            # a trained checkpoint passes a held-out per-mode improvement gate.
            values = torch.cat(
                (
                    values[..., :3],
                    predicted.values[..., 3:4],
                    values[..., 4:],
                ),
                dim=-1,
            )
        batch, objects, _ = output.values.shape
        world_from_camera = predicted.auxiliary["world_from_camera"][:, 0]
        intrinsics = predicted.auxiliary["intrinsics"][:, 0]
        image_size = (image.shape[-2], image.shape[-1])
        structured_valid = torch.zeros(
            values.shape[:2],
            device=image.device,
            dtype=torch.bool,
        )
        structured_count = torch.zeros(
            values.shape[0],
            device=image.device,
            dtype=torch.int64,
        )
        structured_depth_valid = torch.zeros_like(structured_valid)
        if self.config.structured_disc_center_enabled:
            structured = structured_disc_centres_in_rois(
                image,
                predicted.values[..., :2],
                predicted.rois,
                valid_mask=predicted.valid_mask,
                output_size=max(self.config.roi_size, 16),
                threshold=self.config.structured_disc_threshold,
                minimum_pixels=self.config.structured_disc_min_pixels,
                maximum_assignment_distance=(self.config.structured_disc_max_assignment_distance),
            )
            centre = values[..., :2] + (structured.centres - values[..., :2]).detach()
            values = torch.cat((centre, values[..., 2:]), dim=-1)
            structured_valid = structured.valid_mask
            structured_count = structured.valid_mask.sum(dim=-1)
            if self.config.structured_disc_fast_depth_enabled:
                normalized_radius = (structured.radius_pixels / (0.5 * min(image_size))).clamp_min(
                    1.0e-4
                )
                structured_log_radius = normalized_radius.log().unsqueeze(-1)
                analytic_inverse_depth = self._structured_inverse_depth(
                    structured_log_radius,
                    intrinsics,
                    image_size,
                    self.config.default_world_radius,
                    torch.zeros_like(values[..., 3:4]),
                ).unsqueeze(-1)
                depth_valid = structured.depth_valid_mask.unsqueeze(-1)
                values = values.clone()
                values[..., 2:3] = torch.where(
                    depth_valid,
                    structured_log_radius,
                    values[..., 2:3],
                )
                values[..., 3:4] = torch.where(
                    depth_valid,
                    analytic_inverse_depth,
                    values[..., 3:4],
                )
                structured_depth_valid = structured.depth_valid_mask
        measurement_log_variance = output.log_variance.clamp(
            self.config.measurement_log_variance_min,
            self.config.measurement_log_variance_max,
        )
        if self.config.structured_disc_center_enabled:
            height, width = image_size
            pixel_std = self.config.structured_disc_center_std_pixels
            centre_variance = image.new_tensor(
                (
                    (2.0 * pixel_std / max(width - 1, 1)) ** 2,
                    (2.0 * pixel_std / max(height - 1, 1)) ** 2,
                )
            )
            structured_log_variance = centre_variance.log().view(1, 1, 2)
            centre_log_variance = torch.where(
                structured_valid.unsqueeze(-1),
                structured_log_variance,
                measurement_log_variance[..., :2],
            )
            measurement_log_variance = torch.cat(
                (centre_log_variance, measurement_log_variance[..., 2:]),
                dim=-1,
            )
            if (
                self.config.structured_disc_fast_depth_enabled
                and self.config.structured_disc_depth_relative_std is not None
            ):
                relative_variance = self.config.structured_disc_depth_relative_std**2
                radius_log_variance = image.new_full(
                    values[..., 2:3].shape,
                    math.log(relative_variance),
                )
                inverse_depth_log_variance = (
                    (values[..., 3:4].square() * relative_variance).clamp_min(1.0e-10).log()
                )
                depth_valid = structured_depth_valid.unsqueeze(-1)
                measurement_log_variance = measurement_log_variance.clone()
                measurement_log_variance[..., 2:3] = torch.where(
                    depth_valid,
                    radius_log_variance,
                    measurement_log_variance[..., 2:3],
                )
                measurement_log_variance[..., 3:4] = torch.where(
                    depth_valid,
                    inverse_depth_log_variance,
                    measurement_log_variance[..., 3:4],
                )
        world_position = backproject_rgb_measurements(
            values,
            world_from_camera,
            intrinsics,
            image_size,
        )
        world_position_log_variance = backproject_rgb_log_variance(
            values,
            measurement_log_variance,
            world_from_camera,
            intrinsics,
            image_size,
        )
        prior_appearance = predicted.appearance
        if prior_appearance is not None:
            dims = min(prior_appearance.shape[-1], output.appearance.shape[-1])
            mixed_appearance = prior_appearance[..., :dims] + (
                output.appearance_gate
                * (output.appearance[..., :dims] - prior_appearance[..., :dims])
            )
            appearance = F.normalize(mixed_appearance, dim=-1)
        else:
            appearance = output.appearance
        existence_logits = output.existence_logits + torch.logit(
            predicted.visibility.clamp(1.0e-4, 1.0 - 1.0e-4)
        )
        supported_state_fields = [
            "position",
            "geometry",
            "appearance",
        ]
        if (
            not self.config.temporal_velocity_enabled
            or self.config.temporal_velocity_position_innovation_coupling
        ):
            supported_state_fields.insert(1, "velocity_from_position")
        auxiliary = {
            "world_position": world_position,
            "world_radius": predicted.auxiliary["world_radius"],
            "world_log_variance": world_position_log_variance,
            "world_position_log_variance": world_position_log_variance,
            "visibility_logit": output.visibility_logits,
            "visibility_logits": output.visibility_logits,
            "event_features": output.event_features,
            "appearance_gate": output.appearance_gate,
            "raw_centre": raw_centre,
            "structured_centre_valid": structured_valid,
            "structured_depth_valid": structured_depth_valid,
            "structured_component_count": structured_count,
        }
        measurement = MeasurementSet(
            modality=self.modality_name,
            sensor_id=packets[0].sensor_id,
            timestamp=image.new_full((batch,), timestamp),
            values=values,
            log_variance=measurement_log_variance,
            existence_logits=existence_logits,
            measurement_mask=predicted.valid_mask.clone(),
            appearance=appearance,
            class_logits=None,
            frame_id=packets[0].frame_id,
            supported_state_fields=tuple(supported_state_fields),
            auxiliary=auxiliary,
        )
        measurement.validate()
        object_ids = predicted.object_ids
        new_cache = RGBModalityCache(
            feature_map=feature_map,
            object_features=output.object_features,
            rois=predicted.rois,
            support=output.support,
            previous_image=image,
            timestamp=timestamp,
            object_ids=object_ids,
        )
        return measurement, new_cache

    def update_temporal_history(
        self,
        *,
        posterior: WorldBelief,
        measured: MeasurementSet,
        association: AssociationResult,
        history: ModalityHistory | None,
    ) -> tuple[DirectVelocityEvidence | None, ModalityHistory | None]:
        """Update same-ID corrected-position history and emit a causal LS slope."""

        if not (self.config.temporal_velocity_enabled or self.config.temporal_position_enabled):
            return None, history
        object_ids = posterior.objects.object_id
        active_mask = posterior.objects.active
        observed_mask = torch.zeros_like(active_mask)
        confidence = posterior.objects.position.new_zeros(active_mask.shape)
        batch_index, pair_index = torch.nonzero(association.pair_mask, as_tuple=True)
        accepted_batch = batch_index[:0]
        accepted_belief = batch_index[:0]
        accepted_measurement = batch_index[:0]
        if batch_index.numel():
            belief_index = association.belief_indices[batch_index, pair_index]
            measurement_index = association.measurement_indices[batch_index, pair_index]
            nonambiguous = ~association.ambiguous[batch_index, pair_index]
            accepted_batch = batch_index[nonambiguous]
            accepted_belief = belief_index[nonambiguous]
            accepted_measurement = measurement_index[nonambiguous]
            observed_mask[accepted_batch, accepted_belief] = True
            confidence[accepted_batch, accepted_belief] = measured.existence_logits[
                accepted_batch,
                accepted_measurement,
            ].sigmoid()

        position_slice = fast_packing_map(posterior.objects)["position"]
        position_log_variance = posterior.objects.fast_log_variance[..., position_slice]
        history_positions = posterior.objects.position
        history_position_log_variance = position_log_variance
        measured_world_position = measured.auxiliary.get("world_position")
        measured_world_position_log_variance = measured.auxiliary.get("world_position_log_variance")
        if (
            self.config.temporal_velocity_measurement_position_blend > 0.0
            and measured_world_position is not None
            and measured_world_position_log_variance is not None
        ):
            if measured_world_position.shape != (*measured.values.shape[:2], 3):
                raise ValueError("RGB auxiliary.world_position must have shape [B,M,3]")
            if measured_world_position_log_variance.shape != measured_world_position.shape:
                raise ValueError(
                    "RGB auxiliary.world_position_log_variance must match world_position"
                )
            history_positions = history_positions.clone()
            history_position_log_variance = history_position_log_variance.clone()
            if accepted_batch.numel():
                blend = self.config.temporal_velocity_measurement_position_blend
                prior_position = history_positions[accepted_batch, accepted_belief]
                measured_position = measured_world_position[
                    accepted_batch,
                    accepted_measurement,
                ]
                history_positions[accepted_batch, accepted_belief] = prior_position + blend * (
                    measured_position - prior_position
                )
                prior_variance = history_position_log_variance[
                    accepted_batch,
                    accepted_belief,
                ].exp()
                measured_variance = measured_world_position_log_variance[
                    accepted_batch,
                    accepted_measurement,
                ].exp()
                history_position_log_variance[accepted_batch, accepted_belief] = (
                    ((1.0 - blend) * prior_variance + blend * measured_variance)
                    .clamp_min(1.0e-10)
                    .log()
                )
        scale_valid_mask = torch.zeros_like(active_mask)
        measured_scale_valid = measured.auxiliary.get("structured_depth_valid")
        if measured_scale_valid is not None:
            if (
                measured_scale_valid.shape != measured.measurement_mask.shape
                or measured_scale_valid.dtype != torch.bool
            ):
                raise ValueError("RGB auxiliary.structured_depth_valid must be boolean [B,M]")
            if accepted_batch.numel():
                accepted_scale_valid = measured_scale_valid[
                    accepted_batch,
                    accepted_measurement,
                ]
                scale_valid_mask[accepted_batch, accepted_belief] = accepted_scale_valid
                if (
                    self.config.temporal_position_enabled
                    and measured_world_position is not None
                    and measured_world_position_log_variance is not None
                ):
                    scale_batch = accepted_batch[accepted_scale_valid]
                    scale_belief = accepted_belief[accepted_scale_valid]
                    scale_measurement = accepted_measurement[accepted_scale_valid]
                    history_positions = history_positions.clone()
                    history_position_log_variance = history_position_log_variance.clone()
                    history_positions[scale_batch, scale_belief] = measured_world_position[
                        scale_batch,
                        scale_measurement,
                    ]
                    history_position_log_variance[scale_batch, scale_belief] = (
                        measured_world_position_log_variance[
                            scale_batch,
                            scale_measurement,
                        ]
                    )
        if (
            not isinstance(history, RGBTemporalPositionHistory)
            or history.history_size != self.config.temporal_velocity_history_size
        ):
            history = RGBTemporalPositionHistory.empty(
                object_ids=object_ids,
                active_mask=active_mask,
                history_size=self.config.temporal_velocity_history_size,
                dtype=posterior.dtype,
            )
        reset_mask = torch.zeros_like(active_mask)
        if self.config.temporal_velocity_reset_on_collision:
            reset_mask = posterior.objects.motion_mode_logits.argmax(dim=-1) == int(
                MotionMode.COLLISION
            )
        history = history.append(
            object_ids=object_ids,
            active_mask=active_mask,
            observed_mask=observed_mask,
            scale_valid_mask=scale_valid_mask,
            reset_mask=reset_mask,
            timestamp=measured.timestamp,
            positions=history_positions,
            position_log_variance=history_position_log_variance,
            minimum_dt=self.config.temporal_velocity_min_dt,
        )
        velocity, velocity_log_variance, velocity_valid_mask = history.least_squares_velocity(
            minimum_dt=self.config.temporal_velocity_min_dt,
            minimum_samples=self.config.temporal_velocity_min_samples,
            variance_scale=self.config.temporal_velocity_variance_scale,
            variance_floor=self.config.temporal_velocity_variance_floor,
            variance_ceiling=self.config.temporal_velocity_variance_ceiling,
        )
        velocity_valid_mask = velocity_valid_mask & observed_mask & active_mask
        if not self.config.temporal_velocity_enabled:
            velocity_valid_mask = torch.zeros_like(velocity_valid_mask)
        if self.config.temporal_velocity_max_age_steps is not None:
            lifetime_window = (
                posterior.objects.age_steps <= self.config.temporal_velocity_max_age_steps
            )
            post_event_window = torch.zeros_like(lifetime_window)
            if self.config.temporal_velocity_post_event_max_samples is not None:
                post_event_window = history.has_reset & (
                    history.post_reset_sample_count
                    <= self.config.temporal_velocity_post_event_max_samples
                )
            velocity_valid_mask = velocity_valid_mask & (lifetime_window | post_event_window)
        if self.config.temporal_velocity_lateral_only:
            camera_lateral = posterior.camera.world_from_camera[:, :3, 0]
            camera_lateral = F.normalize(camera_lateral, dim=-1).unsqueeze(1)
            prior_velocity = posterior.objects.velocity
            lateral_delta = ((velocity - prior_velocity) * camera_lateral).sum(dim=-1, keepdim=True)
            velocity = prior_velocity + camera_lateral * lateral_delta

            scalar_variance = (velocity_log_variance.exp() * camera_lateral.square()).sum(
                dim=-1, keepdim=True
            )
            observable = camera_lateral.square() >= 1.0e-4
            projected_variance = scalar_variance / camera_lateral.square().clamp_min(1.0e-4)
            unobserved_variance = torch.full_like(
                projected_variance,
                self.config.temporal_velocity_unobserved_variance,
            )
            velocity_log_variance = torch.where(
                observable,
                projected_variance,
                unobserved_variance,
            ).log()

        measurement_velocity = measured.values.new_zeros((*measured.values.shape[:2], 3))
        measurement_log_variance = measured.values.new_full(
            (*measured.values.shape[:2], 3),
            math.log(self.config.temporal_velocity_variance_floor),
        )
        measurement_valid = torch.zeros_like(measured.measurement_mask)
        if batch_index.numel():
            evidence_valid = velocity_valid_mask[batch_index, belief_index] & (
                ~association.ambiguous[batch_index, pair_index]
            )
            valid_batch = batch_index[evidence_valid]
            valid_belief = belief_index[evidence_valid]
            valid_measurement = measurement_index[evidence_valid]
            measurement_velocity[valid_batch, valid_measurement] = velocity[
                valid_batch,
                valid_belief,
            ]
            measurement_log_variance[valid_batch, valid_measurement] = velocity_log_variance[
                valid_batch, valid_belief
            ]
            measurement_valid[valid_batch, valid_measurement] = True
        measured.auxiliary.update(
            {
                "world_velocity": measurement_velocity,
                "world_velocity_log_variance": measurement_log_variance,
                "world_velocity_valid_mask": measurement_valid,
            }
        )
        trajectory_position = None
        trajectory_position_log_variance = None
        trajectory_position_valid_mask = None
        if self.config.temporal_position_enabled:
            (
                trajectory_position,
                trajectory_position_log_variance,
                trajectory_position_valid_mask,
            ) = history.robust_trajectory_position(
                query_timestamp=measured.timestamp,
                minimum_dt=self.config.temporal_velocity_min_dt,
                minimum_samples=self.config.temporal_position_min_samples,
                robust_threshold=self.config.temporal_position_robust_threshold,
                variance_scale=self.config.temporal_position_variance_scale,
                variance_floor=self.config.temporal_position_variance_floor,
                variance_ceiling=self.config.temporal_position_variance_ceiling,
            )
            trajectory_position_valid_mask = (
                trajectory_position_valid_mask & observed_mask & active_mask
            )
            if self.config.temporal_position_depth_only:
                camera_depth_axis = F.normalize(
                    posterior.camera.world_from_camera[:, :3, 2],
                    dim=-1,
                ).unsqueeze(1)
                prior_position = posterior.objects.position
                depth_delta = ((trajectory_position - prior_position) * camera_depth_axis).sum(
                    dim=-1, keepdim=True
                )
                trajectory_position = prior_position + camera_depth_axis * depth_delta
                scalar_variance = (
                    trajectory_position_log_variance.exp() * camera_depth_axis.square()
                ).sum(dim=-1, keepdim=True)
                observable = camera_depth_axis.square() >= 1.0e-4
                projected_variance = scalar_variance / camera_depth_axis.square().clamp_min(1.0e-4)
                unobserved_variance = torch.full_like(
                    projected_variance,
                    self.config.temporal_velocity_unobserved_variance,
                )
                trajectory_position_log_variance = torch.where(
                    observable,
                    projected_variance,
                    unobserved_variance,
                ).log()
            measurement_position = measured.values.new_zeros((*measured.values.shape[:2], 3))
            measurement_position_log_variance = measured.values.new_full(
                (*measured.values.shape[:2], 3),
                math.log(self.config.temporal_position_variance_floor),
            )
            measurement_position_valid = torch.zeros_like(measured.measurement_mask)
            if batch_index.numel():
                position_evidence_valid = trajectory_position_valid_mask[
                    batch_index,
                    belief_index,
                ] & (~association.ambiguous[batch_index, pair_index])
                position_batch = batch_index[position_evidence_valid]
                position_belief = belief_index[position_evidence_valid]
                position_measurement = measurement_index[position_evidence_valid]
                measurement_position[position_batch, position_measurement] = trajectory_position[
                    position_batch, position_belief
                ]
                measurement_position_log_variance[
                    position_batch,
                    position_measurement,
                ] = trajectory_position_log_variance[
                    position_batch,
                    position_belief,
                ]
                measurement_position_valid[position_batch, position_measurement] = True
            measured.auxiliary.update(
                {
                    "world_trajectory_position": measurement_position,
                    "world_trajectory_position_log_variance": (measurement_position_log_variance),
                    "world_trajectory_position_valid_mask": measurement_position_valid,
                }
            )

        evidence = DirectVelocityEvidence(
            velocity=velocity,
            log_variance=velocity_log_variance,
            valid_mask=velocity_valid_mask,
            confidence=confidence,
            position=trajectory_position,
            position_log_variance=trajectory_position_log_variance,
            position_valid_mask=trajectory_position_valid_mask,
        )
        evidence.validate()
        return evidence, history

    def project(
        self,
        belief: WorldBelief,
        sensor_context: SensorContext,
    ) -> PredictedMeasurements:
        return self.projector(belief, sensor_context)

    def innovation(
        self,
        measured: MeasurementSet,
        predicted: PredictedMeasurements,
        association: AssociationResult,
    ) -> InnovationSet:
        result = build_innovation(
            measured=measured,
            predicted=predicted,
            association=association,
            modality_index=self.modality_index,
        )
        threshold = self.config.structured_disc_depth_outlier_relative_threshold
        if threshold is None:
            return result
        measured_values = result.auxiliary["measured_values"]
        predicted_values = result.auxiliary["predicted_values"]
        structured_valid = result.auxiliary.get("measured_structured_centre_valid")
        position_log_variance = result.auxiliary.get("measured_world_position_log_variance")
        if structured_valid is None or position_log_variance is None:
            return result
        measured_inverse_depth = measured_values[..., 3]
        predicted_inverse_depth = predicted_values[..., 3].clamp_min(1.0e-4)
        relative_disagreement = (
            measured_inverse_depth - predicted_inverse_depth
        ).abs() / predicted_inverse_depth
        outlier = result.pair_mask & structured_valid & (relative_disagreement > threshold)
        variance_inflation = math.log(self.config.structured_disc_depth_outlier_variance_scale)
        result.auxiliary["measured_world_position_log_variance"] = torch.where(
            outlier.unsqueeze(-1),
            position_log_variance + variance_inflation,
            position_log_variance,
        )
        result.auxiliary["measured_depth_outlier_mask"] = outlier
        result.auxiliary["measured_depth_relative_disagreement"] = relative_disagreement
        return result

    def training_losses(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor],
        masks: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        return rgb_measurement_losses(outputs, targets, masks)
