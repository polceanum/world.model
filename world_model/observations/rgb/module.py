"""Complete RGB observation module with global and fast residual paths."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor

from world_model.fusion.innovation import build_innovation
from world_model.observations.base import ModalityCache, ObservationModule
from world_model.observations.context import ObservationContext, SensorContext
from world_model.observations.measurements import (
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
    roi_uncertainty_scale: float = 2.5
    default_world_radius: float = 0.15
    proposal_threshold: float = 0.25
    measurement_log_variance_min: float = -8.0
    measurement_log_variance_max: float = 3.0


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
        inverse_depth = self._structured_inverse_depth(
            output.log_radius,
            intrinsics,
            image_size,
            self.config.default_world_radius,
            output.inverse_depth_residual,
        )
        values = torch.cat(
            (
                output.centre,
                output.log_radius,
                inverse_depth.unsqueeze(-1),
                output.colour,
            ),
            dim=-1,
        )
        measurement_log_variance = output.log_variance.clamp(
            self.config.measurement_log_variance_min,
            self.config.measurement_log_variance_max,
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
                "position",
                "velocity_from_position",
                "geometry",
                "appearance",
            ),
            auxiliary={
                "world_position": world_position,
                "world_radius": image.new_full(
                    (*world_position.shape[:2], 1),
                    self.config.default_world_radius,
                ),
                "world_log_variance": world_position_log_variance,
                "world_position_log_variance": world_position_log_variance,
                "visibility_logit": output.visibility_logits,
                "visibility_logits": output.visibility_logits,
                "query_features": output.query_features,
                "attention": output.attention,
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
        previous_features = cache.object_features if isinstance(cache, RGBModalityCache) else None
        output = self.roi_updater(
            feature_map,
            predicted.rois,
            predicted.values,
            previous_object_features=previous_features,
            valid_mask=predicted.valid_mask,
        )
        values = output.values
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
        measurement_log_variance = output.log_variance.clamp(
            self.config.measurement_log_variance_min,
            self.config.measurement_log_variance_max,
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
            supported_state_fields=(
                "position",
                "velocity_from_position",
                "geometry",
                "appearance",
            ),
            auxiliary={
                "world_position": world_position,
                "world_radius": predicted.auxiliary["world_radius"],
                "world_log_variance": world_position_log_variance,
                "world_position_log_variance": world_position_log_variance,
                "visibility_logit": output.visibility_logits,
                "visibility_logits": output.visibility_logits,
                "event_features": output.event_features,
                "appearance_gate": output.appearance_gate,
            },
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
        return build_innovation(
            measured=measured,
            predicted=predicted,
            association=association,
            modality_index=self.modality_index,
        )

    def training_losses(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor],
        masks: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        return rgb_measurement_losses(outputs, targets, masks)
