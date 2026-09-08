"""Public RGB-D observation module.

The module consumes one composite, strictly batched RGB-D packet.  It exposes
the qualified differentiable metric sphere centre as an ordinary
``MeasurementSet`` and derives velocity only from a bounded uniform history of
raw associated metric positions.  No renderer labels, instance maps, object
IDs, or simulator state enter the observation path.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor

from world_model.belief import MotionMode, fast_packing_map
from world_model.dynamics import FreeMotionFitResult
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
from world_model.observations.rgb.soft_geometry import (
    SoftDiscGeometryOutput,
    soft_disc_geometry_from_rgb,
)
from world_model.observations.rgbd.set_proposer import (
    SET_APPEARANCE_DIM,
    SET_FEATURE_DIM,
    SET_FOREGROUND_TEMPERATURE_FLOOR,
    SET_MAX_CONFIGURABLE_LOG_VARIANCE_RESIDUAL,
    SET_MAX_LOG_VARIANCE_RESIDUAL,
    SET_MAX_OBJECTS,
    SET_PROPOSAL_COUNT,
    RGBDSetProposer,
)
from world_model.observations.rgbd.sphere_centres import (
    MAXIMUM_METRIC_DISTANCE_M,
    RGBDSphereCentreMeasurementModule,
    metric_sphere_centres_from_surface_depth,
)
from world_model.observations.rgbd.temporal import RGBDTemporalPositionHistory
from world_model.observations.rgbd.two_disc_geometry import (
    fit_visible_sphere_surfaces,
    two_disc_geometry_from_rgbd,
)

if TYPE_CHECKING:
    from world_model.belief import BirthAssignments, WorldBelief
    from world_model.fusion.association import AssociationResult


@dataclass(frozen=True)
class RGBDObservationConfig:
    """Checkpointed priors and numerical controls for the first RGB-D bridge."""

    proposal_count: int = 1
    appearance_dim: int = 32
    chromatic_temperature: float = 0.05
    minimum_chromatic_eigengap: float = 0.01
    spatial_temperature_pixels: float = 1.0
    chromatic_centre_blend: float = 0.0025
    minimum_silhouette_gap_pixels: float = 2.0
    minimum_boundary_clearance_pixels: float = 2.0
    maximum_surface_radius_relative_error: float = 0.05
    world_radius: float = 0.21
    foreground_threshold: float = 0.04
    foreground_temperature: float = 0.01
    minimum_mass: float = 4.0
    measurement_position_variance: float = 6.4e-5
    temporal_history_size: int = 16
    temporal_min_samples: int = 16
    temporal_min_dt: float = 1.0e-3
    temporal_velocity_variance_floor: float = 1.0e-6
    temporal_velocity_variance_ceiling: float | None = 1.0e-2
    fit_conditioning_limit: float = 100.0
    observation_mode: str = "legacy"
    max_objects: int = SET_MAX_OBJECTS
    set_feature_dim: int = SET_FEATURE_DIM
    set_log_variance_residual_limit: float = SET_MAX_LOG_VARIANCE_RESIDUAL

    def __post_init__(self) -> None:
        if self.observation_mode not in {"legacy", "set"}:
            raise ValueError("RGB-D observation_mode must be 'legacy' or 'set'")
        if (
            isinstance(self.max_objects, bool)
            or not isinstance(self.max_objects, int)
            or self.max_objects <= 0
        ):
            raise ValueError("RGB-D max_objects must be a positive integer")
        if isinstance(self.proposal_count, bool) or not isinstance(self.proposal_count, int):
            if self.observation_mode == "legacy":
                raise ValueError("RGB-D proposal_count must be integer one or two")
            raise ValueError("RGB-D proposal_count must be an integer")
        if self.observation_mode == "legacy" and self.proposal_count not in {1, 2}:
            raise ValueError("RGB-D proposal_count must be integer one or two in legacy mode")
        if self.observation_mode == "set" and self.proposal_count != SET_PROPOSAL_COUNT:
            raise ValueError(f"set RGB-D observation requires proposal_count={SET_PROPOSAL_COUNT}")
        if self.observation_mode == "set" and self.max_objects != SET_MAX_OBJECTS:
            raise ValueError(f"set RGB-D observation requires max_objects={SET_MAX_OBJECTS}")
        if self.set_feature_dim not in {SET_FEATURE_DIM, 2 * SET_FEATURE_DIM}:
            raise ValueError("RGB-D set_feature_dim must be 32 or 64")
        if self.observation_mode == "legacy" and self.set_feature_dim != SET_FEATURE_DIM:
            raise ValueError("legacy RGB-D requires set_feature_dim=32")
        if (
            isinstance(self.set_log_variance_residual_limit, bool)
            or not isinstance(self.set_log_variance_residual_limit, (int, float))
            or not math.isfinite(float(self.set_log_variance_residual_limit))
            or not 0.0
            < float(self.set_log_variance_residual_limit)
            <= SET_MAX_CONFIGURABLE_LOG_VARIANCE_RESIDUAL
        ):
            raise ValueError(
                "RGB-D set_log_variance_residual_limit must be finite and lie in (0,32]"
            )
        if (
            self.observation_mode == "legacy"
            and self.set_log_variance_residual_limit != SET_MAX_LOG_VARIANCE_RESIDUAL
        ):
            raise ValueError("legacy RGB-D requires set_log_variance_residual_limit=4")
        if (
            isinstance(self.appearance_dim, bool)
            or not isinstance(self.appearance_dim, int)
            or self.appearance_dim <= 0
        ):
            raise ValueError("RGB-D appearance_dim must be a positive integer")
        if (
            self.observation_mode == "legacy"
            and self.proposal_count == 2
            and (self.appearance_dim != 3)
        ):
            raise ValueError("two-object RGB-D requires appearance_dim exactly three")
        if self.observation_mode == "set" and self.appearance_dim != SET_APPEARANCE_DIM:
            raise ValueError(f"set RGB-D observation requires appearance_dim={SET_APPEARANCE_DIM}")
        positive = {
            "world_radius": self.world_radius,
            "foreground_threshold": self.foreground_threshold,
            "foreground_temperature": self.foreground_temperature,
            "minimum_mass": self.minimum_mass,
            "chromatic_temperature": self.chromatic_temperature,
            "minimum_chromatic_eigengap": self.minimum_chromatic_eigengap,
            "spatial_temperature_pixels": self.spatial_temperature_pixels,
            "chromatic_centre_blend": self.chromatic_centre_blend,
            "minimum_silhouette_gap_pixels": self.minimum_silhouette_gap_pixels,
            "minimum_boundary_clearance_pixels": self.minimum_boundary_clearance_pixels,
            "maximum_surface_radius_relative_error": (self.maximum_surface_radius_relative_error),
            "measurement_position_variance": self.measurement_position_variance,
            "temporal_min_dt": self.temporal_min_dt,
            "temporal_velocity_variance_floor": self.temporal_velocity_variance_floor,
        }
        for name, value in positive.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"RGB-D {name} must be finite and positive")
        if self.maximum_surface_radius_relative_error > 1.0:
            raise ValueError(
                "RGB-D maximum_surface_radius_relative_error must be no greater than one"
            )
        if self.chromatic_centre_blend > 1.0:
            raise ValueError("RGB-D chromatic_centre_blend must be no greater than one")
        for name, value in (
            ("temporal_history_size", self.temporal_history_size),
            ("temporal_min_samples", self.temporal_min_samples),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"RGB-D {name} must be an integer of at least two")
        if self.temporal_history_size != 16:
            raise ValueError("the first RGB-D bridge requires exactly 16 history samples")
        if self.observation_mode == "legacy" and self.temporal_min_samples != (
            self.temporal_history_size
        ):
            raise ValueError("legacy RGB-D temporal_min_samples must equal temporal_history_size")
        if self.observation_mode == "set" and self.temporal_min_samples != 3:
            raise ValueError("set RGB-D temporal_min_samples must equal three")
        ceiling = self.temporal_velocity_variance_ceiling
        if ceiling is not None and (
            isinstance(ceiling, bool)
            or not isinstance(ceiling, (int, float))
            or not math.isfinite(float(ceiling))
            or ceiling < self.temporal_velocity_variance_floor
        ):
            raise ValueError(
                "RGB-D temporal velocity variance ceiling must be finite and no "
                "smaller than its floor"
            )
        if (
            isinstance(self.fit_conditioning_limit, bool)
            or not isinstance(self.fit_conditioning_limit, (int, float))
            or not math.isfinite(float(self.fit_conditioning_limit))
            or self.fit_conditioning_limit <= 1.0
        ):
            raise ValueError("RGB-D fit_conditioning_limit must be finite and greater than one")


def _composite_payload(packet: ObservationPacket) -> tuple[Tensor, Tensor]:
    if not isinstance(packet.payload, Mapping):
        raise TypeError("RGB-D packet payload must be a mapping")
    if set(packet.payload) != {"rgb", "depth"}:
        raise ValueError("RGB-D packet payload must contain exactly 'rgb' and 'depth'")
    rgb = packet.payload["rgb"]
    depth = packet.payload["depth"]
    if not isinstance(rgb, Tensor) or not isinstance(depth, Tensor):
        raise TypeError("RGB-D payload values must be torch tensors")
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("RGB-D rgb payload must have shape [B,3,H,W]")
    if depth.shape != (rgb.shape[0], 1, *rgb.shape[-2:]):
        raise ValueError("RGB-D depth payload must have shape [B,1,H,W]")
    if rgb.shape[0] <= 0 or min(rgb.shape[-2:]) < 2:
        raise ValueError("RGB-D packet batch must be nonempty and image dimensions at least two")
    if rgb.dtype not in {torch.float32, torch.float64}:
        raise TypeError("RGB-D payload supports only float32 and float64")
    if depth.dtype != rgb.dtype or depth.device != rgb.device:
        raise ValueError("RGB-D rgb and depth must share dtype and device")
    if not torch.isfinite(rgb).all():
        raise ValueError("RGB-D rgb payload contains NaN or Inf")
    if torch.any((rgb < 0.0) | (rgb > 1.0)):
        raise ValueError("RGB-D rgb values must lie in [0,1]")
    return rgb, depth


def _batched_calibration(
    packet: ObservationPacket,
    *,
    batch: int,
    reference: Tensor,
) -> tuple[Tensor, Tensor]:
    if set(packet.calibration) != {"world_from_camera", "intrinsics"}:
        raise ValueError(
            "RGB-D packet calibration must contain exactly world_from_camera and intrinsics"
        )
    world_from_camera = packet.calibration["world_from_camera"]
    intrinsics = packet.calibration["intrinsics"]
    if not isinstance(world_from_camera, Tensor) or not isinstance(intrinsics, Tensor):
        raise TypeError("RGB-D calibration values must be torch tensors")
    if world_from_camera.shape != (batch, 4, 4):
        raise ValueError("RGB-D world_from_camera must have shape [B,4,4]")
    if intrinsics.shape != (batch, 3, 3):
        raise ValueError("RGB-D intrinsics must have shape [B,3,3]")
    for name, value in (
        ("world_from_camera", world_from_camera),
        ("intrinsics", intrinsics),
    ):
        if not value.is_floating_point():
            raise TypeError(f"RGB-D {name} must be floating point")
        if value.dtype != reference.dtype or value.device != reference.device:
            raise ValueError(f"RGB-D {name} must share payload dtype and device")
        if not torch.isfinite(value).all():
            raise ValueError(f"RGB-D {name} contains NaN or Inf")
    if torch.any(intrinsics[:, 0, 0] < 1.0e-3) or torch.any(intrinsics[:, 1, 1] < 1.0e-3):
        raise ValueError("RGB-D intrinsics require positive finite focal lengths")
    tolerance = max(2.0e-5, 64.0 * torch.finfo(reference.dtype).eps)
    expected_intrinsics_row = reference.new_tensor([0.0, 0.0, 1.0])
    expected_transform_row = reference.new_tensor([0.0, 0.0, 0.0, 1.0])
    rotation = world_from_camera[:, :3, :3]
    identity = torch.eye(3, dtype=reference.dtype, device=reference.device).expand(batch, -1, -1)
    canonical = (intrinsics[:, 0, 1].abs() <= tolerance) & (intrinsics[:, 1, 0].abs() <= tolerance)
    canonical &= (intrinsics[:, 2] - expected_intrinsics_row).abs().amax(dim=-1) <= tolerance
    canonical &= (world_from_camera[:, 3] - expected_transform_row).abs().amax(dim=-1) <= tolerance
    canonical &= (rotation.transpose(-1, -2) @ rotation - identity).abs().amax(
        dim=(-2, -1)
    ) <= tolerance
    canonical &= torch.linalg.det(rotation) > 0.0
    if not bool(canonical.all()):
        raise ValueError("RGB-D calibration must be canonical and rigid")
    return world_from_camera, intrinsics


def _explicit_image_size(packet: ObservationPacket, rgb: Tensor) -> tuple[int, int]:
    image_size = packet.metadata.get("image_size")
    if not isinstance(image_size, (tuple, list)) or len(image_size) != 2:
        raise ValueError("RGB-D packet metadata.image_size must be explicit [H,W]")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in image_size):
        raise TypeError("RGB-D metadata.image_size values must be integers")
    resolved = (int(image_size[0]), int(image_size[1]))
    if resolved != tuple(rgb.shape[-2:]):
        raise ValueError("RGB-D metadata.image_size does not match the payload")
    return resolved


def _set_geometry_from_full_masks(
    image: Tensor,
    foreground_probability: Tensor,
    effective_masks: Tensor,
    *,
    minimum_mass: float,
) -> SoftDiscGeometryOutput:
    """Derive set geometry from the exact supervised proposal masks.

    The legacy image-moment primitive consumes independent sigmoid slot
    logits.  Set mode instead owns one categorical background-plus-proposal
    mask, so using those logits through the legacy sigmoid path would make the
    mask loss supervise a different geometry from the one emitted at runtime.
    This function keeps the same analytic moments while making the full-mask
    softmax the single mask owner.
    """

    batch, proposals, height, width = effective_masks.shape
    if image.shape != (batch, 3, height, width):
        raise ValueError("set mask geometry must match its RGB image")
    if foreground_probability.shape != (batch, 1, height, width):
        raise ValueError("set foreground probability must have shape [B,1,H,W]")
    if effective_masks.dtype != image.dtype or effective_masks.device != image.device:
        raise ValueError("set masks must share RGB dtype and device")
    if not bool(torch.isfinite(effective_masks).all()):
        raise ValueError("set masks must be finite")
    if bool(torch.any((effective_masks < 0.0) | (effective_masks > 1.0))):
        raise ValueError("set masks must lie in [0,1]")

    epsilon = max(float(torch.finfo(image.dtype).eps), 1.0e-8)
    y_pixels, x_pixels = torch.meshgrid(
        torch.arange(height, dtype=image.dtype, device=image.device),
        torch.arange(width, dtype=image.dtype, device=image.device),
        indexing="ij",
    )
    y_normalised, x_normalised = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, dtype=image.dtype, device=image.device),
        torch.linspace(-1.0, 1.0, width, dtype=image.dtype, device=image.device),
        indexing="ij",
    )
    mass = effective_masks.sum(dim=(-2, -1))
    safe_mass = mass.clamp_min(epsilon)
    centre_x_normalised = (effective_masks * x_normalised).sum(dim=(-2, -1)) / safe_mass
    centre_y_normalised = (effective_masks * y_normalised).sum(dim=(-2, -1)) / safe_mass
    centres = torch.stack((centre_x_normalised, centre_y_normalised), dim=-1)
    centre_x_pixels = 0.5 * (centre_x_normalised + 1.0) * (width - 1)
    centre_y_pixels = 0.5 * (centre_y_normalised + 1.0) * (height - 1)
    delta_x = x_pixels - centre_x_pixels[..., None, None]
    delta_y = y_pixels - centre_y_pixels[..., None, None]
    variance_x = (effective_masks * delta_x.square()).sum(dim=(-2, -1)) / safe_mass
    variance_y = (effective_masks * delta_y.square()).sum(dim=(-2, -1)) / safe_mass
    covariance_xy = (effective_masks * delta_x * delta_y).sum(dim=(-2, -1)) / safe_mass
    radial_second_moment = variance_x + variance_y
    radius_pixels = (2.0 * radial_second_moment).clamp_min(epsilon).sqrt()
    mass_confidence = 1.0 - torch.exp(-mass / image.new_tensor(minimum_mass))
    covariance_determinant = (variance_x * variance_y - covariance_xy.square()).clamp_min(0.0)
    circularity = (4.0 * covariance_determinant / (radial_second_moment.square() + epsilon)).clamp(
        0.0, 1.0
    )
    edge_clearance = (
        torch.stack(
            (
                centre_x_pixels,
                width - 1 - centre_x_pixels,
                centre_y_pixels,
                height - 1 - centre_y_pixels,
            ),
            dim=-1,
        ).amin(dim=-1)
        - radius_pixels
    )
    boundary_confidence = torch.sigmoid(edge_clearance)
    confidence = (mass_confidence * circularity * boundary_confidence).clamp(0.0, 1.0)
    valid_mask = (mass.detach() >= minimum_mass) & torch.isfinite(centres.detach()).all(dim=-1)
    return SoftDiscGeometryOutput(
        centres=centres,
        radius_pixels=radius_pixels,
        confidence=confidence,
        valid_mask=valid_mask,
        mass=mass,
        foreground_probability=foreground_probability,
        effective_masks=effective_masks,
    )


@register_observation_module("rgbd")
class RGBDObservationModule(ObservationModule):
    """Metric RGB-D observation with legacy and explicit learned-set modes."""

    modality_name = "rgbd"
    modality_index = 2
    requires_post_birth_temporal_history = True

    def __init__(self, config: RGBDObservationConfig | None = None) -> None:
        super().__init__()
        self.config = config or RGBDObservationConfig()
        self.measurement = RGBDSphereCentreMeasurementModule(
            foreground_threshold=self.config.foreground_threshold,
            foreground_temperature=self.config.foreground_temperature,
            minimum_mass=self.config.minimum_mass,
        )
        self.set_proposer = (
            RGBDSetProposer(
                proposal_count=self.config.proposal_count,
                appearance_dim=self.config.appearance_dim,
                feature_dim=self.config.set_feature_dim,
                log_variance_residual_limit=(self.config.set_log_variance_residual_limit),
            )
            if self.config.observation_mode == "set"
            else None
        )

    def validate_packet(self, packet: ObservationPacket) -> None:
        if packet.modality != self.modality_name:
            raise ValueError("RGB-D module accepts only modality='rgbd'")
        rgb, _ = _composite_payload(packet)
        _batched_calibration(packet, batch=rgb.shape[0], reference=rgb)
        _explicit_image_size(packet, rgb)

    @staticmethod
    def _set_appearance(
        image: Tensor,
        masks: Tensor,
        appearance_residual: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        """Return an eight-value observable colour descriptor plus residual."""

        epsilon = torch.finfo(image.dtype).eps
        mass = masks.sum(dim=(-2, -1)).clamp_min(epsilon)
        mean_rgb = torch.einsum("bphw,bchw->bpc", masks, image) / mass.unsqueeze(-1)
        second_rgb = torch.einsum("bphw,bchw->bpc", masks, image.square()) / mass.unsqueeze(-1)
        std_rgb = (second_rgb - mean_rgb.square() + epsilon).clamp_min(epsilon).sqrt()
        intensity = image.mean(dim=1)
        mean_intensity = torch.einsum("bphw,bhw->bp", masks, intensity) / mass
        second_intensity = torch.einsum("bphw,bhw->bp", masks, intensity.square()) / mass
        std_intensity = (
            (second_intensity - mean_intensity.square() + epsilon).clamp_min(epsilon).sqrt()
        )
        observable = torch.cat(
            (
                mean_rgb,
                std_rgb,
                mean_intensity.unsqueeze(-1),
                std_intensity.unsqueeze(-1),
            ),
            dim=-1,
        )
        appearance = F.normalize(observable + appearance_residual, dim=-1, eps=epsilon)
        return torch.where(valid_mask.unsqueeze(-1), appearance, torch.zeros_like(appearance))

    def _measure_set(
        self,
        packet: ObservationPacket,
        rgb: Tensor,
        depth: Tensor,
        world_from_camera: Tensor,
        intrinsics: Tensor,
    ) -> MeasurementSet:
        if self.set_proposer is None:
            raise RuntimeError("set RGB-D mode requires its learned proposer")
        foreground_temperature = max(
            self.config.foreground_temperature,
            SET_FOREGROUND_TEMPERATURE_FLOOR,
        )
        analytic_geometry = soft_disc_geometry_from_rgb(
            rgb,
            foreground_threshold=self.config.foreground_threshold,
            foreground_temperature=foreground_temperature,
            minimum_mass=self.config.minimum_mass,
        )
        valid_depth = torch.isfinite(depth) & (depth > 0.0) & (depth <= MAXIMUM_METRIC_DISTANCE_M)
        safe_depth = torch.where(valid_depth, depth, torch.ones_like(depth))
        log_depth = torch.where(valid_depth, safe_depth.log(), torch.zeros_like(depth))
        height, width = rgb.shape[-2:]
        y_axis = torch.linspace(-1.0, 1.0, height, dtype=rgb.dtype, device=rgb.device)
        x_axis = torch.linspace(-1.0, 1.0, width, dtype=rgb.dtype, device=rgb.device)
        yy, xx = torch.meshgrid(y_axis, x_axis, indexing="ij")
        image_coordinates = (
            torch.stack((xx, yy), dim=0)
            .unsqueeze(0)
            .expand(
                rgb.shape[0],
                -1,
                -1,
                -1,
            )
        )
        proposal = self.set_proposer(
            rgb,
            log_depth,
            valid_depth,
            analytic_geometry.foreground_probability,
            image_coordinates,
        )
        full_mask_probability = proposal.full_mask_probability
        geometry = _set_geometry_from_full_masks(
            rgb,
            analytic_geometry.foreground_probability,
            full_mask_probability[:, 1:],
            minimum_mass=self.config.minimum_mass,
        )
        centre_metric = metric_sphere_centres_from_surface_depth(
            geometry.centres,
            depth,
            self.config.world_radius,
            world_from_camera,
            intrinsics,
        )
        expected_radius = rgb.new_full(
            geometry.mass.shape,
            self.config.world_radius,
        )
        (
            fitted_world_position,
            _fitted_camera_position,
            fitted_radius,
            fit_condition_number,
            surface_fit_valid,
        ) = fit_visible_sphere_surfaces(
            depth,
            geometry.effective_masks,
            expected_radius,
            world_from_camera,
            intrinsics,
            conditioning_limit=self.config.fit_conditioning_limit,
        )
        # Admissible protocol scenes use the multi-pixel fixed-radius surface
        # fit.  The centre-depth fallback retains the public module's legacy
        # behavior for synthetic/non-spherical callers whose surface system is
        # unidentifiable; evaluation records which branch supplied each row.
        use_surface_fit = surface_fit_valid
        valid_mask = geometry.valid_mask & (surface_fit_valid | centre_metric.valid_mask)
        value_gate = valid_mask.unsqueeze(-1)
        selected_world_position = torch.where(
            use_surface_fit.unsqueeze(-1),
            fitted_world_position,
            centre_metric.world_position,
        )
        world_position = torch.where(
            value_gate,
            selected_world_position,
            torch.zeros_like(selected_world_position),
        )
        appearance = self._set_appearance(
            rgb,
            geometry.effective_masks,
            proposal.appearance_residual,
            valid_mask,
        )
        epsilon = torch.finfo(rgb.dtype).eps
        analytic_component = geometry.valid_mask.detach()
        base_existence_probability = torch.where(
            analytic_component,
            rgb.new_full(analytic_component.shape, 0.99),
            rgb.new_full(analytic_component.shape, 0.01),
        )
        base_existence_logits = torch.logit(base_existence_probability)
        existence_logits = (base_existence_logits + proposal.existence_residual).clamp(-12.0, 12.0)
        proposals = self.config.proposal_count
        batch = rgb.shape[0]
        base_log_variance = rgb.new_full(
            (batch, proposals, 3),
            math.log(self.config.measurement_position_variance),
        )
        log_variance = base_log_variance + proposal.log_variance_residual
        valid_depth_float = valid_depth[:, 0].to(rgb.dtype)
        surface_weight = geometry.effective_masks * valid_depth_float[:, None]
        surface_support = surface_weight.sum(dim=(-2, -1))
        mean_surface_depth = torch.einsum(
            "bphw,bhw->bp",
            surface_weight,
            torch.where(valid_depth[:, 0], depth[:, 0], torch.zeros_like(depth[:, 0])),
        ) / surface_support.clamp_min(epsilon)
        selected_surface_depth = torch.where(
            use_surface_fit,
            mean_surface_depth,
            centre_metric.surface_depth,
        )
        depth_support = torch.where(
            use_surface_fit,
            (surface_support / geometry.mass.clamp_min(epsilon)).clamp(0.0, 1.0),
            centre_metric.depth_support,
        )
        confidence = (
            geometry.confidence
            * depth_support
            * existence_logits.sigmoid()
            * float(packet.confidence)
        ).clamp(0.0, 1.0)
        confidence = torch.where(valid_mask, confidence, torch.zeros_like(confidence))
        valid_axes = valid_mask.unsqueeze(-1).expand(batch, proposals, 3)
        result = MeasurementSet(
            modality=self.modality_name,
            sensor_id=packet.sensor_id,
            timestamp=rgb.new_full((batch,), packet.timestamp),
            values=world_position,
            log_variance=log_variance,
            existence_logits=existence_logits,
            measurement_mask=valid_mask,
            appearance=appearance,
            class_logits=None,
            frame_id=packet.frame_id,
            supported_state_fields=("position",),
            auxiliary={
                "world_position": world_position,
                "world_log_variance": log_variance,
                "world_position_log_variance": log_variance,
                "world_position_independent_axis_mask": valid_axes,
                "world_radius": rgb.new_full(
                    (batch, proposals, 1),
                    self.config.world_radius,
                ),
                "position_confidence": confidence,
                "visibility_logit": existence_logits,
                "metric_confidence": confidence,
                "metric_surface_depth": torch.where(
                    valid_mask,
                    selected_surface_depth,
                    torch.zeros_like(selected_surface_depth),
                ),
                "surface_fit_valid": surface_fit_valid,
                "surface_fit_condition_number": fit_condition_number,
                "surface_fit_radius": fitted_radius,
                "surface_fit_radius_relative_error": (
                    (fitted_radius - expected_radius).abs() / expected_radius.clamp_min(epsilon)
                ),
                "image_centres": geometry.centres,
                "image_radius_pixels": geometry.radius_pixels,
                "foreground_mass": geometry.mass,
                "set_anchor_points": proposal.anchor_points.unsqueeze(0).expand(batch, -1, -1),
                "set_mask_residual_rms": proposal.mask_residual.square().mean(dim=(-2, -1)).sqrt(),
                "set_full_mask_logits": proposal.full_mask_logits,
                "set_background_mask_logits": proposal.background_mask_logits,
                "set_existence_residual": proposal.existence_residual,
                "set_appearance_residual": proposal.appearance_residual,
                "set_log_variance_residual": proposal.log_variance_residual,
            },
        )
        result.validate()
        return result

    def _measure(self, packet: ObservationPacket) -> MeasurementSet:
        self.validate_packet(packet)
        rgb, depth = _composite_payload(packet)
        world_from_camera, intrinsics = _batched_calibration(
            packet,
            batch=rgb.shape[0],
            reference=rgb,
        )
        if self.config.observation_mode == "set":
            return self._measure_set(
                packet,
                rgb,
                depth,
                world_from_camera,
                intrinsics,
            )
        if self.config.proposal_count == 1:
            measured = self.measurement(
                rgb,
                depth,
                self.config.world_radius,
                world_from_camera,
                intrinsics,
            )
            measured_appearance = None
            measurement_diagnostics: dict[str, Tensor] = {}
        else:
            measured = two_disc_geometry_from_rgbd(
                rgb,
                depth,
                self.config.world_radius,
                world_from_camera,
                intrinsics,
                foreground_threshold=self.config.foreground_threshold,
                foreground_temperature=self.config.foreground_temperature,
                minimum_mass=self.config.minimum_mass,
                chromatic_temperature=self.config.chromatic_temperature,
                minimum_chromatic_eigengap=self.config.minimum_chromatic_eigengap,
                spatial_temperature_pixels=self.config.spatial_temperature_pixels,
                chromatic_centre_blend=self.config.chromatic_centre_blend,
                minimum_silhouette_gap_pixels=self.config.minimum_silhouette_gap_pixels,
                minimum_boundary_clearance_pixels=(self.config.minimum_boundary_clearance_pixels),
                maximum_surface_radius_relative_error=(
                    self.config.maximum_surface_radius_relative_error
                ),
                surface_fit_conditioning_limit=self.config.fit_conditioning_limit,
            )
            measured_appearance = rgb.new_zeros(
                (*measured.appearance.shape[:2], self.config.appearance_dim)
            )
            measured_appearance[..., :3] = measured.appearance
            measurement_diagnostics = {
                "chromatic_eigengap": measured.chromatic_eigengap,
                "pair_valid_mask": measured.pair_valid_mask,
                "image_centres": measured.centres,
                "provisional_image_centres": measured.provisional_centres,
                "image_radius_pixels": measured.radius_pixels,
                "surface_fit_condition_number": measured.surface_fit_condition_number,
                "surface_fit_radius": measured.surface_fit_radius,
                "surface_fit_radius_relative_error": (measured.surface_fit_radius_relative_error),
                "silhouette_gap_pixels": measured.silhouette_gap_pixels,
                "boundary_clearance_pixels": measured.boundary_clearance_pixels,
                "chromatic_world_position": measured.chromatic_world_position,
            }
        batch, proposals = measured.valid_mask.shape
        if proposals != self.config.proposal_count:
            raise RuntimeError("RGB-D measurement proposal count disagrees with configuration")
        log_variance = rgb.new_full(
            (batch, proposals, 3),
            math.log(self.config.measurement_position_variance),
        )
        epsilon = torch.finfo(rgb.dtype).eps
        existence_probability = torch.where(
            measured.valid_mask,
            rgb.new_full((batch, proposals), 1.0 - epsilon),
            rgb.new_full((batch, proposals), epsilon),
        )
        existence_logits = torch.logit(existence_probability)
        position_confidence = (measured.confidence * float(packet.confidence)).clamp(0.0, 1.0)
        valid_axes = measured.valid_mask.unsqueeze(-1).expand(batch, proposals, 3)
        result = MeasurementSet(
            modality=self.modality_name,
            sensor_id=packet.sensor_id,
            timestamp=rgb.new_full((batch,), packet.timestamp),
            values=measured.world_position,
            log_variance=log_variance,
            existence_logits=existence_logits,
            measurement_mask=measured.valid_mask,
            appearance=measured_appearance,
            class_logits=None,
            frame_id=packet.frame_id,
            supported_state_fields=("position",),
            auxiliary={
                "world_position": measured.world_position,
                "world_log_variance": log_variance,
                "world_position_log_variance": log_variance,
                "world_position_independent_axis_mask": valid_axes,
                "world_radius": rgb.new_full(
                    (batch, proposals, 1),
                    self.config.world_radius,
                ),
                "position_confidence": position_confidence,
                "visibility_logit": existence_logits,
                "metric_confidence": measured.confidence,
                "metric_surface_depth": measured.surface_depth,
                **measurement_diagnostics,
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
        if len(packets) != 1:
            raise ValueError("RGB-D expects one composite packet per timestamp")
        return self._measure(packets[0])

    def encode_measurements(
        self,
        packets: Sequence[ObservationPacket],
        prior: WorldBelief,
        predicted: PredictedMeasurements,
        cache: ModalityCache | None,
    ) -> tuple[MeasurementSet, ModalityCache]:
        del prior, predicted
        if len(packets) != 1:
            raise ValueError("RGB-D expects one composite packet per timestamp")
        return self._measure(packets[0]), cache or ModalityCache()

    def project(
        self,
        belief: WorldBelief,
        sensor_context: SensorContext,
    ) -> PredictedMeasurements:
        if self.config.observation_mode == "set":
            if belief.objects.max_objects != self.config.max_objects:
                raise ValueError("set RGB-D belief object count must equal max_objects")
        elif belief.objects.max_objects != self.config.proposal_count:
            raise ValueError("RGB-D belief object count must equal proposal_count")
        objects = belief.objects
        position_slice = fast_packing_map(objects)["position"]
        batch, objects_count = objects.active.shape
        belief_indices = (
            torch.arange(objects_count, dtype=torch.int64, device=belief.device)
            .unsqueeze(0)
            .expand(batch, -1)
        )
        result = PredictedMeasurements(
            modality=self.modality_name,
            sensor_id=sensor_context.sensor_id,
            timestamp=belief.timestamp,
            values=objects.position,
            log_variance=objects.fast_log_variance[..., position_slice],
            object_ids=objects.object_id,
            belief_indices=belief_indices,
            valid_mask=objects.active,
            visibility=objects.visibility_logit.sigmoid(),
            rois=None,
            appearance=(
                objects.appearance
                if self.config.observation_mode == "set" or self.config.proposal_count == 2
                else None
            ),
            auxiliary={"world_position": objects.position},
        )
        result.validate()
        return result

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

    def _history(
        self,
        posterior: WorldBelief,
        history: ModalityHistory | None,
    ) -> RGBDTemporalPositionHistory:
        if history is None:
            return RGBDTemporalPositionHistory.empty(
                object_ids=posterior.objects.object_id,
                active_mask=posterior.objects.active,
                history_size=self.config.temporal_history_size,
                dtype=posterior.dtype,
            )
        if not isinstance(history, RGBDTemporalPositionHistory):
            raise TypeError("RGB-D sensor history has an incompatible modality type")
        return history

    @staticmethod
    def _associated_positions(
        posterior: WorldBelief,
        measured: MeasurementSet,
        association: AssociationResult,
    ) -> tuple[Tensor, Tensor]:
        positions = posterior.objects.position.new_zeros((*posterior.objects.active.shape, 3))
        valid = torch.zeros_like(posterior.objects.active)
        pair_batch, pair_index = torch.nonzero(association.pair_mask, as_tuple=True)
        if pair_batch.numel() == 0:
            return positions, valid
        belief_index = association.belief_indices[pair_batch, pair_index]
        measurement_index = association.measurement_indices[pair_batch, pair_index]
        accepted = (
            ~association.ambiguous[pair_batch, pair_index]
            & measured.measurement_mask[pair_batch, measurement_index]
            & posterior.objects.active[pair_batch, belief_index]
        )
        accepted_batch = pair_batch[accepted]
        accepted_belief = belief_index[accepted]
        accepted_measurement = measurement_index[accepted]
        if accepted_batch.numel():
            raw = measured.auxiliary["world_position"]
            positions[accepted_batch, accepted_belief] = raw[
                accepted_batch,
                accepted_measurement,
            ]
            valid[accepted_batch, accepted_belief] = True
        return positions, valid

    def _velocity_variance(self, fit: FreeMotionFitResult, valid: Tensor) -> Tensor:
        """Return a bounded OLS-residual evidence scale, not a posterior claim."""

        identity = torch.eye(
            2,
            dtype=fit.normal_matrix.dtype,
            device=fit.normal_matrix.device,
        )
        safe_normal = torch.where(
            fit.valid[..., None, None],
            fit.normal_matrix,
            identity,
        )
        inverse_normal = torch.linalg.inv(safe_normal)
        if self.config.observation_mode == "legacy":
            # Preserve the accepted exact sixteen-sample arithmetic.
            sample_count = self.config.temporal_history_size
            degrees_of_freedom = sample_count - 2
            residual_covariance = fit.residual_covariance * (sample_count / degrees_of_freedom)
            coefficient_scale = inverse_normal[..., 1, 1] / sample_count
            variance = residual_covariance.diagonal(
                dim1=-2,
                dim2=-1,
            ) * coefficient_scale.unsqueeze(-1)
            variance = variance.clamp_min(self.config.temporal_velocity_variance_floor)
        else:
            sample_count = fit.support_count.to(dtype=fit.normal_matrix.dtype).clamp_min(3.0)
            degrees_of_freedom = (sample_count - 2.0).clamp_min(1.0)
            residual_covariance = fit.residual_covariance * (
                sample_count / degrees_of_freedom
            ).unsqueeze(-1).unsqueeze(-1)
            coefficient_scale = inverse_normal[..., 1, 1] / sample_count
            maturity_inflation = (
                fit.normal_matrix.new_tensor(float(self.config.temporal_history_size))
                / sample_count
            ).square()
            # The protocol requires the *bounded base evidence scale* to be
            # inflated by (16 / valid_samples)^2.  Flooring only after the
            # multiplication would leave noiseless axes at the same floor for
            # immature and mature histories, silently defeating that rule.
            base_variance = (
                residual_covariance.diagonal(dim1=-2, dim2=-1) * coefficient_scale.unsqueeze(-1)
            ).clamp_min(self.config.temporal_velocity_variance_floor)
            variance = base_variance * maturity_inflation.unsqueeze(-1)
        if self.config.temporal_velocity_variance_ceiling is not None:
            variance = variance.clamp_max(self.config.temporal_velocity_variance_ceiling)
        return torch.where(valid.unsqueeze(-1), variance, torch.ones_like(variance))

    def update_temporal_history(
        self,
        *,
        posterior: WorldBelief,
        measured: MeasurementSet,
        association: AssociationResult,
        history: ModalityHistory | None,
    ) -> tuple[DirectVelocityEvidence | None, ModalityHistory | None]:
        """Append raw associated metric positions and emit uniform-fit velocity."""

        resolved = self._history(posterior, history)
        positions, valid = self._associated_positions(posterior, measured, association)
        reset_mask = torch.zeros_like(posterior.objects.active)
        if self.config.observation_mode == "set":
            prior_interval_collision = measured.auxiliary.get("prior_interval_collision_mask")
            if prior_interval_collision is not None:
                if (
                    prior_interval_collision.shape != reset_mask.shape
                    or prior_interval_collision.dtype is not torch.bool
                    or prior_interval_collision.device != reset_mask.device
                ):
                    raise ValueError("RGB-D prior_interval_collision_mask must be boolean [B,N]")
                reset_mask |= prior_interval_collision
            prior_interval_known_action = measured.auxiliary.get("prior_interval_known_action_mask")
            if prior_interval_known_action is not None:
                if (
                    prior_interval_known_action.shape != reset_mask.shape
                    or prior_interval_known_action.dtype is not torch.bool
                    or prior_interval_known_action.device != reset_mask.device
                ):
                    raise ValueError("RGB-D prior_interval_known_action_mask must be boolean [B,N]")
                reset_mask |= prior_interval_known_action
            reset_mask |= posterior.objects.mode == int(MotionMode.COLLISION)
        resolved = resolved.append(
            object_ids=posterior.objects.object_id,
            active_mask=posterior.objects.active,
            append_mask=posterior.objects.active,
            timestamp=measured.timestamp,
            positions=positions,
            valid_mask=valid,
            minimum_dt=self.config.temporal_min_dt,
            reset_mask=reset_mask,
        )
        fit, fit_valid = resolved.fit(
            gravity=posterior.gravity,
            drag=posterior.objects.drag,
            minimum_support=self.config.temporal_min_samples,
            minimum_dt=self.config.temporal_min_dt,
            conditioning_limit=self.config.fit_conditioning_limit,
            require_complete_window=self.config.observation_mode == "legacy",
        )
        fit_valid = fit_valid & posterior.objects.active
        if not bool(fit_valid.any()):
            return None, resolved
        variance = self._velocity_variance(fit, fit_valid)
        evidence = DirectVelocityEvidence(
            velocity=torch.where(
                fit_valid.unsqueeze(-1),
                fit.velocity,
                torch.zeros_like(fit.velocity),
            ),
            log_variance=variance.log(),
            valid_mask=fit_valid,
            confidence=fit_valid.to(posterior.dtype),
            axis_valid_mask=fit_valid.unsqueeze(-1).expand_as(fit.velocity),
            # Consecutive estimates reuse almost the entire sliding history.
            # They are not independent observations of the prior velocity and
            # therefore must not repeatedly contract below their own variance.
            correlated_with_prior=self.config.observation_mode == "set",
        )
        evidence.validate()
        return evidence, resolved

    def validate_temporal_history_packet(
        self,
        *,
        posterior: WorldBelief,
        packet: ObservationPacket,
        history: ModalityHistory | None,
    ) -> None:
        if history is None:
            return
        if not isinstance(history, RGBDTemporalPositionHistory):
            raise TypeError("RGB-D sensor history has an incompatible modality type")
        timestamp = posterior.timestamp.new_full(
            posterior.timestamp.shape,
            packet.timestamp,
        )
        history.validate_next_timestamp(
            object_ids=posterior.objects.object_id,
            active_mask=posterior.objects.active,
            timestamp=timestamp,
            minimum_dt=self.config.temporal_min_dt,
        )

    def update_temporal_history_after_births(
        self,
        *,
        posterior: WorldBelief,
        measured: MeasurementSet,
        birth_assignments: BirthAssignments,
        history: ModalityHistory | None,
    ) -> ModalityHistory | None:
        """Seed newly allocated persistent IDs from their raw birth measurement."""

        birth_assignments.validate()
        resolved = self._history(posterior, history)
        append_mask = torch.zeros_like(posterior.objects.active)
        valid = torch.zeros_like(posterior.objects.active)
        positions = torch.zeros_like(posterior.objects.position)
        batch_index = birth_assignments.batch_indices
        belief_index = birth_assignments.belief_indices
        measurement_index = birth_assignments.measurement_indices
        if batch_index.numel():
            assigned_ids = birth_assignments.object_ids
            if not torch.equal(
                posterior.objects.object_id[batch_index, belief_index],
                assigned_ids,
            ):
                raise ValueError("RGB-D birth assignment object IDs do not match posterior")
            measurement_valid = measured.measurement_mask[batch_index, measurement_index]
            raw = measured.auxiliary["world_position"]
            append_mask[batch_index, belief_index] = True
            valid[batch_index, belief_index] = measurement_valid
            positions[batch_index, belief_index] = raw[batch_index, measurement_index]
        return resolved.append(
            object_ids=posterior.objects.object_id,
            active_mask=posterior.objects.active,
            append_mask=append_mask,
            timestamp=measured.timestamp,
            positions=positions,
            valid_mask=valid,
            minimum_dt=self.config.temporal_min_dt,
        )


__all__ = ["RGBDObservationConfig", "RGBDObservationModule"]
