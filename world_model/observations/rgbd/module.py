"""Public one-slot RGB-D observation module.

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
from torch import Tensor

from world_model.belief import fast_packing_map
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
from world_model.observations.rgbd.sphere_centres import (
    RGBDSphereCentreMeasurementModule,
)
from world_model.observations.rgbd.temporal import RGBDTemporalPositionHistory
from world_model.observations.rgbd.two_disc_geometry import (
    two_disc_geometry_from_rgbd,
)

if TYPE_CHECKING:
    from world_model.belief import BirthAssignments, WorldBelief
    from world_model.fusion.association import AssociationResult


@dataclass(frozen=True)
class RGBDObservationConfig:
    """Checkpointed priors and numerical controls for the RGB-D bridge.

    Radius remains a fixed public prior unless metric-radius estimation is
    explicitly enabled, in which case the module emits typed radius evidence
    from RGB-D surface geometry.
    """

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
    metric_radius_estimation_enabled: bool = False
    minimum_world_radius: float = 0.05
    maximum_world_radius: float = 1.0
    measurement_radius_variance: float = 1.0e-5
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

    def __post_init__(self) -> None:
        if (
            isinstance(self.proposal_count, bool)
            or not isinstance(self.proposal_count, int)
            or self.proposal_count not in {1, 2}
        ):
            raise ValueError("RGB-D proposal_count must be integer one or two")
        if (
            isinstance(self.appearance_dim, bool)
            or not isinstance(self.appearance_dim, int)
            or self.appearance_dim <= 0
        ):
            raise ValueError("RGB-D appearance_dim must be a positive integer")
        if self.proposal_count == 2 and self.appearance_dim != 3:
            raise ValueError("two-object RGB-D requires appearance_dim exactly three")
        if not isinstance(self.metric_radius_estimation_enabled, bool):
            raise TypeError("RGB-D metric_radius_estimation_enabled must be boolean")
        if self.metric_radius_estimation_enabled and self.proposal_count != 2:
            raise ValueError("RGB-D metric radius estimation requires exactly two proposals")
        positive = {
            "world_radius": self.world_radius,
            "minimum_world_radius": self.minimum_world_radius,
            "maximum_world_radius": self.maximum_world_radius,
            "measurement_radius_variance": self.measurement_radius_variance,
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
        if self.metric_radius_estimation_enabled:
            if self.minimum_world_radius >= self.maximum_world_radius:
                raise ValueError("RGB-D world-radius bounds must be strictly ordered")
            if not self.minimum_world_radius <= self.world_radius <= self.maximum_world_radius:
                raise ValueError("RGB-D world_radius must lie within its declared bounds")
        for name, value in (
            ("temporal_history_size", self.temporal_history_size),
            ("temporal_min_samples", self.temporal_min_samples),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"RGB-D {name} must be an integer of at least two")
        if self.temporal_history_size != 16:
            raise ValueError("the first RGB-D bridge requires exactly 16 history samples")
        if self.temporal_min_samples != self.temporal_history_size:
            raise ValueError("RGB-D temporal_min_samples must equal temporal_history_size")
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


@register_observation_module("rgbd")
class RGBDObservationModule(ObservationModule):
    """Parameter-free metric RGB-D observation with raw free-motion history."""

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

    def validate_packet(self, packet: ObservationPacket) -> None:
        if packet.modality != self.modality_name:
            raise ValueError("RGB-D module accepts only modality='rgbd'")
        rgb, _ = _composite_payload(packet)
        _batched_calibration(packet, batch=rgb.shape[0], reference=rgb)
        _explicit_image_size(packet, rgb)

    def _measure(self, packet: ObservationPacket) -> MeasurementSet:
        self.validate_packet(packet)
        rgb, depth = _composite_payload(packet)
        world_from_camera, intrinsics = _batched_calibration(
            packet,
            batch=rgb.shape[0],
            reference=rgb,
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
                estimate_world_radius=self.config.metric_radius_estimation_enabled,
                minimum_world_radius=self.config.minimum_world_radius,
                maximum_world_radius=self.config.maximum_world_radius,
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
        if self.config.metric_radius_estimation_enabled:
            measured_world_radius = measured.surface_fit_radius.unsqueeze(-1)
            measured_radius_log_variance = rgb.new_full(
                (batch, proposals, 1),
                math.log(self.config.measurement_radius_variance),
            )
            measured_radius_valid = measured.valid_mask
            supported_state_fields = ("position", "radius")
        else:
            measured_world_radius = rgb.new_full(
                (batch, proposals, 1),
                self.config.world_radius,
            )
            measured_radius_log_variance = None
            measured_radius_valid = None
            supported_state_fields = ("position",)
        radius_auxiliary: dict[str, Tensor] = {}
        if measured_radius_log_variance is not None and measured_radius_valid is not None:
            radius_auxiliary = {
                "world_radius_log_variance": measured_radius_log_variance,
                "world_radius_valid_mask": measured_radius_valid,
            }
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
            supported_state_fields=supported_state_fields,
            auxiliary={
                "world_position": measured.world_position,
                "world_log_variance": log_variance,
                "world_position_log_variance": log_variance,
                "world_position_independent_axis_mask": valid_axes,
                "world_radius": measured_world_radius,
                **radius_auxiliary,
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
        if belief.objects.max_objects != self.config.proposal_count:
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
            appearance=(objects.appearance if self.config.proposal_count == 2 else None),
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
        sample_count = self.config.temporal_history_size
        degrees_of_freedom = sample_count - 2
        residual_covariance = fit.residual_covariance * (sample_count / degrees_of_freedom)
        coefficient_scale = inverse_normal[..., 1, 1] / sample_count
        variance = residual_covariance.diagonal(dim1=-2, dim2=-1) * coefficient_scale.unsqueeze(-1)
        variance = variance.clamp_min(self.config.temporal_velocity_variance_floor)
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
        resolved = resolved.append(
            object_ids=posterior.objects.object_id,
            active_mask=posterior.objects.active,
            append_mask=posterior.objects.active,
            timestamp=measured.timestamp,
            positions=positions,
            valid_mask=valid,
            minimum_dt=self.config.temporal_min_dt,
        )
        fit, fit_valid = resolved.fit(
            gravity=posterior.gravity,
            drag=posterior.objects.drag,
            minimum_support=self.config.temporal_min_samples,
            minimum_dt=self.config.temporal_min_dt,
            conditioning_limit=self.config.fit_conditioning_limit,
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
