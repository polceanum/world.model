"""Typed contracts for executable predictive abstractions.

These contracts are derived views of :class:`WorldBelief`.  They never replace
the persistent belief or become an independent runtime state.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch
from torch import Tensor

from world_model.belief._base import TensorDataclassMixin


class AbstractionKind(IntEnum):
    """Predictive model family selected for an entity at the current scale."""

    POINT_TRAJECTORY = 0
    RIGID_SPHERE = 1


class AbstractionReason(IntEnum):
    """Why the router selected an abstraction family."""

    FREE_MOTION = 0
    CONTACT_OR_EVENT = 1


class PredictiveTokenType(IntEnum):
    """Stable token vocabulary for the current belief-token adapter."""

    SCENE = 0
    CAMERA = 1
    ENTITY_KINEMATIC = 2
    ENTITY_PROGRAM = 3
    ENTITY_LIFECYCLE = 4


@dataclass(frozen=True)
class AbstractionSpec:
    """One executable abstraction registered with the router."""

    kind: AbstractionKind
    name: str
    execution_operator: str
    required_state_fields: tuple[str, ...]
    complexity_cost: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("abstraction name must be nonempty")
        if not self.execution_operator:
            raise ValueError("execution_operator must be nonempty")
        if not self.required_state_fields:
            raise ValueError("required_state_fields must be nonempty")
        if self.complexity_cost <= 0:
            raise ValueError("complexity_cost must be positive")


@dataclass
class AbstractionAssignment(TensorDataclassMixin):
    """Selected abstraction per padded entity slot.

    All tensors have shape ``[B,N]``.  ``confidence`` describes confidence in
    the model-family choice, not confidence in the physical state itself.
    State uncertainty remains authoritative in ``WorldBelief``.
    """

    kind: Tensor
    confidence: Tensor
    complexity_cost: Tensor
    reason: Tensor
    active_mask: Tensor

    def validate(self) -> AbstractionAssignment:
        if self.kind.ndim != 2:
            raise ValueError("abstraction kind must have shape [B,N]")
        shape = self.kind.shape
        for name, value in (
            ("confidence", self.confidence),
            ("complexity_cost", self.complexity_cost),
            ("reason", self.reason),
            ("active_mask", self.active_mask),
        ):
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {tuple(shape)}")
        if self.kind.dtype is not torch.int64 or self.reason.dtype is not torch.int64:
            raise TypeError("abstraction kind and reason must be torch.int64")
        if self.active_mask.dtype is not torch.bool:
            raise TypeError("abstraction active_mask must be torch.bool")
        if not self.confidence.is_floating_point() or not self.complexity_cost.is_floating_point():
            raise TypeError("abstraction confidence and complexity must be floating point")
        reference_device = self.kind.device
        reference_dtype = self.confidence.dtype
        for name, value in (
            ("confidence", self.confidence),
            ("complexity_cost", self.complexity_cost),
            ("reason", self.reason),
            ("active_mask", self.active_mask),
        ):
            if value.device != reference_device:
                raise ValueError(f"abstraction {name} is on a different device")
        if self.complexity_cost.dtype != reference_dtype:
            raise ValueError("abstraction confidence and complexity dtypes must match")
        if not torch.isfinite(self.confidence).all():
            raise ValueError("abstraction confidence must be finite")
        if not torch.isfinite(self.complexity_cost).all():
            raise ValueError("abstraction complexity must be finite")
        if torch.any((self.confidence < 0) | (self.confidence > 1)):
            raise ValueError("abstraction confidence must lie in [0,1]")
        if torch.any(self.complexity_cost < 0):
            raise ValueError("abstraction complexity must be nonnegative")
        active_kinds = self.kind[self.active_mask]
        minimum_kind = min(int(kind) for kind in AbstractionKind)
        maximum_kind = max(int(kind) for kind in AbstractionKind)
        if torch.any((active_kinds < minimum_kind) | (active_kinds > maximum_kind)):
            raise ValueError("active slot has an unknown abstraction kind")
        return self


@dataclass(frozen=True)
class BeliefTokenSchema:
    """Dimensions needed to reversibly serialize a belief into typed tokens."""

    max_objects: int
    fast_state_dim: int
    slow_state_dim: int
    modal_parameter_dim: int
    parameter_memory_dim: int
    motion_mode_dim: int
    global_code_dim: int
    global_variance_dim: int
    camera_variance_dim: int
    token_width: int

    @property
    def sequence_length(self) -> int:
        return 2 + 3 * self.max_objects


@dataclass
class PredictiveTokenBatch(TensorDataclassMixin):
    """Reversible, typed token view of a persistent ``WorldBelief``.

    ``values`` has shape ``[B,L,Dtoken]``.  Token types and object-slot indices
    are structural metadata, so a future transformer does not have to infer
    whether a vector represents a scene, camera, kinematic state, dynamical
    programme, or lifecycle state.
    """

    values: Tensor
    valid_mask: Tensor
    token_type: Tensor
    object_slot: Tensor
    object_id: Tensor
    abstraction_kind: Tensor
    timestamp: Tensor
    next_object_id: Tensor
    camera_calibrated: Tensor
    schema: BeliefTokenSchema

    def validate(self) -> PredictiveTokenBatch:
        if self.values.ndim != 3:
            raise ValueError("token values must have shape [B,L,Dtoken]")
        batch, length, width = self.values.shape
        if length != self.schema.sequence_length or width != self.schema.token_width:
            raise ValueError("token values do not match their schema")
        for name, value in (
            ("valid_mask", self.valid_mask),
            ("object_id", self.object_id),
            ("abstraction_kind", self.abstraction_kind),
        ):
            if value.shape != (batch, length):
                raise ValueError(f"{name} must have shape [B,L]")
        if self.token_type.shape != (length,) or self.object_slot.shape != (length,):
            raise ValueError("token_type and object_slot must have shape [L]")
        if self.timestamp.shape != (batch,) or self.next_object_id.shape != (batch,):
            raise ValueError("timestamp and next_object_id must have shape [B]")
        if self.camera_calibrated.shape != (batch,):
            raise ValueError("camera_calibrated must have shape [B]")
        if self.valid_mask.dtype is not torch.bool:
            raise TypeError("token valid_mask must be torch.bool")
        if self.camera_calibrated.dtype is not torch.bool:
            raise TypeError("camera_calibrated must be torch.bool")
        for name, value in (
            ("token_type", self.token_type),
            ("object_slot", self.object_slot),
            ("object_id", self.object_id),
            ("abstraction_kind", self.abstraction_kind),
            ("next_object_id", self.next_object_id),
        ):
            if value.dtype is not torch.int64:
                raise TypeError(f"{name} must be torch.int64")
        if not self.values.is_floating_point() or not torch.isfinite(self.values).all():
            raise ValueError("token values must be finite floating point")
        if not self.timestamp.is_floating_point() or not torch.isfinite(self.timestamp).all():
            raise ValueError("token timestamps must be finite floating point")
        reference_device = self.values.device
        for name, value in (
            ("valid_mask", self.valid_mask),
            ("token_type", self.token_type),
            ("object_slot", self.object_slot),
            ("object_id", self.object_id),
            ("abstraction_kind", self.abstraction_kind),
            ("timestamp", self.timestamp),
            ("next_object_id", self.next_object_id),
            ("camera_calibrated", self.camera_calibrated),
        ):
            if value.device != reference_device:
                raise ValueError(f"{name} is on a different device from token values")
        if self.timestamp.dtype != self.values.dtype:
            raise ValueError("token values and timestamps must have the same dtype")
        if self.token_type[0] != int(PredictiveTokenType.SCENE):
            raise ValueError("token zero must be the scene token")
        if self.token_type[1] != int(PredictiveTokenType.CAMERA):
            raise ValueError("token one must be the camera token")
        if not self.valid_mask[:, :2].all():
            raise ValueError("scene and camera tokens must always be valid")
        entity_valid = self.valid_mask[:, 2:]
        if torch.any(self.object_id[:, 2:][entity_valid] < 0):
            raise ValueError("valid entity tokens must have nonnegative object IDs")
        if torch.any(self.object_id[:, 2:][~entity_valid] != -1):
            raise ValueError("invalid entity tokens must have object ID -1")
        entity_ids = self.object_id[:, 2:].reshape(
            batch,
            self.schema.max_objects,
            3,
        )
        entity_kinds = self.abstraction_kind[:, 2:].reshape(
            batch,
            self.schema.max_objects,
            3,
        )
        if not torch.equal(entity_ids, entity_ids[..., :1].expand_as(entity_ids)):
            raise ValueError("an entity's three tokens must share one object ID")
        if not torch.equal(entity_kinds, entity_kinds[..., :1].expand_as(entity_kinds)):
            raise ValueError("an entity's three tokens must share one abstraction kind")
        return self
