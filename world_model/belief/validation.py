"""Runtime validators for canonical belief invariants."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor

from world_model.belief.camera_belief import CameraBelief
from world_model.belief.object_belief import ObjectBeliefTensor
from world_model.belief.world_belief import WorldBelief


def _shape(name: str, tensor: Tensor, expected: tuple[int, ...]) -> None:
    if tensor.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {tuple(tensor.shape)}")


def _floating_tensors(objects: ObjectBeliefTensor) -> Iterable[tuple[str, Tensor]]:
    for name in (
        "existence_logit",
        "position",
        "velocity",
        "orientation",
        "angular_velocity",
        "geometry",
        "appearance",
        "residual_dynamics",
        "modal_state",
        "modal_frequency",
        "modal_decay_raw",
        "log_mass",
        "restitution_logit",
        "log_drag",
        "friction_logit",
        "motion_mode_logits",
        "visibility_logit",
        "fast_log_variance",
        "slow_log_variance",
        "parameter_memory",
    ):
        yield name, getattr(objects, name)


def validate_object_belief(
    objects: ObjectBeliefTensor,
    *,
    log_variance_bounds: tuple[float, float] = (-30.0, 20.0),
    quaternion_tolerance: float = 1e-4,
) -> None:
    if objects.object_id.ndim != 2:
        raise ValueError("object_id must have shape [B,N]")
    batch, count = objects.object_id.shape
    base = (batch, count)
    _shape("active", objects.active, base)
    _shape("existence_logit", objects.existence_logit, base)
    _shape("position", objects.position, (*base, 3))
    _shape("velocity", objects.velocity, (*base, 3))
    _shape("orientation", objects.orientation, (*base, 4))
    _shape("angular_velocity", objects.angular_velocity, (*base, 3))
    if objects.geometry.ndim != 3 or objects.geometry.shape[:2] != base:
        raise ValueError("geometry must have shape [B,N,Dg]")
    if objects.appearance.ndim != 3 or objects.appearance.shape[:2] != base:
        raise ValueError("appearance must have shape [B,N,Da]")
    if objects.residual_dynamics.ndim != 3 or objects.residual_dynamics.shape[:2] != base:
        raise ValueError("residual_dynamics must have shape [B,N,Dd]")
    if objects.modal_state.ndim != 5 or objects.modal_state.shape[:2] != base:
        raise ValueError("modal_state must have shape [B,N,K,2,Dm]")
    if objects.modal_state.shape[-2] != 2:
        raise ValueError("modal_state paired axis must have size 2")
    modal_shape = (
        batch,
        count,
        objects.modal_count,
        objects.modal_dim,
    )
    _shape("modal_frequency", objects.modal_frequency, modal_shape)
    _shape("modal_decay_raw", objects.modal_decay_raw, modal_shape)
    for name in ("log_mass", "restitution_logit", "log_drag", "friction_logit"):
        _shape(name, getattr(objects, name), (*base, 1))
    if objects.motion_mode_logits.ndim != 3 or (objects.motion_mode_logits.shape[:2] != base):
        raise ValueError("motion_mode_logits must have shape [B,N,Cmode]")
    _shape("visibility_logit", objects.visibility_logit, base)
    _shape(
        "fast_log_variance",
        objects.fast_log_variance,
        (*base, objects.fast_state_dim),
    )
    _shape(
        "slow_log_variance",
        objects.slow_log_variance,
        (*base, objects.slow_state_dim),
    )
    if objects.parameter_memory.ndim != 3 or objects.parameter_memory.shape[:2] != base:
        raise ValueError("parameter_memory must have shape [B,N,Dh]")

    if objects.object_id.dtype is not torch.int64:
        raise TypeError("object_id must be torch.int64")
    if objects.active.dtype is not torch.bool:
        raise TypeError("active must be torch.bool")
    if objects.age_steps.dtype is not torch.int64:
        raise TypeError("age_steps must be torch.int64")
    if objects.missed_steps.dtype is not torch.int64:
        raise TypeError("missed_steps must be torch.int64")
    _shape("age_steps", objects.age_steps, base)
    _shape("missed_steps", objects.missed_steps, base)

    if torch.any(objects.object_id[objects.active] < 0):
        raise ValueError("active objects must have nonnegative IDs")
    if torch.any(objects.object_id[~objects.active] != -1):
        raise ValueError("inactive padding objects must have ID -1")
    for batch_index in range(batch):
        ids = objects.object_id[batch_index, objects.active[batch_index]]
        if ids.numel() != torch.unique(ids).numel():
            raise ValueError(f"object IDs must be unique in batch {batch_index}")

    reference_device = objects.position.device
    reference_dtype = objects.position.dtype
    for name, value in _floating_tensors(objects):
        if not value.is_floating_point():
            raise TypeError(f"{name} must be floating point")
        if value.device != reference_device:
            raise ValueError(f"{name} is on a different device")
        if value.dtype != reference_dtype:
            raise ValueError(f"{name} has a different dtype")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains NaN or Inf")
    for name in ("object_id", "active", "age_steps", "missed_steps"):
        if getattr(objects, name).device != reference_device:
            raise ValueError(f"{name} is on a different device")

    norms = torch.linalg.vector_norm(objects.orientation, dim=-1)
    if not torch.allclose(
        norms,
        torch.ones_like(norms),
        atol=quaternion_tolerance,
        rtol=quaternion_tolerance,
    ):
        raise ValueError("orientation quaternions must have unit norm")
    if torch.any(objects.modal_frequency < 0):
        raise ValueError("modal frequencies must be nonnegative")
    lower, upper = log_variance_bounds
    for name, value in (
        ("fast_log_variance", objects.fast_log_variance),
        ("slow_log_variance", objects.slow_log_variance),
    ):
        if torch.any((value < lower) | (value > upper)):
            raise ValueError(f"{name} lies outside [{lower}, {upper}]")


def validate_camera_belief(
    camera: CameraBelief,
    *,
    log_variance_bounds: tuple[float, float] = (-30.0, 20.0),
) -> None:
    if camera.world_from_camera.ndim != 3:
        raise ValueError("world_from_camera must have shape [B,4,4]")
    batch = camera.world_from_camera.shape[0]
    _shape("world_from_camera", camera.world_from_camera, (batch, 4, 4))
    _shape("camera.linear_velocity", camera.linear_velocity, (batch, 3))
    _shape("camera.angular_velocity", camera.angular_velocity, (batch, 3))
    _shape("camera.intrinsics", camera.intrinsics, (batch, 3, 3))
    if camera.log_variance.ndim != 2 or camera.log_variance.shape[0] != batch:
        raise ValueError("camera.log_variance must have shape [B,Dcamera]")
    _shape("camera.calibrated", camera.calibrated, (batch,))
    if camera.calibrated.dtype is not torch.bool:
        raise TypeError("camera.calibrated must be torch.bool")
    reference_device = camera.world_from_camera.device
    reference_dtype = camera.world_from_camera.dtype
    for name, value in (
        ("linear_velocity", camera.linear_velocity),
        ("angular_velocity", camera.angular_velocity),
        ("intrinsics", camera.intrinsics),
        ("log_variance", camera.log_variance),
    ):
        if value.device != reference_device or value.dtype != reference_dtype:
            raise ValueError(f"camera {name} has inconsistent device or dtype")
    if camera.calibrated.device != reference_device:
        raise ValueError("camera calibrated mask is on a different device")
    for name, value in (
        ("world_from_camera", camera.world_from_camera),
        ("linear_velocity", camera.linear_velocity),
        ("angular_velocity", camera.angular_velocity),
        ("intrinsics", camera.intrinsics),
        ("log_variance", camera.log_variance),
    ):
        if not torch.isfinite(value).all():
            raise ValueError(f"camera {name} contains NaN or Inf")
    expected_last_row = camera.world_from_camera.new_tensor([0.0, 0.0, 0.0, 1.0])
    if not torch.allclose(
        camera.world_from_camera[:, 3, :],
        expected_last_row.expand(batch, -1),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError("world_from_camera must be a homogeneous transform")
    lower, upper = log_variance_bounds
    if torch.any((camera.log_variance < lower) | (camera.log_variance > upper)):
        raise ValueError("camera log variance lies outside configured bounds")


def validate_world_belief(
    belief: WorldBelief,
    *,
    log_variance_bounds: tuple[float, float] = (-30.0, 20.0),
    quaternion_tolerance: float = 1e-4,
) -> None:
    if belief.timestamp.ndim != 1:
        raise ValueError("world timestamp must have shape [B]")
    batch = belief.timestamp.shape[0]
    if not belief.timestamp.is_floating_point() or not torch.isfinite(belief.timestamp).all():
        raise ValueError("world timestamps must be finite floating-point seconds")
    validate_object_belief(
        belief.objects,
        log_variance_bounds=log_variance_bounds,
        quaternion_tolerance=quaternion_tolerance,
    )
    validate_camera_belief(
        belief.camera,
        log_variance_bounds=log_variance_bounds,
    )
    if belief.objects.batch_size != batch or belief.camera.batch_size != batch:
        raise ValueError("world/object/camera batch sizes must match")
    _shape("gravity", belief.gravity, (batch, 3))
    if belief.global_code.ndim != 2 or belief.global_code.shape[0] != batch:
        raise ValueError("global_code must have shape [B,Dglobal]")
    if belief.global_log_variance.ndim != 2 or belief.global_log_variance.shape[0] != batch:
        raise ValueError("global_log_variance must have shape [B,Dglobal_var]")
    _shape("next_object_id", belief.next_object_id, (batch,))
    if belief.next_object_id.dtype is not torch.int64:
        raise TypeError("next_object_id must be torch.int64")
    reference_device = belief.timestamp.device
    reference_dtype = belief.timestamp.dtype
    for name, value in (
        ("gravity", belief.gravity),
        ("global_code", belief.global_code),
        ("global_log_variance", belief.global_log_variance),
    ):
        if value.device != reference_device or value.dtype != reference_dtype:
            raise ValueError(f"{name} has inconsistent device or dtype")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains NaN or Inf")
    for name, value in (
        ("next_object_id", belief.next_object_id),
        ("objects", belief.objects.position),
        ("camera", belief.camera.world_from_camera),
    ):
        if value.device != reference_device:
            raise ValueError(f"{name} is on a different device")
    if belief.objects.position.dtype != reference_dtype:
        raise ValueError("object belief dtype differs from world belief dtype")
    if belief.camera.world_from_camera.dtype != reference_dtype:
        raise ValueError("camera belief dtype differs from world belief dtype")
    lower, upper = log_variance_bounds
    if torch.any((belief.global_log_variance < lower) | (belief.global_log_variance > upper)):
        raise ValueError("global log variance lies outside configured bounds")


def clamp_log_variance(
    value: Tensor,
    bounds: tuple[float, float] = (-20.0, 10.0),
) -> Tensor:
    """Finite clamp shared by dynamics and filtering updates."""

    if not torch.isfinite(value).all():
        raise ValueError("cannot clamp a non-finite log variance")
    return value.clamp(min=bounds[0], max=bounds[1])
