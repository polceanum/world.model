"""Explicit canonical packing maps for belief means."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

import torch
from torch import Tensor

from world_model.belief.object_belief import ObjectBeliefTensor


@dataclass(frozen=True)
class PackingMap:
    """Named half-open slices into one canonical packed state vector."""

    fields: Mapping[str, slice]
    size: int

    def __getitem__(self, name: str) -> slice:
        return self.fields[name]


def _make_map(widths: tuple[tuple[str, int], ...]) -> PackingMap:
    offset = 0
    fields: dict[str, slice] = {}
    for name, width in widths:
        fields[name] = slice(offset, offset + width)
        offset += width
    return PackingMap(fields=fields, size=offset)


def fast_packing_map(objects: ObjectBeliefTensor) -> PackingMap:
    return _make_map(
        (
            ("position", 3),
            ("velocity", 3),
            ("orientation", 4),
            ("angular_velocity", 3),
            ("modal_state", objects.modal_count * 2 * objects.modal_dim),
        )
    )


def slow_packing_map(objects: ObjectBeliefTensor) -> PackingMap:
    return _make_map(
        (
            ("log_mass", 1),
            ("restitution_logit", 1),
            ("log_drag", 1),
            ("friction_logit", 1),
            ("geometry", objects.geometry_dim),
            ("appearance", objects.appearance_dim),
            ("residual_dynamics", objects.residual_dynamics_dim),
        )
    )


def pack_fast_state(objects: ObjectBeliefTensor) -> Tensor:
    """Pack fast means to ``[B,N,Dfast]`` in the documented canonical order."""

    return torch.cat(
        (
            objects.position,
            objects.velocity,
            objects.orientation,
            objects.angular_velocity,
            objects.modal_state.flatten(start_dim=2),
        ),
        dim=-1,
    )


def unpack_fast_state(
    packed: Tensor,
    template: ObjectBeliefTensor,
) -> ObjectBeliefTensor:
    """Replace fast means in ``template`` from a packed state."""

    packing = fast_packing_map(template)
    expected = (*template.object_id.shape, packing.size)
    if packed.shape != expected:
        raise ValueError(f"packed fast state must have shape {expected}")
    modes = packed[..., packing["modal_state"]].reshape_as(template.modal_state)
    return replace(
        template,
        position=packed[..., packing["position"]],
        velocity=packed[..., packing["velocity"]],
        orientation=packed[..., packing["orientation"]],
        angular_velocity=packed[..., packing["angular_velocity"]],
        modal_state=modes,
    )


def pack_slow_state(objects: ObjectBeliefTensor) -> Tensor:
    """Pack slow physical parameters/codes to ``[B,N,Dslow]``."""

    return torch.cat(
        (
            objects.log_mass,
            objects.restitution_logit,
            objects.log_drag,
            objects.friction_logit,
            objects.geometry,
            objects.appearance,
            objects.residual_dynamics,
        ),
        dim=-1,
    )


def unpack_slow_state(
    packed: Tensor,
    template: ObjectBeliefTensor,
) -> ObjectBeliefTensor:
    """Replace slow means in ``template`` from a packed state."""

    packing = slow_packing_map(template)
    expected = (*template.object_id.shape, packing.size)
    if packed.shape != expected:
        raise ValueError(f"packed slow state must have shape {expected}")
    return replace(
        template,
        log_mass=packed[..., packing["log_mass"]],
        restitution_logit=packed[..., packing["restitution_logit"]],
        log_drag=packed[..., packing["log_drag"]],
        friction_logit=packed[..., packing["friction_logit"]],
        geometry=packed[..., packing["geometry"]],
        appearance=packed[..., packing["appearance"]],
        residual_dynamics=packed[..., packing["residual_dynamics"]],
    )


# Concise aliases are convenient inside filtering code.
pack_fast = pack_fast_state
unpack_fast = unpack_fast_state
pack_slow = pack_slow_state
unpack_slow = unpack_slow_state
