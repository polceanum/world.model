"""Observable scene preflight for specification-1.61 episode materializers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from world_model.training.dynamic_set_protocol import PhysicalManifestRow

DYNAMIC_SET_FRAMES = 56
DYNAMIC_SET_FRAME_RATE_HZ = 20.0
DYNAMIC_SET_IMAGE_SIZE = (64, 64)
DYNAMIC_SET_VERTICAL_FOV_DEGREES = 32.0
DYNAMIC_SET_MAX_OBJECTS = 6
DYNAMIC_SET_RADIUS_M = 0.21
DYNAMIC_SET_MASS = 1.0
DYNAMIC_SET_DRAG = 0.05
DYNAMIC_SET_RESTITUTION = 0.7
DYNAMIC_SET_FRICTION = 0.2


@dataclass(frozen=True)
class DynamicSetSceneCertificate:
    peak_object_count: int
    natural_pair_collision_count: int
    action_induced_pair_collision_count: int
    lifecycle_birth_count: int
    lifecycle_removal_count: int
    known_action_count: int
    minimum_visible_fraction: float
    minimum_image_clearance_pixels: float


def _tensor(mapping: Mapping[str, Any], name: str) -> Tensor:
    value = mapping.get(name)
    if not isinstance(value, Tensor):
        raise TypeError(f"episode field {name!r} must be a tensor")
    return value


def _assert_fixed_parameter(
    objects: Mapping[str, Any],
    active: Tensor,
    *,
    name: str,
    expected: float,
) -> None:
    value = _tensor(objects, name)
    support = active.unsqueeze(-1) if value.ndim == active.ndim + 1 else active
    observed = value[support]
    if observed.numel() and not torch.equal(observed, observed.new_full(observed.shape, expected)):
        raise ValueError(f"active objects must use fixed {name}={expected}")


def _lifecycle_counts(events: Mapping[str, Any], active: Tensor) -> tuple[int, int]:
    created = _tensor(events, "created")
    removed = _tensor(events, "removed")
    if created.shape != active.shape or removed.shape != active.shape:
        raise ValueError("lifecycle event tensors must have shape [T,N]")
    later_births = int(created[1:].sum())
    removals = int(removed.sum())
    return later_births, removals


def _validate_lifecycle(
    row: PhysicalManifestRow,
    events: Mapping[str, Any],
    objects: Mapping[str, Any],
    active: Tensor,
) -> tuple[int, int]:
    births, removals = _lifecycle_counts(events, active)
    expected = {
        "none": (0, 0),
        "birth": (1, 0),
        "removal": (0, 1),
        "remove_then_birth": (1, 1),
    }[row.lifecycle_schedule]
    if (births, removals) != expected:
        raise ValueError(
            f"lifecycle schedule {row.lifecycle_schedule!r} expected {expected}, "
            f"got {(births, removals)}"
        )
    if row.lifecycle_schedule == "remove_then_birth":
        created = _tensor(events, "created")
        removed = _tensor(events, "removed")
        removal_frames = torch.nonzero(removed.any(dim=-1), as_tuple=False).flatten()
        birth_frames = torch.nonzero(created[1:].any(dim=-1), as_tuple=False).flatten() + 1
        if removal_frames.numel() != 1 or birth_frames.numel() != 1:
            raise ValueError("remove-then-birth requires one unambiguous event of each kind")
        if int(removal_frames[0]) >= int(birth_frames[0]):
            raise ValueError("remove-then-birth events are not temporally ordered")
        ids = _tensor(objects, "id")
        birth_slot = int(torch.nonzero(created[int(birth_frames[0])], as_tuple=False)[0])
        new_id = int(ids[int(birth_frames[0]), birth_slot])
        earlier_ids = ids[: int(birth_frames[0])][ids[: int(birth_frames[0])] >= 0]
        if bool((earlier_ids == new_id).any()):
            raise ValueError("slot reuse must allocate a fresh persistent ID")
    return births, removals


def preflight_dynamic_set_episode(
    episode: Mapping[str, Any],
    row: PhysicalManifestRow,
    *,
    known_action_observed: Tensor,
    counterfactual_no_action_pair_collision: Tensor | None = None,
) -> DynamicSetSceneCertificate:
    """Reject any episode outside the complete-observation physical family.

    ``known_action_observed`` is supplied by the public action materializer,
    not inferred from truth labels.  Equality with simulator actuation is
    checked here so an unannounced impulse can never become deterministic
    training support.  Known-action contact rows additionally require the
    private paired collision trace obtained by replaying the same initial
    scene and lifecycle with only the impulse removed.
    """

    row_cell_count = row.object_count
    if not 1 <= row_cell_count <= DYNAMIC_SET_MAX_OBJECTS:
        raise ValueError("manifest object_count lies outside [1,6]")
    rgb = _tensor(episode, "rgb")
    depth = _tensor(episode, "depth")
    timestamps = _tensor(episode, "timestamps")
    if rgb.shape != (DYNAMIC_SET_FRAMES, 3, *DYNAMIC_SET_IMAGE_SIZE):
        raise ValueError("dynamic-set RGB must have shape [56,3,64,64]")
    if depth.shape != (DYNAMIC_SET_FRAMES, 1, *DYNAMIC_SET_IMAGE_SIZE):
        raise ValueError("dynamic-set depth must have shape [56,1,64,64]")
    expected_timestamps = torch.arange(DYNAMIC_SET_FRAMES, dtype=timestamps.dtype) / (
        DYNAMIC_SET_FRAME_RATE_HZ
    )
    if timestamps.device.type != "cpu" or not torch.equal(timestamps, expected_timestamps):
        raise ValueError("dynamic-set timestamps must be the exact CPU 20 Hz grid")
    if not bool(torch.isfinite(rgb).all()) or not bool(torch.isfinite(depth).all()):
        raise ValueError("complete RGB-D observations must be finite")

    objects_value = episode.get("objects")
    labels_value = episode.get("labels")
    events_value = episode.get("events")
    camera_value = episode.get("camera")
    if not all(
        isinstance(value, Mapping)
        for value in (objects_value, labels_value, events_value, camera_value)
    ):
        raise TypeError("objects, labels, events, and camera must be mappings")
    objects = objects_value
    labels = labels_value
    events = events_value
    camera = camera_value
    active = _tensor(objects, "active")
    if active.shape != (DYNAMIC_SET_FRAMES, DYNAMIC_SET_MAX_OBJECTS):
        raise ValueError("dynamic-set objects must use six padded slots")
    if active.dtype is not torch.bool:
        raise TypeError("objects.active must be boolean")
    active_count = active.sum(dim=-1)
    peak_count = int(active_count.max())
    if peak_count != row_cell_count:
        raise ValueError("episode peak cardinality differs from its physical cell")

    for name, expected in (
        ("radius", DYNAMIC_SET_RADIUS_M),
        ("mass", DYNAMIC_SET_MASS),
        ("drag", DYNAMIC_SET_DRAG),
        ("restitution", DYNAMIC_SET_RESTITUTION),
        ("friction", DYNAMIC_SET_FRICTION),
    ):
        _assert_fixed_parameter(objects, active, name=name, expected=expected)

    calibrated = _tensor(camera, "calibrated")
    if calibrated.shape != (DYNAMIC_SET_FRAMES,) or calibrated.dtype is not torch.bool:
        raise ValueError("camera.calibrated must be boolean [56]")
    if not bool(calibrated.all()):
        raise ValueError("every camera transform must be known and calibrated")

    visible_fraction = _tensor(objects, "visible_fraction")
    projected_valid = _tensor(labels, "projected_valid")
    segmentation = _tensor(labels, "segmentation_mask")
    full_mask = _tensor(labels, "full_mask")
    if visible_fraction.shape != active.shape or projected_valid.shape != active.shape:
        raise ValueError("visibility/projected validity must have shape [56,6]")
    if bool((active & ~projected_valid).any()):
        raise ValueError("every active object must remain projectable")
    if bool((visible_fraction[active] < 1.0).any()):
        raise ValueError("zero pixel occlusion requires visible_fraction exactly one")
    if segmentation.shape != (DYNAMIC_SET_FRAMES, DYNAMIC_SET_MAX_OBJECTS, *DYNAMIC_SET_IMAGE_SIZE):
        raise ValueError("segmentation masks must have shape [56,6,64,64]")
    if full_mask.shape != segmentation.shape:
        raise ValueError("full masks must match segmentation masks")
    if bool((segmentation[active] != full_mask[active]).any()):
        raise ValueError("zero pixel occlusion requires visible and full masks to agree")
    foreground = segmentation.any(dim=1)
    if bool((foreground & (depth[:, 0] <= 0.0)).any()):
        raise ValueError("every visible foreground pixel needs a positive depth return")

    centres = _tensor(labels, "projected_center_pixels")
    radii = _tensor(labels, "apparent_radius")
    if centres.shape != (DYNAMIC_SET_FRAMES, DYNAMIC_SET_MAX_OBJECTS, 2):
        raise ValueError("projected centres must have shape [56,6,2]")
    if radii.shape != active.shape:
        raise ValueError("apparent radii must have shape [56,6]")
    x_clearance = torch.minimum(centres[..., 0] - radii, 63.0 - centres[..., 0] - radii)
    y_clearance = torch.minimum(centres[..., 1] - radii, 63.0 - centres[..., 1] - radii)
    clearance = torch.minimum(x_clearance, y_clearance)
    if bool((clearance[active] < 0.0).any()):
        raise ValueError("every active silhouette must remain inside the image")

    boundary_contact = _tensor(events, "boundary_contact")
    boundary_collision = _tensor(events, "boundary_collision")
    if bool(boundary_contact.any()) or bool(boundary_collision.any()):
        raise ValueError("dynamic-set scenes exclude boundary contact")
    pair_contact = _tensor(events, "pair_contact")
    pair_collision = _tensor(events, "pair_collision")
    expected_pair_shape = (
        DYNAMIC_SET_FRAMES,
        DYNAMIC_SET_MAX_OBJECTS,
        DYNAMIC_SET_MAX_OBJECTS,
    )
    if pair_contact.shape != expected_pair_shape or pair_collision.shape != expected_pair_shape:
        raise ValueError("pair contact/collision must have shape [56,6,6]")
    if pair_contact.dtype is not torch.bool or pair_collision.dtype is not torch.bool:
        raise TypeError("pair contact/collision must be boolean")
    if not torch.equal(pair_contact, pair_contact.transpose(-1, -2)) or not torch.equal(
        pair_collision,
        pair_collision.transpose(-1, -2),
    ):
        raise ValueError("pair contact/collision evidence must be symmetric")
    upper = torch.triu(torch.ones(6, 6, dtype=torch.bool), diagonal=1)
    contact_count_by_frame = (pair_contact & upper).sum(dim=(-2, -1))
    collision_count_by_frame = (pair_collision & upper).sum(dim=(-2, -1))
    if int(contact_count_by_frame.max()) > 1 or int(collision_count_by_frame.max()) > 1:
        raise ValueError("contact complexity exceeds one pair per observation interval")
    has_contact = bool(pair_contact.any() or pair_collision.any())
    if has_contact != row.contact:
        raise ValueError("episode contact presence differs from its physical cell")

    externally_actuated = _tensor(events, "externally_actuated")
    if known_action_observed.shape != active.shape or known_action_observed.dtype is not torch.bool:
        raise ValueError("known_action_observed must be boolean [56,6]")
    if not torch.equal(known_action_observed, externally_actuated):
        raise ValueError(
            "every simulator impulse must be public and every public action must occur"
        )
    known_action_count = int(known_action_observed.sum())
    if known_action_count != int(row.known_action):
        raise ValueError("known-action count differs from the manifest")

    births, removals = _validate_lifecycle(row, events, objects, active)
    observed_collision = pair_collision & upper
    natural_collision_count = int(observed_collision.sum()) if not row.known_action else 0
    action_collision_count = 0
    if row.known_action and row.contact:
        no_action_collision = counterfactual_no_action_pair_collision
        if (
            not isinstance(no_action_collision, Tensor)
            or no_action_collision.device.type != "cpu"
            or no_action_collision.shape != expected_pair_shape
            or no_action_collision.dtype is not torch.bool
        ):
            raise ValueError("paired no-action collision evidence must be boolean [56,6,6]")
        if not torch.equal(no_action_collision, no_action_collision.transpose(-1, -2)):
            raise ValueError("paired no-action collision evidence must be symmetric")
        action_frames = torch.nonzero(
            known_action_observed.any(dim=-1),
            as_tuple=False,
        ).flatten()
        if action_frames.numel() != 1:
            raise ValueError("a known-action contact row requires one unambiguous action frame")
        action_frame = int(action_frames[0])
        post_action = torch.arange(DYNAMIC_SET_FRAMES) > action_frame
        counterfactual_observed = no_action_collision & upper

        if row.contact_origin == "action_induced":
            if bool(observed_collision[: action_frame + 1].any()):
                raise ValueError("action-induced contact has a collision before its action")
            if bool(counterfactual_observed.any()):
                raise ValueError(
                    "action-induced collision also occurs when the known impulse is removed"
                )
            causal_collision = observed_collision & post_action[:, None, None]
            if not bool(causal_collision.any()):
                raise ValueError(
                    "action-induced contact cell lacks a collision caused by its known action"
                )
            target_slot = int(torch.nonzero(known_action_observed[action_frame])[0])
            unrelated = causal_collision.clone()
            unrelated[:, target_slot, :] = False
            unrelated[:, :, target_slot] = False
            if bool(unrelated.any()):
                raise ValueError(
                    "action-induced episode contains an unrelated post-action collision"
                )
            if int(causal_collision.any(dim=0).sum()) != 1:
                raise ValueError(
                    "action-induced episode must certify exactly one causal collision pair"
                )
            action_collision_count = int(causal_collision.sum())
        elif row.contact_origin == "natural":
            # The paired control is the causal negative control.  Exact frame
            # equality is intentionally unnecessary: a small orthogonal public
            # action may shift numerical contact timing by one observation,
            # while the same unordered collision pair must remain present.
            actual_pairs = observed_collision.any(dim=0)
            no_action_pairs = counterfactual_observed.any(dim=0)
            if not bool(no_action_pairs.any()):
                raise ValueError("natural contact disappears when its known action is removed")
            if not torch.equal(actual_pairs, no_action_pairs):
                raise ValueError(
                    "known noncausal action changes the natural collision-pair identity"
                )
            natural_collision_count = int(observed_collision.sum())
        else:
            raise ValueError("known-action contact row has invalid contact provenance")
    elif row.contact_origin == "natural" and natural_collision_count == 0:
        raise ValueError("natural-contact cell lacks a natural pair collision")

    return DynamicSetSceneCertificate(
        peak_object_count=peak_count,
        natural_pair_collision_count=natural_collision_count,
        action_induced_pair_collision_count=action_collision_count,
        lifecycle_birth_count=births,
        lifecycle_removal_count=removals,
        known_action_count=known_action_count,
        minimum_visible_fraction=float(visible_fraction[active].min()),
        minimum_image_clearance_pixels=float(clearance[active].min()),
    )


__all__ = [
    "DYNAMIC_SET_DRAG",
    "DYNAMIC_SET_FRAME_RATE_HZ",
    "DYNAMIC_SET_VERTICAL_FOV_DEGREES",
    "DYNAMIC_SET_FRAMES",
    "DYNAMIC_SET_FRICTION",
    "DYNAMIC_SET_IMAGE_SIZE",
    "DYNAMIC_SET_MASS",
    "DYNAMIC_SET_MAX_OBJECTS",
    "DYNAMIC_SET_RADIUS_M",
    "DYNAMIC_SET_RESTITUTION",
    "DynamicSetSceneCertificate",
    "preflight_dynamic_set_episode",
]
