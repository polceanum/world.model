from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from world_model.training.dynamic_set_protocol import PhysicalManifestRow
from world_model.training.dynamic_set_scene import preflight_dynamic_set_episode


def _row(
    *,
    object_count: int = 1,
    contact: bool = False,
    dynamic: bool = False,
    lifecycle: str = "none",
    known_action: bool = False,
    contact_origin: str = "none",
) -> PhysicalManifestRow:
    return PhysicalManifestRow(
        split="development",
        ordinal=0,
        seed=71_000_000,
        cell_index=0,
        object_count=object_count,
        contact=contact,
        dynamic_membership=dynamic,
        lifecycle_schedule=lifecycle,  # type: ignore[arg-type]
        known_action=known_action,
        contact_origin=contact_origin,  # type: ignore[arg-type]
        action_target_rank=0 if known_action else None,
        action_time_stratum=0 if known_action else None,
        camera_stratum=0,
        contact_geometry="head_on" if contact else "none",
        distribution="in_distribution",
    )


def _episode(count: int = 1) -> dict[str, object]:
    frames, slots, size = 56, 6, 64
    active = torch.zeros(frames, slots, dtype=torch.bool)
    active[:, :count] = True
    object_id = torch.full((frames, slots), -1, dtype=torch.int64)
    object_id[:, :count] = torch.arange(count, dtype=torch.int64)
    segmentation = torch.zeros(frames, slots, size, size, dtype=torch.bool)
    centres = torch.zeros(frames, slots, 2)
    radii = torch.zeros(frames, slots)
    for slot in range(count):
        x = 12 + 9 * slot
        segmentation[:, slot, 30:33, x : x + 3] = True
        centres[:, slot] = torch.tensor([x + 1.0, 31.0])
        radii[:, slot] = 1.0
    depth = torch.zeros(frames, 1, size, size)
    depth[:, 0][segmentation.any(dim=1)] = 2.0
    created = torch.zeros_like(active)
    created[0, :count] = True
    pair = torch.zeros(frames, slots, slots, dtype=torch.bool)
    boundary = torch.zeros(frames, slots, 6, dtype=torch.bool)
    fixed = lambda value: torch.full((frames, slots, 1), value)  # noqa: E731
    return {
        "rgb": torch.zeros(frames, 3, size, size),
        "depth": depth,
        "timestamps": torch.arange(frames, dtype=torch.float32) / 20.0,
        "objects": {
            "active": active,
            "id": object_id,
            "radius": fixed(0.21),
            "mass": fixed(1.0),
            "drag": fixed(0.05),
            "restitution": fixed(0.7),
            "friction": fixed(0.2),
            "visible_fraction": active.to(torch.float32),
        },
        "labels": {
            "projected_valid": active.clone(),
            "segmentation_mask": segmentation,
            "full_mask": segmentation.clone(),
            "projected_center_pixels": centres,
            "apparent_radius": radii,
        },
        "events": {
            "created": created,
            "removed": torch.zeros_like(active),
            "pair_contact": pair.clone(),
            "pair_collision": pair.clone(),
            "boundary_contact": boundary.clone(),
            "boundary_collision": boundary.clone(),
            "externally_actuated": torch.zeros_like(active),
        },
        "camera": {"calibrated": torch.ones(frames, dtype=torch.bool)},
    }


def test_complete_static_scene_passes_and_hidden_impulse_fails() -> None:
    episode = _episode()
    known = torch.zeros(56, 6, dtype=torch.bool)
    certificate = preflight_dynamic_set_episode(episode, _row(), known_action_observed=known)
    assert certificate.peak_object_count == 1
    assert certificate.minimum_visible_fraction == 1.0
    assert certificate.minimum_image_clearance_pixels >= 0.0

    hidden = deepcopy(episode)
    hidden["events"]["externally_actuated"][12, 0] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="public"):
        preflight_dynamic_set_episode(hidden, _row(), known_action_observed=known)


def test_remove_then_birth_requires_order_and_fresh_persistent_id() -> None:
    episode = _episode()
    active = episode["objects"]["active"]  # type: ignore[index]
    ids = episode["objects"]["id"]  # type: ignore[index]
    created = episode["events"]["created"]  # type: ignore[index]
    removed = episode["events"]["removed"]  # type: ignore[index]
    active[20:30, 0] = False
    ids[20:30, 0] = -1
    ids[30:, 0] = 7
    removed[20, 0] = True
    created[30, 0] = True
    for name in ("visible_fraction",):
        episode["objects"][name] = active.to(torch.float32)  # type: ignore[index]
    episode["labels"]["projected_valid"] = active.clone()  # type: ignore[index]
    for name in ("segmentation_mask", "full_mask"):
        masks = episode["labels"][name]  # type: ignore[index]
        masks[20:30, 0] = False
    known = torch.zeros_like(active)
    row = _row(dynamic=True, lifecycle="remove_then_birth")
    certificate = preflight_dynamic_set_episode(episode, row, known_action_observed=known)
    assert certificate.lifecycle_birth_count == 1
    assert certificate.lifecycle_removal_count == 1

    ids[30:, 0] = 0
    with pytest.raises(ValueError, match="fresh persistent ID"):
        preflight_dynamic_set_episode(episode, row, known_action_observed=known)


def test_action_induced_contact_may_follow_action_in_a_later_interval() -> None:
    episode = _episode(2)
    known = torch.zeros(56, 6, dtype=torch.bool)
    known[5, 0] = True
    episode["events"]["externally_actuated"] = known.clone()  # type: ignore[index]
    for name in ("pair_contact", "pair_collision"):
        episode["events"][name][10, 0, 1] = True  # type: ignore[index]
        episode["events"][name][10, 1, 0] = True  # type: ignore[index]
    row = _row(
        object_count=2,
        contact=True,
        known_action=True,
        contact_origin="action_induced",
    )

    no_action = torch.zeros(56, 6, 6, dtype=torch.bool)
    certificate = preflight_dynamic_set_episode(
        episode,
        row,
        known_action_observed=known,
        counterfactual_no_action_pair_collision=no_action,
    )
    assert certificate.action_induced_pair_collision_count == 1

    no_action[10, 0, 1] = True
    no_action[10, 1, 0] = True
    with pytest.raises(ValueError, match="also occurs"):
        preflight_dynamic_set_episode(
            episode,
            row,
            known_action_observed=known,
            counterfactual_no_action_pair_collision=no_action,
        )


def test_known_noncausal_action_retains_natural_collision_in_paired_control() -> None:
    episode = _episode(2)
    known = torch.zeros(56, 6, dtype=torch.bool)
    known[5, 0] = True
    episode["events"]["externally_actuated"] = known.clone()  # type: ignore[index]
    for name in ("pair_contact", "pair_collision"):
        episode["events"][name][10, 0, 1] = True  # type: ignore[index]
        episode["events"][name][10, 1, 0] = True  # type: ignore[index]
    no_action = torch.zeros(56, 6, 6, dtype=torch.bool)
    no_action[9, 0, 1] = True
    no_action[9, 1, 0] = True
    row = _row(
        object_count=2,
        contact=True,
        known_action=True,
        contact_origin="natural",
    )

    certificate = preflight_dynamic_set_episode(
        episode,
        row,
        known_action_observed=known,
        counterfactual_no_action_pair_collision=no_action,
    )
    assert certificate.natural_pair_collision_count == 1
    assert certificate.action_induced_pair_collision_count == 0


def test_action_induced_preflight_rejects_unrelated_post_action_collision() -> None:
    episode = _episode(3)
    known = torch.zeros(56, 6, dtype=torch.bool)
    known[5, 0] = True
    episode["events"]["externally_actuated"] = known.clone()  # type: ignore[index]
    for frame, pair in ((10, (0, 1)), (12, (1, 2))):
        for name in ("pair_contact", "pair_collision"):
            first, second = pair
            episode["events"][name][frame, first, second] = True  # type: ignore[index]
            episode["events"][name][frame, second, first] = True  # type: ignore[index]
    row = _row(
        object_count=3,
        contact=True,
        known_action=True,
        contact_origin="action_induced",
    )

    with pytest.raises(ValueError, match="unrelated"):
        preflight_dynamic_set_episode(
            episode,
            row,
            known_action_observed=known,
            counterfactual_no_action_pair_collision=torch.zeros(56, 6, 6, dtype=torch.bool),
        )
