"""Focused public-episode tests for the specification-1.61 materializer."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
import torch

import world_model.training.dynamic_set_materializer as materializer
from world_model.belief import BeliefFactory
from world_model.datasets import collate_episodes
from world_model.dynamics import AnalyticFreeMotionDynamics, WorldImpulseAction
from world_model.training.dynamic_set_protocol import PHYSICAL_CELLS, physical_manifest
from world_model.training.dynamic_set_scene import (
    DYNAMIC_SET_FRAMES,
    DYNAMIC_SET_IMAGE_SIZE,
    DYNAMIC_SET_MAX_OBJECTS,
    DynamicSetSceneCertificate,
)

DEVELOPMENT_ROWS = physical_manifest("development")


def test_public_materializer_rejects_protected_rows_before_truth_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = physical_manifest("selector")[0]

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("protected simulator truth was generated without evaluator authority")

    monkeypatch.setattr(materializer, "_materialize_dynamic_set_episode", forbidden)
    with pytest.raises(PermissionError, match="governed evaluator"):
        materializer.materialize_dynamic_set_episode(row)


def test_protected_physical_capability_is_ordered_split_bound_and_one_shot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = physical_manifest("selector")[:2]
    generated: list[int] = []

    def counted(row: object, **kwargs: object) -> object:
        del kwargs
        generated.append(row.ordinal)  # type: ignore[attr-defined]
        return object()

    monkeypatch.setattr(materializer, "_materialize_dynamic_set_episode_core", counted)

    def capability():
        return materializer._mint_protected_physical_materialization_capability(
            split="selector",
            rows=rows,
            protocol_sha256="a" * 64,
            manifest_sha256="b" * 64,
            permit_index=0,
            permit_nonce="c" * 64,
        )

    wrong_order = capability()
    with pytest.raises(PermissionError, match="row order"):
        materializer._materialize_protected_dynamic_set_episode(rows[1], capability=wrong_order)
    with pytest.raises(PermissionError, match="closed"):
        materializer._materialize_protected_dynamic_set_episode(rows[0], capability=wrong_order)
    assert generated == []

    wrong_split = capability()
    with pytest.raises(PermissionError, match="public row"):
        materializer._materialize_protected_dynamic_set_episode(
            DEVELOPMENT_ROWS[0], capability=wrong_split
        )
    assert generated == []

    one_row = materializer._mint_protected_physical_materialization_capability(
        split="selector",
        rows=rows[:1],
        protocol_sha256="a" * 64,
        manifest_sha256="b" * 64,
        permit_index=0,
        permit_nonce="c" * 64,
    )
    materializer._materialize_protected_dynamic_set_episode(rows[0], capability=one_row)
    with pytest.raises(PermissionError, match="row order"):
        materializer._materialize_protected_dynamic_set_episode(rows[0], capability=one_row)
    assert generated == [rows[0].ordinal]


def _checkpoint_belief_for_public_action(
    frame: materializer.DynamicSetPublicFrame,
    *,
    object_count: int,
    first_object_id: int = 9_000,
):
    belief = BeliefFactory(max_objects=DYNAMIC_SET_MAX_OBJECTS, appearance_dim=8).create(
        timestamp=frame.timestamp - 1.0 / 20.0,
        gravity=(0.0, 0.0, 0.0),
    )
    objects = belief.objects.clone()
    objects.active[0, :object_count] = True
    objects.object_id[0, :object_count] = torch.arange(
        first_object_id,
        first_object_id + object_count,
        dtype=torch.int64,
    )
    objects.appearance[0, 0] = frame.known_action_appearance_handle[0]
    if object_count > 1:
        objects.appearance[0, 1:object_count] = -frame.known_action_appearance_handle[0]
    return replace(
        belief,
        objects=objects,
        next_object_id=torch.tensor([first_object_id + object_count], dtype=torch.int64),
    ).validate()


@pytest.mark.parametrize("cell_index", range(len(PHYSICAL_CELLS)))
def test_every_physical_cell_materializes_a_complete_deterministic_episode(
    cell_index: int,
) -> None:
    row = DEVELOPMENT_ROWS[cell_index]
    result = materializer.materialize_dynamic_set_episode(row)
    episode = result.episode

    assert row.cell_index == cell_index
    assert result.attempt_count >= 1
    assert result.attempt_count == result.rejection_count + 1
    assert result.certificate.peak_object_count == row.object_count
    assert result.certificate.minimum_visible_fraction == 1.0
    assert result.certificate.minimum_image_clearance_pixels >= 0.0
    assert episode["rgb"].shape == (DYNAMIC_SET_FRAMES, 3, *DYNAMIC_SET_IMAGE_SIZE)
    assert episode["depth"].shape == (DYNAMIC_SET_FRAMES, 1, *DYNAMIC_SET_IMAGE_SIZE)
    assert episode["objects"]["active"].shape == (
        DYNAMIC_SET_FRAMES,
        DYNAMIC_SET_MAX_OBJECTS,
    )
    assert torch.equal(
        episode["labels"]["segmentation_mask"][episode["objects"]["active"]],
        episode["labels"]["full_mask"][episode["objects"]["active"]],
    )
    assert not bool(episode["events"]["boundary_contact"].any())
    assert not bool(episode["events"]["boundary_collision"].any())
    assert (
        bool(episode["events"]["pair_contact"].any() or episode["events"]["pair_collision"].any())
        == row.contact
    )
    if row.dynamic_membership:
        assert result.certificate.lifecycle_birth_count == 1
    else:
        assert result.certificate.lifecycle_birth_count == 0

    if cell_index == 0:
        repeated = materializer.materialize_dynamic_set_episode(row)
        assert repeated.accepted_seed == result.accepted_seed
        assert repeated.rejection_reasons == result.rejection_reasons
        assert torch.equal(repeated.episode["rgb"], episode["rgb"])
        assert torch.equal(repeated.episode["objects"]["position"], episode["objects"]["position"])


def test_action_contact_removal_and_fresh_replacement_are_realized_publicly() -> None:
    # Select by causal factors so this stays a provenance test rather than an
    # accidental assertion about frozen manifest ordinal arithmetic.
    action_removal_row = next(
        row
        for row in DEVELOPMENT_ROWS
        if row.contact_origin == "action_induced"
        and row.dynamic_membership
        and row.lifecycle_schedule == "removal"
    )
    replacement_row = next(
        row
        for row in DEVELOPMENT_ROWS
        if row.contact_origin == "action_induced"
        and row.lifecycle_schedule == "remove_then_birth"
        and row.action_target_rank == row.object_count - 1
        and row.action_time_stratum is not None
        and row.action_time_stratum > 0
    )
    natural_replacement_row = next(
        row
        for row in DEVELOPMENT_ROWS
        if row.contact_origin == "natural"
        and not row.known_action
        and row.lifecycle_schedule == "remove_then_birth"
    )
    action_removal = materializer.materialize_dynamic_set_episode(action_removal_row)
    replacement = materializer.materialize_dynamic_set_episode(replacement_row)
    natural_replacement = materializer.materialize_dynamic_set_episode(natural_replacement_row)

    for result in (action_removal, replacement):
        events = result.episode["events"]
        assert result.certificate.action_induced_pair_collision_count >= 1
        assert result.certificate.known_action_count == 1
        assert torch.equal(events["known_action_observed"], events["externally_actuated"])
        assert torch.equal(events["known_impulse_world"], events["external_impulse"])
        assert not bool(events["external_impulse"][~events["known_action_observed"]].ne(0).any())
        action_frame = int(torch.nonzero(events["known_action_observed"].any(dim=-1))[0])
        action_mask = events["known_action_observed"][action_frame]
        assert torch.equal(
            events["known_action_timestamp"][action_frame, action_mask],
            result.episode["timestamps"][action_frame].expand(int(action_mask.sum())),
        )
        assert bool(
            (events["known_action_timestamp"][~events["known_action_observed"]] == -1).all()
        )

    assert action_removal.certificate.lifecycle_removal_count == 1
    assert action_removal.certificate.lifecycle_birth_count == 0
    assert replacement.certificate.lifecycle_removal_count == 1
    assert replacement.certificate.lifecycle_birth_count == 1
    objects = replacement.episode["objects"]
    events = replacement.episode["events"]
    birth_frame = int(torch.nonzero(events["created"][1:].any(dim=-1))[0]) + 1
    removal_frame = int(torch.nonzero(events["removed"].any(dim=-1))[0])
    birth_slot = int(torch.nonzero(events["created"][birth_frame])[0])
    replacement_id = int(objects["id"][birth_frame, birth_slot])
    earlier_ids = objects["id"][:birth_frame]
    assert replacement_id >= 0
    assert not bool((earlier_ids == replacement_id).any())
    replacement_action = torch.nonzero(events["known_action_observed"], as_tuple=False)
    assert replacement_action.shape == (1, 2)
    replacement_action_frame, replacement_action_slot = replacement_action[0].tolist()
    assert replacement_action_slot == birth_slot
    assert (
        int(events["known_action_object_id"][replacement_action_frame, replacement_action_slot])
        == replacement_id
    )
    old_albedo = objects["albedo"][removal_frame - 1, birth_slot]
    assert not bool(objects["position"][removal_frame:birth_frame, birth_slot].ne(0).any())
    assert not bool(objects["albedo"][removal_frame:birth_frame, birth_slot].ne(0).any())
    assert not torch.equal(objects["albedo"][birth_frame, birth_slot], old_albedo)

    frames = list(replacement.public_frames())
    assert len(frames) == DYNAMIC_SET_FRAMES
    assert [frame.frame_index for frame in frames] == list(range(DYNAMIC_SET_FRAMES))
    assert torch.equal(torch.stack([frame.rgb for frame in frames]), replacement.episode["rgb"])
    assert torch.equal(
        torch.stack([frame.known_action_observed for frame in frames]),
        replacement.known_action_observed.any(dim=-1, keepdim=True),
    )
    assert sum(int(frame.known_action_observed.sum()) for frame in frames) == 1
    action_frame = next(frame for frame in frames if bool(frame.known_action_observed.any()))
    assert not hasattr(action_frame, "known_action_object_id")
    assert action_frame.known_action_observed.shape == (1,)
    assert action_frame.known_action_timestamp[0].item() == pytest.approx(action_frame.timestamp)
    assert action_frame.known_action_appearance_handle.shape == (1, 8)
    assert bool(action_frame.known_impulse_world.ne(0).any())
    assert frames[0].world_impulse_action() is None
    checkpoint_belief = _checkpoint_belief_for_public_action(
        action_frame,
        object_count=replacement.row.object_count,
    )
    public_action = action_frame.world_impulse_action(checkpoint_belief)
    assert public_action is not None
    assert public_action.timestamp.item() == pytest.approx(action_frame.timestamp)
    assert public_action.object_id.tolist() == [9_000]
    private_target_id = int(
        events["known_action_object_id"][events["known_action_observed"]].item()
    )
    assert int(public_action.object_id[0]) != private_target_id
    torch.testing.assert_close(
        public_action.impulse_world,
        action_frame.known_impulse_world,
    )
    public_impulse = action_frame.known_impulse_world.clone()
    public_action.impulse_world.zero_()
    assert torch.equal(action_frame.known_impulse_world, public_impulse)
    assert natural_replacement_row.contact_origin == "natural"
    assert natural_replacement_row.lifecycle_schedule == "remove_then_birth"
    assert natural_replacement.certificate.natural_pair_collision_count >= 1
    assert natural_replacement.certificate.action_induced_pair_collision_count == 0
    assert natural_replacement.certificate.lifecycle_removal_count == 1
    assert natural_replacement.certificate.lifecycle_birth_count == 1
    assert natural_replacement.certificate.known_action_count == 0

    batch = collate_episodes([action_removal.episode, natural_replacement.episode])
    assert batch["rgb"].shape == (2, DYNAMIC_SET_FRAMES, 3, *DYNAMIC_SET_IMAGE_SIZE)
    assert batch["events"]["known_action_observed"].shape == (
        2,
        DYNAMIC_SET_FRAMES,
        DYNAMIC_SET_MAX_OBJECTS,
    )
    expected_action_frame = (10, 15, 34, 44)[action_removal_row.action_time_stratum]
    assert batch["metadata"]["action_frame"].tolist() == [expected_action_frame, -1]


def test_real_contact_cell_balances_both_causal_action_regimes() -> None:
    rows = tuple(DEVELOPMENT_ROWS[5 + cycle * len(PHYSICAL_CELLS)] for cycle in range(2))
    assert [(row.contact_origin, row.known_action) for row in rows] == [
        ("natural", False),
        ("action_induced", True),
    ]

    natural_no_action, induced_known_action = (
        materializer.materialize_dynamic_set_episode(row) for row in rows
    )
    assert natural_no_action.certificate.natural_pair_collision_count >= 1
    assert natural_no_action.certificate.known_action_count == 0
    assert induced_known_action.certificate.natural_pair_collision_count == 0
    assert induced_known_action.certificate.action_induced_pair_collision_count >= 1
    assert induced_known_action.certificate.known_action_count == 1
    assert all(
        "counterfactual_no_action_pair_collision" not in item.episode["events"]
        for item in (
            natural_no_action,
            induced_known_action,
        )
    )


@pytest.mark.parametrize(
    ("cell_index", "cell_cycle", "expected_action_frame"),
    ((5, 1, 10), (5, 5, 15), (5, 9, 34), (5, 13, 44)),
)
def test_all_action_time_strata_preserve_lifecycle_and_induce_contact(
    cell_index: int,
    cell_cycle: int,
    expected_action_frame: int,
) -> None:
    row = DEVELOPMENT_ROWS[cell_index + cell_cycle * len(PHYSICAL_CELLS)]
    result = materializer.materialize_dynamic_set_episode(row)
    events = result.episode["events"]
    action_rows = torch.nonzero(events["known_action_observed"].any(dim=-1)).flatten()

    assert action_rows.tolist() == [expected_action_frame]
    assert result.certificate.known_action_count == 1
    assert result.certificate.action_induced_pair_collision_count >= 1
    assert result.certificate.natural_pair_collision_count == 0
    collision_rows = torch.nonzero(events["pair_collision"].any(dim=(-2, -1))).flatten()
    assert collision_rows.numel() >= 1
    assert int(collision_rows.min()) == expected_action_frame + 6
    objects = result.episode["objects"]
    stable_slice = slice(expected_action_frame - 1, expected_action_frame + 7)
    assert bool(
        objects["active"][stable_slice].eq(objects["active"][expected_action_frame - 1]).all()
    )
    assert bool(objects["id"][stable_slice].eq(objects["id"][expected_action_frame - 1]).all())
    assert row.contact_origin == "action_induced"
    assert row.lifecycle_schedule in {"birth", "removal", "remove_then_birth"}


def test_second_action_stratum_precedes_remove_then_birth_without_rank_skew() -> None:
    row = next(
        value
        for value in DEVELOPMENT_ROWS
        if value.contact_origin == "action_induced"
        and value.lifecycle_schedule == "remove_then_birth"
        and value.action_time_stratum == 1
        and value.action_target_rank == value.object_count - 1
    )
    result = materializer.materialize_dynamic_set_episode(row)
    events = result.episode["events"]
    action_frame = int(torch.nonzero(events["known_action_observed"].any(dim=-1))[0])
    target_slot = row.object_count - 1

    assert action_frame == 15
    assert bool(events["known_action_observed"][action_frame, target_slot])
    assert int(events["known_action_object_id"][action_frame, target_slot]) == int(
        result.episode["objects"]["id"][action_frame, target_slot]
    )
    assert bool(result.episode["objects"]["active"][14:22, target_slot].all())


def test_rejection_sampler_advances_a_deterministic_candidate_seed(monkeypatch) -> None:
    row = DEVELOPMENT_ROWS[0]
    known = torch.zeros(DYNAMIC_SET_FRAMES, DYNAMIC_SET_MAX_OBJECTS, dtype=torch.bool)
    no_action = torch.zeros(
        DYNAMIC_SET_FRAMES,
        DYNAMIC_SET_MAX_OBJECTS,
        DYNAMIC_SET_MAX_OBJECTS,
        dtype=torch.bool,
    )
    candidate = SimpleNamespace(
        episode={"candidate": True},
        known_action_observed=known,
        counterfactual_no_action_pair_collision=no_action,
    )
    candidate_seeds: list[int] = []
    preflight_calls = 0
    certificate = DynamicSetSceneCertificate(
        peak_object_count=1,
        natural_pair_collision_count=0,
        action_induced_pair_collision_count=0,
        lifecycle_birth_count=0,
        lifecycle_removal_count=0,
        known_action_count=0,
        minimum_visible_fraction=1.0,
        minimum_image_clearance_pixels=1.0,
    )

    def fake_build(_row, candidate_seed):
        candidate_seeds.append(candidate_seed)
        return candidate

    def reject_once(
        _episode,
        _row,
        *,
        known_action_observed,
        counterfactual_no_action_pair_collision,
    ):
        nonlocal preflight_calls
        assert known_action_observed is known
        assert counterfactual_no_action_pair_collision is no_action
        preflight_calls += 1
        if preflight_calls == 1:
            raise ValueError("deterministic containment rejection")
        return certificate

    monkeypatch.setattr(materializer, "_build_candidate_episode", fake_build)
    monkeypatch.setattr(materializer, "preflight_dynamic_set_episode", reject_once)
    monkeypatch.setattr(materializer, "_preflight_lifecycle_visibility", lambda _episode: None)
    result = materializer.materialize_dynamic_set_episode(row, maximum_attempts=2)

    assert result.attempt_count == 2
    assert result.rejection_count == 1
    assert result.rejection_rate == 0.5
    assert "containment rejection" in result.rejection_reasons[0]
    assert candidate_seeds[0] == row.seed
    assert candidate_seeds[1] != candidate_seeds[0]
    assert result.accepted_seed == candidate_seeds[1]


def test_birth_truth_is_absent_before_the_public_creation_frame() -> None:
    row = DEVELOPMENT_ROWS[1]
    result = materializer.materialize_dynamic_set_episode(row)
    episode = result.episode
    birth_frame = int(torch.nonzero(episode["events"]["created"][1:].any(dim=-1))[0]) + 1
    birth_slot = int(torch.nonzero(episode["events"]["created"][birth_frame])[0])

    assert not bool(episode["objects"]["active"][:birth_frame, birth_slot].any())
    assert not bool(episode["objects"]["position"][:birth_frame, birth_slot].ne(0).any())
    assert not bool(episode["objects"]["velocity"][:birth_frame, birth_slot].ne(0).any())
    assert not bool(episode["objects"]["albedo"][:birth_frame, birth_slot].ne(0).any())
    assert bool(episode["objects"]["position"][birth_frame, birth_slot].ne(0).any())
    assert bool(episode["objects"]["albedo"][birth_frame, birth_slot].ne(0).any())


def test_lifecycle_preflight_binds_birth_and_removal_to_visible_state_transitions() -> None:
    birth = materializer.materialize_dynamic_set_episode(DEVELOPMENT_ROWS[1]).episode
    birth_frame = int(torch.nonzero(birth["events"]["created"][1:].any(dim=-1))[0]) + 1
    birth_slot = int(torch.nonzero(birth["events"]["created"][birth_frame])[0])
    bad_birth = {
        **birth,
        "objects": {**birth["objects"], "active": birth["objects"]["active"].clone()},
    }
    bad_birth["objects"]["active"][birth_frame, birth_slot] = False
    with pytest.raises(ValueError, match="creation must produce an active"):
        materializer._preflight_lifecycle_visibility(bad_birth)

    stale_birth = {
        **birth,
        "labels": {
            **birth["labels"],
            "projected_valid": birth["labels"]["projected_valid"].clone(),
        },
    }
    stale_birth["labels"]["projected_valid"][birth_frame - 1, birth_slot] = True
    with pytest.raises(ValueError, match="pre-event projection"):
        materializer._preflight_lifecycle_visibility(stale_birth)

    removal_row = DEVELOPMENT_ROWS[3 + len(PHYSICAL_CELLS)]
    removal = materializer.materialize_dynamic_set_episode(removal_row).episode
    removal_frame = int(torch.nonzero(removal["events"]["removed"].any(dim=-1))[0])
    removal_slot = int(torch.nonzero(removal["events"]["removed"][removal_frame])[0])
    bad_removal = {
        **removal,
        "objects": {
            **removal["objects"],
            "visible_fraction": removal["objects"]["visible_fraction"].clone(),
        },
    }
    bad_removal["objects"]["visible_fraction"][removal_frame - 1, removal_slot] = 0.5
    with pytest.raises(ValueError, match="fully visible immediately beforehand"):
        materializer._preflight_lifecycle_visibility(bad_removal)

    stale_removal = {
        **removal,
        "labels": {
            **removal["labels"],
            "projected_valid": removal["labels"]["projected_valid"].clone(),
        },
    }
    stale_removal["labels"]["projected_valid"][removal_frame, removal_slot] = True
    with pytest.raises(ValueError, match="event-frame projection"):
        materializer._preflight_lifecycle_visibility(stale_removal)


def test_public_action_ledger_rejects_an_undeclared_simulator_target(monkeypatch) -> None:
    row = DEVELOPMENT_ROWS[2 + 22]
    original = materializer._known_action_impulse

    def add_hidden_target(row, pair, camera):
        impulse = original(row, pair, camera)
        hidden_slot = 0 if row.action_target_rank != 0 else 1
        impulse[hidden_slot, 0] = 0.25
        return impulse

    monkeypatch.setattr(materializer, "_known_action_impulse", add_hidden_target)
    with pytest.raises(materializer.DynamicSetMaterializationError, match="undeclared public"):
        materializer.materialize_dynamic_set_episode(row, maximum_attempts=1)


def test_public_action_timestamp_matches_endpoint_impulse_semantics() -> None:
    row = DEVELOPMENT_ROWS[2 + len(PHYSICAL_CELLS)]
    action_candidate = materializer._build_candidate_episode(row, row.seed)
    action_episode = action_candidate.episode
    action_frame = int(
        torch.nonzero(action_episode["events"]["known_action_observed"].any(dim=-1))[0]
    )
    action_mask = action_episode["events"]["known_action_observed"][action_frame]
    target_slot = int(torch.nonzero(action_mask)[0])
    action_timestamp = action_episode["events"]["known_action_timestamp"][action_frame, target_slot]
    assert action_timestamp == action_episode["timestamps"][action_frame]
    assert action_timestamp > action_episode["timestamps"][action_frame - 1]

    action_free_row = replace(
        row,
        known_action=False,
        action_target_rank=None,
        action_time_stratum=None,
    )
    action_free_episode = materializer._build_candidate_episode(action_free_row, row.seed).episode
    torch.testing.assert_close(
        action_episode["objects"]["position"][: action_frame + 1],
        action_free_episode["objects"]["position"][: action_frame + 1],
        rtol=0.0,
        atol=0.0,
    )
    simulator_delta_velocity = (
        action_episode["objects"]["velocity"][action_frame, target_slot]
        - action_free_episode["objects"]["velocity"][action_frame, target_slot]
    )
    impulse = action_episode["events"]["known_impulse_world"][action_frame, target_slot]
    mass = action_episode["objects"]["mass"][action_frame, target_slot]
    torch.testing.assert_close(simulator_delta_velocity, impulse / mass, rtol=0.0, atol=0.0)

    source = BeliefFactory(max_objects=DYNAMIC_SET_MAX_OBJECTS, appearance_dim=8).create(
        timestamp=action_episode["timestamps"][action_frame - 1],
        gravity=(0.0, 0.0, 0.0),
    )
    truth = action_free_episode["objects"]
    objects = source.objects.clone()
    objects.active[0] = truth["active"][action_frame - 1]
    objects.object_id[0] = truth["id"][action_frame - 1]
    objects.position[0] = truth["position"][action_frame - 1]
    objects.velocity[0] = truth["velocity"][action_frame - 1]
    objects.geometry[0, :, 0] = truth["radius"][action_frame - 1, :, 0]
    objects.log_mass[0] = truth["mass"][action_frame - 1].log()
    objects.log_drag[0] = truth["drag"][action_frame - 1].log()
    objects.restitution_logit[0] = torch.logit(truth["restitution"][action_frame - 1])
    objects.friction_logit[0] = torch.logit(truth["friction"][action_frame - 1])
    objects.age_steps[0, objects.active[0]] = 16
    source = replace(
        source,
        objects=objects,
        next_object_id=torch.tensor([row.object_count], dtype=torch.int64),
    ).validate()
    model_action = WorldImpulseAction(
        timestamp=action_timestamp.unsqueeze(0),
        object_id=truth["id"][action_frame, target_slot].unsqueeze(0),
        impulse_world=impulse.unsqueeze(0),
    )
    query = source.timestamp.new_tensor([[1.0 / 20.0]])
    dynamics = AnalyticFreeMotionDynamics()
    model_free = dynamics.rollout(source, query)
    model_acted = dynamics.rollout(source, query, action=model_action)
    torch.testing.assert_close(model_acted.positions, model_free.positions, rtol=0.0, atol=0.0)
    model_delta_velocity = (
        model_acted.velocities[0, 0, target_slot] - model_free.velocities[0, 0, target_slot]
    )
    torch.testing.assert_close(model_delta_velocity, impulse / mass, rtol=1.0e-6, atol=1.0e-7)


def test_n1_replacement_action_binds_only_through_its_public_appearance() -> None:
    row = DEVELOPMENT_ROWS[111]
    assert row.object_count == 1
    assert row.lifecycle_schedule == "remove_then_birth"
    assert row.known_action
    result = materializer.materialize_dynamic_set_episode(row)
    action_frame = next(
        frame for frame in result.public_frames() if bool(frame.known_action_observed[0])
    )
    checkpoint_belief = _checkpoint_belief_for_public_action(
        action_frame,
        object_count=1,
        first_object_id=88_001,
    )

    action = action_frame.world_impulse_action(checkpoint_belief)

    assert action is not None
    assert action.object_id.tolist() == [88_001]
    assert not hasattr(action_frame, "known_action_object_id")
    private_id = int(
        result.episode["events"]["known_action_object_id"][
            result.episode["events"]["known_action_observed"]
        ].item()
    )
    assert private_id != int(action.object_id[0])


def test_public_action_accepts_frozen_dynamic_set_uncertainty_lower_bound() -> None:
    result = materializer.materialize_dynamic_set_episode(DEVELOPMENT_ROWS[26])
    action_frame = next(
        frame for frame in result.public_frames() if bool(frame.known_action_observed[0])
    )
    belief = _checkpoint_belief_for_public_action(action_frame, object_count=2)
    belief = replace(
        belief,
        objects=replace(
            belief.objects,
            fast_log_variance=torch.full_like(belief.objects.fast_log_variance, -32.0),
        ),
    )

    action = action_frame.world_impulse_action(belief)

    assert action is not None
    assert action.object_id.tolist() == [9_000]


def test_public_frames_own_their_tensor_storage() -> None:
    result = materializer.materialize_dynamic_set_episode(DEVELOPMENT_ROWS[0])
    frame = next(result.public_frames())
    original_rgb = result.episode["rgb"][0].clone()
    original_intrinsics = result.episode["camera"]["intrinsics"][0].clone()

    frame.rgb.zero_()
    frame.intrinsics.zero_()

    assert torch.equal(result.episode["rgb"][0], original_rgb)
    assert torch.equal(result.episode["camera"]["intrinsics"][0], original_intrinsics)
    assert frame.world_impulse_action() is None
    frame.known_action_appearance_handle[0, 0] = 1.0
    with pytest.raises(ValueError, match="only sentinels"):
        frame.world_impulse_action()


def test_public_boundary_is_exact_row_bound_and_deterministic() -> None:
    result = materializer.materialize_dynamic_set_episode(DEVELOPMENT_ROWS[0])

    frames, evidence = result.public_frames_with_boundary()
    repeated_frames, repeated = result.public_frames_with_boundary()

    assert len(frames) == DYNAMIC_SET_FRAMES
    assert all(type(frame) is materializer.DynamicSetPublicFrame for frame in frames)
    assert all(not hasattr(frame, "__dict__") for frame in frames)
    assert evidence.truth_bound is True
    assert evidence.frame_count == DYNAMIC_SET_FRAMES
    assert evidence.exact_frame_type_count == DYNAMIC_SET_FRAMES
    assert evidence.exact_schema_frame_count == DYNAMIC_SET_FRAMES
    assert evidence.public_tensor_count == DYNAMIC_SET_FRAMES * 8
    assert evidence.truth_tensor_count > 0
    assert evidence.truth_storage_alias_count == 0
    assert evidence.public_storage_alias_count == 0
    assert evidence.truth_leakage_count == 0
    assert evidence.boundary_sha256 == repeated.boundary_sha256
    assert all(
        torch.equal(first.rgb, second.rgb)
        for first, second in zip(frames, repeated_frames, strict=True)
    )


def test_public_boundary_rejects_frame_subclass_and_unexpected_field() -> None:
    result = materializer.materialize_dynamic_set_episode(DEVELOPMENT_ROWS[0])
    frames, _ = result.public_frames_with_boundary()

    @dataclass(frozen=True, slots=True)
    class ExtendedPublicFrame(materializer.DynamicSetPublicFrame):
        private_object_id: int

    first = frames[0]
    extended = ExtendedPublicFrame(
        **{
            name: getattr(first, name)
            for name in (
                "frame_index",
                "timestamp",
                "rgb",
                "depth",
                "world_from_camera",
                "intrinsics",
                "known_action_observed",
                "known_action_timestamp",
                "known_action_appearance_handle",
                "known_impulse_world",
            )
        },
        private_object_id=17,
    )
    tampered = (extended, *frames[1:])

    evidence = materializer.inspect_dynamic_set_public_boundary(result, tampered)

    assert evidence.unexpected_frame_type_count == 1
    assert evidence.subclass_frame_count == 1
    assert evidence.unexpected_field_count == 1
    assert evidence.truth_leakage_count > 0
    with pytest.raises(ValueError, match="leakage violation"):
        materializer.certify_dynamic_set_public_boundary(result, tampered)


def test_public_boundary_rejects_tensor_subclass_without_invoking_it() -> None:
    result = materializer.materialize_dynamic_set_episode(DEVELOPMENT_ROWS[0])
    frames, _ = result.public_frames_with_boundary()

    class HostileTensor(torch.Tensor):
        @staticmethod
        def __new__(cls, value: torch.Tensor) -> HostileTensor:
            return torch.Tensor._make_subclass(cls, value, False)

        def detach(self) -> torch.Tensor:
            raise AssertionError("boundary inspection invoked an untrusted Tensor method")

    tampered = (replace(frames[0], rgb=HostileTensor(frames[0].rgb)), *frames[1:])

    evidence = materializer.inspect_dynamic_set_public_boundary(result, tampered)

    assert evidence.invalid_public_field_count == 1
    assert evidence.truth_leakage_count > 0
    with pytest.raises(ValueError, match="leakage violation"):
        materializer.certify_dynamic_set_public_boundary(result, tampered)


def test_public_boundary_rejects_truth_and_public_storage_aliases() -> None:
    result = materializer.materialize_dynamic_set_episode(DEVELOPMENT_ROWS[0])
    frames, _ = result.public_frames_with_boundary()
    objects = dict(result.episode["objects"])
    objects["injected_public_alias"] = frames[0].rgb
    aliased_materialization = replace(
        result,
        episode={**result.episode, "objects": objects},
    )

    truth_alias = materializer.inspect_dynamic_set_public_boundary(
        aliased_materialization,
        frames,
    )
    public_alias_frames = list(frames)
    public_alias_frames[1] = replace(
        public_alias_frames[1],
        intrinsics=public_alias_frames[0].intrinsics.detach(),
    )
    public_alias = materializer.inspect_dynamic_set_public_boundary(
        result,
        tuple(public_alias_frames),
    )

    assert truth_alias.truth_storage_alias_count == 1
    assert truth_alias.truth_leakage_count > 0
    assert public_alias.public_storage_alias_count == 1
    assert public_alias.truth_leakage_count > 0
    with pytest.raises(ValueError, match="leakage violation"):
        materializer.certify_dynamic_set_public_boundary(aliased_materialization, frames)


def test_materializer_rejects_rows_that_disagree_with_the_frozen_cell() -> None:
    row = DEVELOPMENT_ROWS[0]
    with pytest.raises(ValueError, match="disagree"):
        materializer.materialize_dynamic_set_episode(replace(row, object_count=2))
    with pytest.raises(ValueError, match="positive integer"):
        materializer.materialize_dynamic_set_episode(row, maximum_attempts=True)
    with pytest.raises(TypeError, match="known_action must be boolean"):
        materializer.materialize_dynamic_set_episode(replace(row, known_action=1))
    with pytest.raises(ValueError, match="lifecycle schedule"):
        materializer.materialize_dynamic_set_episode(
            replace(row, dynamic_membership=True, lifecycle_schedule="future_birth")
        )
