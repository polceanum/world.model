from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

import world_model.training.loop as training_loop
from world_model.belief import BeliefFactory
from world_model.datasets import collate_episodes
from world_model.runtime import OnlineWorldModel
from world_model.simulator import generate_episode
from world_model.training.loop import (
    _belief_state_losses,
    _parameter_supervision_masks,
    _reset_parameter_history_for_identity_change,
    _runtime_observed_belief_slots,
    _target_observed_runtime_ids,
    run_closed_loop_batch,
)
from world_model.utils.config import load_config


def _batch(*, frames: int, object_count: int) -> dict[str, object]:
    active = torch.ones(1, frames, object_count, dtype=torch.bool)
    velocity = torch.zeros(1, frames, object_count, 3)
    velocity[..., 0] = 1.0
    restitution = torch.full((1, frames, object_count, 1), 0.5)
    return {
        "objects": {
            "active": active,
            "velocity": velocity,
            "restitution": restitution,
        },
        "events": {
            "collision": torch.zeros(1, frames, object_count, dtype=torch.bool),
            "contact": torch.zeros(1, frames, object_count, dtype=torch.bool),
            "externally_actuated": torch.zeros(
                1,
                frames,
                object_count,
                dtype=torch.bool,
            ),
            "pair_collision": torch.zeros(
                1,
                frames,
                object_count,
                object_count,
                dtype=torch.bool,
            ),
            "boundary_collision": torch.zeros(
                1,
                frames,
                object_count,
                2,
                dtype=torch.bool,
            ),
        },
    }


def _identity_slots(object_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.arange(object_count, dtype=torch.int64)[None],
        torch.ones(1, object_count, dtype=torch.bool),
    )


def test_newborn_slot_cannot_open_slow_parameter_observation_gate() -> None:
    belief = BeliefFactory(max_objects=1).create()
    objects = belief.objects.clone()
    objects.active[0, 0] = True
    objects.object_id[0, 0] = 7
    belief = belief.replace(objects=objects)
    model = SimpleNamespace(
        updater=SimpleNamespace(
            last_diagnostics=SimpleNamespace(
                observed_mask=torch.tensor([[False]]),
            )
        )
    )

    observed = _runtime_observed_belief_slots(model, belief)

    assert not observed.any()


def test_runtime_identity_change_resets_parameter_temporal_baseline() -> None:
    indices = torch.tensor([[0, 1]])
    matched = torch.tensor([[True, True]])
    observed = torch.tensor([[True, False]])
    runtime_ids = torch.tensor([[12, 13]])
    observed_ids = _target_observed_runtime_ids(
        indices,
        matched,
        observed,
        runtime_ids,
        target_count=2,
    )
    torch.testing.assert_close(observed_ids, torch.tensor([[12, -1]]))

    frames, reset = _reset_parameter_history_for_identity_change(
        torch.tensor([[4, 5]], dtype=torch.int64),
        torch.tensor([[9, 13]], dtype=torch.int64),
        observed_ids,
    )

    torch.testing.assert_close(frames, torch.tensor([[-1, 5]], dtype=torch.int64))
    torch.testing.assert_close(reset, torch.tensor([[True, False]]))


def test_drag_requires_two_runtime_observations_not_instantaneous_speed() -> None:
    batch = _batch(frames=2, object_count=1)
    indices, matched = _identity_slots(1)
    config = load_config("configs/tiny_overfit.yaml")

    first, history = _parameter_supervision_masks(
        batch,
        config,
        0,
        indices=indices,
        matched=matched,
        runtime_observed=torch.tensor([[True]]),
        last_observed_target_frame=torch.tensor([[-1]], dtype=torch.int64),
    )
    assert not first.drag.any()
    assert first.drag_speed_only_rejected.item()
    torch.testing.assert_close(history, torch.tensor([[0]], dtype=torch.int64))

    second, history = _parameter_supervision_masks(
        batch,
        config,
        1,
        indices=indices,
        matched=matched,
        runtime_observed=torch.tensor([[True]]),
        last_observed_target_frame=history,
    )
    assert second.temporal_baseline.item()
    assert second.drag.item()
    torch.testing.assert_close(history, torch.tensor([[1]], dtype=torch.int64))


def test_drag_rejects_unobserved_or_intervened_temporal_intervals() -> None:
    config = load_config("configs/tiny_overfit.yaml")
    indices, matched = _identity_slots(1)
    last_observed = torch.tensor([[0]], dtype=torch.int64)

    unobserved_batch = _batch(frames=2, object_count=1)
    unobserved, unchanged_history = _parameter_supervision_masks(
        unobserved_batch,
        config,
        1,
        indices=indices,
        matched=matched,
        runtime_observed=torch.tensor([[False]]),
        last_observed_target_frame=last_observed,
    )
    assert not unobserved.drag.any()
    torch.testing.assert_close(unchanged_history, last_observed)

    collision_batch = _batch(frames=2, object_count=1)
    collision_batch["events"]["collision"][0, 1, 0] = True
    collision, _ = _parameter_supervision_masks(
        collision_batch,
        config,
        1,
        indices=indices,
        matched=matched,
        runtime_observed=torch.tensor([[True]]),
        last_observed_target_frame=last_observed,
    )
    assert collision.temporal_baseline.item()
    assert not collision.drag.any()

    actuation_batch = _batch(frames=2, object_count=1)
    actuation_batch["events"]["externally_actuated"][0, 1, 0] = True
    actuated, _ = _parameter_supervision_masks(
        actuation_batch,
        config,
        1,
        indices=indices,
        matched=matched,
        runtime_observed=torch.tensor([[True]]),
        last_observed_target_frame=last_observed,
    )
    assert not actuated.drag.any()


def test_pair_restitution_labels_only_the_minimum_restitution_partner() -> None:
    batch = _batch(frames=2, object_count=2)
    batch["objects"]["restitution"][0, :, 0, 0] = 0.35
    batch["objects"]["restitution"][0, :, 1, 0] = 0.80
    batch["events"]["collision"][0, 1] = True
    batch["events"]["pair_collision"][0, 1, 0, 1] = True
    batch["events"]["pair_collision"][0, 1, 1, 0] = True
    indices, matched = _identity_slots(2)

    masks, _ = _parameter_supervision_masks(
        batch,
        load_config("configs/tiny_overfit.yaml"),
        1,
        indices=indices,
        matched=matched,
        runtime_observed=torch.tensor([[True, True]]),
        last_observed_target_frame=torch.tensor([[0, 0]], dtype=torch.int64),
    )

    torch.testing.assert_close(
        masks.pair_restitution,
        torch.tensor([[True, False]]),
    )
    torch.testing.assert_close(
        masks.restitution,
        torch.tensor([[True, False]]),
    )
    torch.testing.assert_close(
        masks.pair_higher_restitution_rejected,
        torch.tensor([[False, True]]),
    )


def test_equal_pair_restitution_identifies_both_partners() -> None:
    batch = _batch(frames=2, object_count=2)
    batch["events"]["collision"][0, 1] = True
    batch["events"]["pair_collision"][0, 1, 0, 1] = True
    batch["events"]["pair_collision"][0, 1, 1, 0] = True
    indices, matched = _identity_slots(2)

    masks, _ = _parameter_supervision_masks(
        batch,
        load_config("configs/tiny_overfit.yaml"),
        1,
        indices=indices,
        matched=matched,
        runtime_observed=torch.tensor([[True, True]]),
        last_observed_target_frame=torch.tensor([[0, 0]], dtype=torch.int64),
    )

    assert masks.pair_restitution.all()
    assert not masks.pair_higher_restitution_rejected.any()


def test_boundary_impact_identifies_the_impacted_object_directly() -> None:
    batch = _batch(frames=2, object_count=2)
    batch["objects"]["restitution"][0, :, 1, 0] = 0.95
    batch["events"]["collision"][0, 1, 1] = True
    batch["events"]["boundary_collision"][0, 1, 1, 0] = True
    indices, matched = _identity_slots(2)

    masks, _ = _parameter_supervision_masks(
        batch,
        load_config("configs/tiny_overfit.yaml"),
        1,
        indices=indices,
        matched=matched,
        runtime_observed=torch.tensor([[False, True]]),
        last_observed_target_frame=torch.tensor([[-1, 0]], dtype=torch.int64),
    )

    torch.testing.assert_close(
        masks.boundary_restitution,
        torch.tensor([[False, True]]),
    )
    torch.testing.assert_close(
        masks.restitution,
        torch.tensor([[False, True]]),
    )


def test_old_collision_label_is_not_observable_without_preimpact_rgb_evidence() -> None:
    batch = _batch(frames=3, object_count=2)
    batch["events"]["collision"][0, 1] = True
    batch["events"]["pair_collision"][0, 1, 0, 1] = True
    batch["events"]["pair_collision"][0, 1, 1, 0] = True
    indices, matched = _identity_slots(2)
    config = load_config("configs/tiny_overfit.yaml")

    first_seen_after_impact, _ = _parameter_supervision_masks(
        batch,
        config,
        2,
        indices=indices,
        matched=matched,
        runtime_observed=torch.tensor([[True, True]]),
        last_observed_target_frame=torch.tensor([[-1, -1]], dtype=torch.int64),
    )
    assert not first_seen_after_impact.restitution.any()

    only_one_partner_seen, _ = _parameter_supervision_masks(
        batch,
        config,
        2,
        indices=indices,
        matched=matched,
        runtime_observed=torch.tensor([[True, False]]),
        last_observed_target_frame=torch.tensor([[0, 0]], dtype=torch.int64),
    )
    assert not only_one_partner_seen.pair_restitution.any()

    observations_both_after_impact, _ = _parameter_supervision_masks(
        batch,
        config,
        2,
        indices=indices,
        matched=matched,
        runtime_observed=torch.tensor([[True, True]]),
        last_observed_target_frame=torch.tensor([[1, 1]], dtype=torch.int64),
    )
    assert not observations_both_after_impact.restitution.any()


def test_belief_losses_do_not_infer_parameter_support_from_privileged_history() -> None:
    config = load_config("configs/tiny_overfit.yaml")
    belief = BeliefFactory(max_objects=1).create()
    belief = belief.replace(
        objects=belief.objects.replace(
            active=torch.tensor([[True]]),
            object_id=torch.tensor([[0]], dtype=torch.int64),
            velocity=torch.tensor([[[2.0, 0.0, 0.0]]]),
        )
    )
    batch = {
        "objects": {
            "active": torch.ones(1, 1, 1, dtype=torch.bool),
            "position": torch.zeros(1, 1, 1, 3),
            "velocity": torch.tensor([[[[2.0, 0.0, 0.0]]]]),
            "drag": torch.full((1, 1, 1, 1), 0.5),
            "restitution": torch.full((1, 1, 1, 1), 0.8),
        },
        # A permanent scene-history collision label is deliberately present.
        # Without the explicit runtime evidence masks it is not supervision.
        "events": {
            "collision": torch.ones(1, 1, 1, dtype=torch.bool),
        },
    }

    losses, _, matched = _belief_state_losses(
        belief,
        batch,
        config,
        0,
        indices=torch.tensor([[0]], dtype=torch.int64),
        matched=torch.tensor([[True]]),
    )

    assert matched.item()
    assert "parameter_drag" not in losses
    assert "parameter_restitution" not in losses


def test_closed_loop_distance_gates_privileged_parameter_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config("configs/tiny_overfit.yaml")
    batch = collate_episodes([generate_episode(config, seed=7)])
    model = OnlineWorldModel.from_config(config, device="cpu")
    original_masks = training_loop._parameter_supervision_masks
    received_matches: list[torch.Tensor] = []

    def reject_all_physical_matches(
        prediction: torch.Tensor,
        aligned_target: torch.Tensor,
        assignment_mask: torch.Tensor,
        **_: Any,
    ) -> torch.Tensor:
        assert prediction.shape == aligned_target.shape
        return torch.zeros_like(assignment_mask)

    def record_parameter_matches(*args: Any, **kwargs: Any) -> Any:
        received_matches.append(kwargs["matched"].detach().clone())
        return original_masks(*args, **kwargs)

    monkeypatch.setattr(
        training_loop,
        "_distance_gate_physical_matches",
        reject_all_physical_matches,
    )
    monkeypatch.setattr(
        training_loop,
        "_parameter_supervision_masks",
        record_parameter_matches,
    )

    result = run_closed_loop_batch(
        model,
        batch,
        config,
        window_steps=2,
        apply_perturbations=False,
        include_measurement_supervision=False,
        rollout_anchors_per_window=1,
    )

    assert received_matches
    assert not torch.stack(received_matches).any()
    assert result.metrics["parameter_runtime_observed_object_count"] == 0.0
    assert result.metrics["parameter_drag_observable_object_count"] == 0.0
    assert result.metrics["parameter_restitution_observable_object_count"] == 0.0
