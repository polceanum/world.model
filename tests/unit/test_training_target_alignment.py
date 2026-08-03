from __future__ import annotations

import torch

from world_model.belief import BeliefFactory
from world_model.training.loop import PersistentTargetMatcher


def _belief(
    positions: torch.Tensor,
    object_ids: torch.Tensor,
    active: torch.Tensor | None = None,
):
    factory = BeliefFactory(max_objects=positions.shape[1])
    base = factory.create(batch_size=positions.shape[0])
    if active is None:
        active = torch.ones_like(object_ids, dtype=torch.bool)
    return base.replace(
        objects=base.objects.replace(
            position=positions,
            object_id=object_ids,
            active=active,
        )
    )


def test_persistent_target_matcher_does_not_swap_close_contact_velocities() -> None:
    matcher = PersistentTargetMatcher()
    active = torch.tensor([[True, True]])

    initial = _belief(
        torch.tensor([[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]),
        torch.tensor([[10, 20]]),
    )
    initial_indices, _ = matcher.match(
        initial,
        torch.tensor([[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]),
        active,
    )
    torch.testing.assert_close(initial_indices, torch.tensor([[0, 1]]))

    # At contact the estimated points lie closer to the opposite identities.
    # Independent nearest-position matching would swap both velocity targets.
    contact = _belief(
        torch.tensor([[[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]]]),
        torch.tensor([[10, 20]]),
    )
    contact_indices, matched = matcher.match(
        contact,
        torch.tensor([[[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]]]),
        active,
    )

    torch.testing.assert_close(contact_indices, torch.tensor([[0, 1]]))
    assert matched.all()


def test_persistent_target_matcher_releases_target_after_track_death() -> None:
    matcher = PersistentTargetMatcher()
    matcher.match(
        _belief(
            torch.tensor([[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]),
            torch.tensor([[10, 20]]),
        ),
        torch.tensor([[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]),
        torch.tensor([[True, True]]),
    )

    replacement = _belief(
        torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]),
        torch.tensor([[30, 20]]),
    )
    indices, matched = matcher.match(
        replacement,
        torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]),
        torch.tensor([[True, True]]),
    )

    torch.testing.assert_close(indices, torch.tensor([[0, 1]]))
    assert matched.all()


def test_persistent_target_matcher_leaves_far_new_track_and_target_unmatched() -> None:
    matcher = PersistentTargetMatcher()
    belief = _belief(
        torch.tensor([[[0.1, 0.0, 0.0], [3.0, 0.0, 0.0]]]),
        torch.tensor([[10, 20]]),
    )

    indices, matched = matcher.match(
        belief,
        torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]),
        torch.tensor([[True, True]]),
    )

    torch.testing.assert_close(indices, torch.tensor([[0, -1]]))
    torch.testing.assert_close(matched, torch.tensor([[True, False]]))
    assert matcher.mappings == [{10: 0}]


def test_persistent_target_matcher_prefers_close_new_track_over_earlier_false_track() -> None:
    matcher = PersistentTargetMatcher()
    belief = _belief(
        torch.tensor([[[2.0, 0.0, 0.0], [0.1, 0.0, 0.0]]]),
        torch.tensor([[10, 20]]),
    )

    indices, matched = matcher.match(
        belief,
        torch.tensor([[[0.0, 0.0, 0.0]]]),
        torch.tensor([[True]]),
    )

    torch.testing.assert_close(indices, torch.tensor([[-1, 0]]))
    torch.testing.assert_close(matched, torch.tensor([[False, True]]))
    assert matcher.mappings == [{20: 0}]


def test_persistent_target_matcher_uses_inclusive_half_metre_bootstrap_gate() -> None:
    matcher = PersistentTargetMatcher()
    belief = _belief(
        torch.tensor([[[0.5, 0.0, 0.0]], [[0.5001, 0.0, 0.0]]]),
        torch.tensor([[10], [20]]),
    )

    indices, matched = matcher.match(
        belief,
        torch.zeros((2, 1, 3)),
        torch.tensor([[True], [True]]),
    )

    torch.testing.assert_close(indices, torch.tensor([[0], [-1]]))
    torch.testing.assert_close(matched, torch.tensor([[True], [False]]))
    assert matcher.mappings == [{10: 0}, {}]


def test_persistent_target_matcher_keeps_existing_mapping_beyond_bootstrap_gate() -> None:
    matcher = PersistentTargetMatcher()
    active = torch.tensor([[True]])
    object_ids = torch.tensor([[10]])
    target = torch.tensor([[[0.0, 0.0, 0.0]]])

    matcher.match(
        _belief(torch.tensor([[[0.0, 0.0, 0.0]]]), object_ids),
        target,
        active,
    )
    indices, matched = matcher.match(
        _belief(torch.tensor([[[2.0, 0.0, 0.0]]]), object_ids),
        target,
        active,
    )

    torch.testing.assert_close(indices, torch.tensor([[0]]))
    assert matched.all()
    assert matcher.mappings == [{10: 0}]
