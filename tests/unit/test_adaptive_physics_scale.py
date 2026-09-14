from __future__ import annotations

import torch

from world_model.evaluation import multicontact_six_dof as state_gate
from world_model.evaluation.adaptive_physics_scale import (
    ADAPTIVE_PHYSICS_SCALE_SCHEMA,
    _heterogeneous_truth,
    _tolerance_aligned_repeated_contact_f1,
    adaptive_physics_manifest_sha256,
)


def test_adaptive_physics_manifest_is_deterministic_and_single_model() -> None:
    first = adaptive_physics_manifest_sha256()
    second = adaptive_physics_manifest_sha256()

    assert first == second
    assert len(first) == 64
    assert ADAPTIVE_PHYSICS_SCALE_SCHEMA == "world_model_adaptive_physics_scale_v1"


def test_heterogeneous_truth_varies_parameters_without_mutating_source() -> None:
    source = state_gate._state(4)
    original_mass = source.mass.clone()

    varied = _heterogeneous_truth(source, 4)

    assert varied.mass[:, 0].unique().numel() == 4
    assert varied.drag[:, 0].unique().numel() == 4
    assert varied.restitution[:, 0].unique().numel() == 4
    assert varied.friction[:, 0].unique().numel() == 4
    torch.testing.assert_close(source.mass, original_mass)


def test_repeated_contacts_allow_only_declared_one_frame_alignment() -> None:
    model = torch.zeros(6, 2, 2, dtype=torch.bool)
    reference = torch.zeros_like(model)
    model[[1, 3], 0, 1] = True
    model[[1, 3], 1, 0] = True
    reference[[2, 4], 0, 1] = True
    reference[[2, 4], 1, 0] = True

    assert _tolerance_aligned_repeated_contact_f1(model, reference) == 1.0
    assert _tolerance_aligned_repeated_contact_f1(model, reference, tolerance_frames=0) == 0.0


def test_repeated_contact_alignment_rejects_invalid_inputs() -> None:
    trace = torch.zeros(2, 2, 2, dtype=torch.bool)
    try:
        _tolerance_aligned_repeated_contact_f1(trace, trace, tolerance_frames=-1)
    except ValueError as error:
        assert "nonnegative" in str(error)
    else:
        raise AssertionError("negative tolerance was accepted")

    try:
        _tolerance_aligned_repeated_contact_f1(trace, trace[0])
    except ValueError as error:
        assert "[T,N,N]" in str(error)
    else:
        raise AssertionError("invalid collision trace shape was accepted")
