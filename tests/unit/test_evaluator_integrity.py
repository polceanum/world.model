from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest
import torch

from world_model.belief import BeliefFactory, BeliefTrajectory
from world_model.evaluation.evaluator import (
    _canonical_sha256,
    _capture_checkpoint_snapshot,
    _recovery_persistent_support,
    _require_finite_belief,
    _require_finite_metrics,
    _require_finite_trajectory,
    _resolved_evaluation_protocol,
    _runtime_hypothesis_candidate_names,
    _runtime_hypothesis_learned_fallback_diagnostics,
    _sum_runtime_hypothesis_counts_on_host,
    _validate_runtime_hypothesis_composition_counts,
)
from world_model.evaluation.reports import write_evaluation_report
from world_model.evaluation.seed_protocol import make_evaluation_seed_protocol
from world_model.utils.config import load_config


def _trajectory(*, event_value: float = 0.0) -> BeliefTrajectory:
    return BeliefTrajectory(
        timestamps=torch.tensor([[0.1]]),
        positions=torch.zeros(1, 1, 1, 3),
        velocities=torch.zeros(1, 1, 1, 3),
        orientations=torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]]),
        motion_mode_logits=torch.zeros(1, 1, 1, 4),
        fast_log_variance=torch.zeros(1, 1, 1, 3),
        active_mask=torch.ones(1, 1, 1, dtype=torch.bool),
        event_logits=torch.full((1, 1, 1, 4), event_value),
    )


def test_runtime_hypothesis_diagnostic_counts_reduce_on_host() -> None:
    value = torch.arange(8 * 5 * 6 * 6, dtype=torch.int64).reshape(8, 5, 6, 6)
    value.requires_grad_(False)

    reduced = _sum_runtime_hypothesis_counts_on_host(value, dim=(0, 1, 2))

    assert reduced.device.type == "cpu"
    assert reduced.dtype == torch.int64
    assert torch.equal(reduced, value.sum(dim=(0, 1, 2)))
    with pytest.raises(TypeError, match="must use torch.int64"):
        _sum_runtime_hypothesis_counts_on_host(value.float(), dim=(0, 1, 2))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_runtime_hypothesis_batched_mps_diagnostic_counts_reduce_on_host() -> None:
    value = torch.ones((8, 5, 6, 6), dtype=torch.int64, device="mps")

    reduced = _sum_runtime_hypothesis_counts_on_host(value, dim=(0, 1, 2))

    assert reduced.device.type == "cpu"
    assert torch.equal(reduced, torch.full((6,), 8 * 5 * 6, dtype=torch.int64))


def test_nonfinite_belief_and_event_logits_fail_before_scoring() -> None:
    belief = BeliefFactory(max_objects=1).create()
    invalid_objects = belief.objects.clone()
    invalid_objects.position[0, 0, 0] = float("nan")
    invalid_belief = belief.replace(objects=invalid_objects)
    with pytest.raises(FloatingPointError, match=r"objects\.position"):
        _require_finite_belief(invalid_belief, context="test")

    trajectory = _trajectory(event_value=float("nan"))
    with pytest.raises(FloatingPointError, match="event_logits"):
        _require_finite_trajectory(trajectory, context="test")
    with pytest.raises(ValueError, match="event_logits"):
        trajectory.validate()


def test_final_metrics_and_json_report_reject_nonfinite_values(tmp_path) -> None:
    with pytest.raises(FloatingPointError, match="bad"):
        _require_finite_metrics({"good": 1.0, "bad": float("nan")})
    with pytest.raises(ValueError, match="Out of range float values"):
        write_evaluation_report(
            tmp_path,
            metadata={},
            metrics={"bad": float("inf")},
            limitations=[],
        )
    assert not (tmp_path / "evaluation.json").exists()


def test_recovery_support_excludes_inactive_prior_and_slot_reuse() -> None:
    base = BeliefFactory(max_objects=3).create()
    prior_objects = base.objects.clone()
    prior_objects.active[:] = torch.tensor([[True, True, False]])
    prior_objects.object_id[:] = torch.tensor([[5, 6, -1]])
    posterior_objects = base.objects.clone()
    posterior_objects.active[:] = torch.tensor([[True, True, True]])
    # Slot 1 was recycled to a new ID; slot 2 is a birth into an inactive prior.
    posterior_objects.object_id[:] = torch.tensor([[5, 9, 10]])
    prior = base.replace(objects=prior_objects)
    posterior = base.replace(objects=posterior_objects)

    support = _recovery_persistent_support(
        prior,
        posterior,
        torch.tensor([[True, True, True]]),
    )
    assert torch.equal(support, torch.tensor([[True, False, False]]))


def test_checkpoint_snapshot_is_immutable_when_source_path_changes(tmp_path) -> None:
    source = tmp_path / "last.pt"
    original = b"first immutable checkpoint payload"
    source.write_bytes(original)
    expected_hash = hashlib.sha256(original).hexdigest()

    with _capture_checkpoint_snapshot(source) as captured:
        source.write_bytes(b"new checkpoint at same mutable path")
        assert captured.snapshot_path.read_bytes() == original
        assert captured.sha256 == expected_hash
        snapshot_path = captured.snapshot_path
    assert not snapshot_path.exists()


def test_resolved_protocol_hash_binds_manifest_batch_horizons_and_intervention() -> None:
    config = load_config("configs/tiny_overfit.yaml")
    config = replace(
        config,
        evaluation=replace(config.evaluation, episodes=2, horizons_seconds=(0.1, 0.25)),
    )
    seed_protocol = make_evaluation_seed_protocol(
        name="standard",
        split="test",
        episode_count=2,
        training_validation_episodes=config.training.validation_episodes,
    )
    protocol = _resolved_evaluation_protocol(
        config,
        checkpoint_sha256="a" * 64,
        resolved_seed_protocol=seed_protocol,
        batch_size=2,
        runtime_hypothesis_pool=False,
    )
    repeated = _resolved_evaluation_protocol(
        config,
        checkpoint_sha256="a" * 64,
        resolved_seed_protocol=seed_protocol,
        batch_size=2,
        runtime_hypothesis_pool=False,
    )
    changed_batch = _resolved_evaluation_protocol(
        config,
        checkpoint_sha256="a" * 64,
        resolved_seed_protocol=seed_protocol,
        batch_size=1,
        runtime_hypothesis_pool=False,
    )
    changed_intervention = _resolved_evaluation_protocol(
        config,
        checkpoint_sha256="a" * 64,
        resolved_seed_protocol=seed_protocol,
        batch_size=2,
        runtime_hypothesis_pool=True,
    )

    assert protocol["seed_manifest"] == list(seed_protocol.manifest.seeds)
    assert protocol["horizons_observation_grid"] == ["0.100s", "0.250s"]
    assert _canonical_sha256(protocol) == _canonical_sha256(repeated)
    assert _canonical_sha256(protocol) != _canonical_sha256(changed_batch)
    assert _canonical_sha256(protocol) != _canonical_sha256(changed_intervention)


def test_runtime_hypothesis_composition_counts_are_fail_closed() -> None:
    local_shape = (1, 2, 1, 3)
    candidates = torch.zeros((*local_shape, 4), dtype=torch.int64)
    candidates[..., 0] = 1
    candidates[:, 1, ..., 0] = 0
    candidates[:, 1, ..., 1] = 2
    total = candidates.sum(dim=-1)
    fallback = torch.zeros(local_shape, dtype=torch.int64)
    fallback[:, 0] = 1
    regimes = torch.zeros((1, 2, 1, 6), dtype=torch.int64)
    regimes[..., 0] = total[..., 0]

    arguments = {
        "local_shape": local_shape,
        "candidate_count": candidates,
        "fallback_count": fallback,
        "total_count": total,
        "regime_count": regimes,
        "independent_axes": (0, 1, 2),
        "candidate_size": 4,
        "regime_size": 6,
    }
    _validate_runtime_hypothesis_composition_counts(**arguments)

    invalid_total = total.clone()
    invalid_total[..., 0] += 1
    with pytest.raises(RuntimeError, match="candidate counts must partition"):
        _validate_runtime_hypothesis_composition_counts(
            **{**arguments, "total_count": invalid_total}
        )

    invalid_fallback = fallback.clone()
    invalid_fallback[..., 0] = 2
    with pytest.raises(RuntimeError, match="fallback count must be contained"):
        _validate_runtime_hypothesis_composition_counts(
            **{**arguments, "fallback_count": invalid_fallback}
        )

    invalid_regimes = regimes.clone()
    invalid_regimes[:, 1, ..., 0] -= 1
    with pytest.raises(RuntimeError, match="regime counts must partition"):
        _validate_runtime_hypothesis_composition_counts(
            **{**arguments, "regime_count": invalid_regimes}
        )


def test_runtime_hypothesis_candidate_schema_expands_only_when_enabled() -> None:
    legacy = load_config("configs/tiny_overfit.yaml")
    enabled = load_config(
        "configs/tiny_overfit.yaml",
        overrides=[
            "model.rgb.temporal_velocity_enabled=true",
            "runtime.hypothesis_local_applicability_enabled=true",
            "runtime.hypothesis_online_acceleration_enabled=true",
        ],
    )

    assert _runtime_hypothesis_candidate_names(legacy) == (
        "learned",
        "constant_velocity",
        "damped_constant_velocity",
        "ballistic_contact",
    )
    assert _runtime_hypothesis_candidate_names(enabled) == (
        "learned",
        "constant_velocity",
        "damped_constant_velocity",
        "ballistic_contact",
        "online_local_acceleration",
    )
    local_shape = (1, 1, 1, 3)
    candidate_count = torch.zeros((*local_shape, 5), dtype=torch.int64)
    candidate_count[..., 4] = 1
    total_count = candidate_count.sum(dim=-1)
    regime_count = torch.zeros((1, 1, 1, 6), dtype=torch.int64)
    regime_count[..., 0] = total_count[..., 0]
    _validate_runtime_hypothesis_composition_counts(
        local_shape=local_shape,
        candidate_count=candidate_count,
        fallback_count=torch.zeros(local_shape, dtype=torch.int64),
        total_count=total_count,
        regime_count=regime_count,
        independent_axes=(0, 1, 2),
        candidate_size=len(_runtime_hypothesis_candidate_names(enabled)),
        regime_size=6,
    )


def test_runtime_hypothesis_learned_fallback_has_complete_local_diagnostics() -> None:
    belief = BeliefFactory(max_objects=1).create()
    trajectory = _trajectory()

    class Controller:
        @staticmethod
        def _trajectory_regime(source: object, forecast: BeliefTrajectory) -> torch.Tensor:
            assert source is belief
            return torch.full_like(forecast.active_mask, 2, dtype=torch.int64)

    indices, supported, regimes = _runtime_hypothesis_learned_fallback_diagnostics(
        Controller(),
        belief,
        trajectory,
    )

    assert indices.shape == (1, 1, 1, 3)
    assert indices.dtype == torch.int64
    assert not indices.any()
    assert supported.shape == indices.shape
    assert supported.dtype == torch.bool
    assert not supported.any()
    assert torch.equal(regimes, torch.tensor([[[2]]], dtype=torch.int64))
