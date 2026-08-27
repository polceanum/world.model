from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest
import torch

from world_model.belief import BeliefFactory, BeliefTrajectory
from world_model.evaluation.evaluator import (
    _canonical_sha256,
    _capture_checkpoint_snapshot,
    _record_temporal_velocity_measurements,
    _recovery_persistent_support,
    _require_finite_belief,
    _require_finite_metrics,
    _require_finite_trajectory,
    _resolved_evaluation_protocol,
    _temporal_velocity_measurement_metric_source,
    evaluate_checkpoint,
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


class _TemporalMeasurementRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def update(self, value: object) -> None:
        self.calls.append(("measurement", value))

    def update_direct(self, value: object) -> None:
        self.calls.append(("direct", value))


def test_evaluator_records_one_modality_owned_temporal_velocity_representation() -> None:
    measurements = object()
    direct = object()
    rgb = _TemporalMeasurementRecorder()
    _record_temporal_velocity_measurements(
        rgb,  # type: ignore[arg-type]
        modality="rgb",
        measurements=measurements,  # type: ignore[arg-type]
        direct_evidence=direct,  # type: ignore[arg-type]
    )
    assert rgb.calls == [("measurement", measurements)]

    rgbd = _TemporalMeasurementRecorder()
    _record_temporal_velocity_measurements(
        rgbd,  # type: ignore[arg-type]
        modality="rgbd",
        measurements=measurements,  # type: ignore[arg-type]
        direct_evidence=direct,  # type: ignore[arg-type]
    )
    assert rgbd.calls == [("measurement", measurements), ("direct", direct)]
    assert _temporal_velocity_measurement_metric_source("rgb") == (
        "fresh_runtime_last_measurements_explicit_auxiliary_fields_only"
    )
    assert _temporal_velocity_measurement_metric_source("rgbd") == (
        "fresh_runtime_last_direct_velocity_evidence_after_association"
    )


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


def test_rgbd_resolved_protocol_has_distinct_truthful_schema_and_warmup() -> None:
    config = load_config("configs/rgbd_online_free_motion_cpu.yaml")
    config = replace(
        config,
        evaluation=replace(config.evaluation, episodes=2),
    )
    seed_protocol = make_evaluation_seed_protocol(
        name="standard",
        split="test",
        episode_count=2,
        training_validation_episodes=config.training.validation_episodes,
    )

    protocol = _resolved_evaluation_protocol(
        config,
        checkpoint_sha256="b" * 64,
        resolved_seed_protocol=seed_protocol,
        batch_size=2,
        runtime_hypothesis_pool=False,
    )

    assert protocol["schema_version"] == "held_out_rgbd_online_v1"
    assert protocol["metric_schema_version"] == "held_out_rgbd_metrics_v1"
    assert protocol["observation_modality"] == "rgbd"
    assert protocol["rgb_only"] is False
    assert protocol["temporal_warmup_frames"] == 15


def test_rgbd_evaluator_rejects_rgb_only_hypothesis_pool_before_checkpoint_access(
    tmp_path,
) -> None:
    config = load_config("configs/rgbd_online_free_motion_cpu.yaml")

    with pytest.raises(ValueError, match="supports only RGB evaluation"):
        evaluate_checkpoint(
            config,
            tmp_path / "missing.pt",
            runtime_hypothesis_pool=True,
        )
