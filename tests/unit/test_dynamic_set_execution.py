from __future__ import annotations

import inspect
import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import scripts.run_rgbd_dynamic_set_campaign as campaign_cli
import world_model.training.dynamic_set_execution as execution
from world_model.training.dynamic_set_campaign import DEFAULT_CAMPAIGN
from world_model.training.dynamic_set_config import OrpheusConfig
from world_model.training.dynamic_set_trainer import (
    CHECKPOINT_SCHEMA,
    dynamic_set_model_state_sha256,
)
from world_model.training.qualification_core import canonical_sha256, sha256_bytes

SOURCE = {
    "source_sha256": "d" * 64,
    "commit": "a" * 40,
    "tree": "b" * 40,
    "upstream_commit": "a" * 40,
    "clean": True,
    "published": True,
}
ARCHITECTURE_BINDING = execution._ArchitectureExecutionBinding(
    architecture_attempt_index=1,
    architecture_choice="base",
    base_config_sha256="c" * 64,
    resolved_config_sha256="c" * 64,
    prior_attempt_cumulative_seconds=0.0,
    architecture_attempt_sha256="e" * 64,
).validate()


def _payload(
    completed: int,
    *,
    rejected: int = 0,
    seed: int = execution.DEFAULT_SCHEDULE_SEED,
    weight: float | None = None,
) -> dict[str, Any]:
    model_state = {
        "weight": torch.tensor(
            [float(completed) if weight is None else weight],
            dtype=torch.float32,
        )
    }
    return {
        "schema": CHECKPOINT_SCHEMA,
        "model_state": model_state,
        "model_state_sha256": dynamic_set_model_state_sha256(model_state),
        "trainer_state": {
            "completed_updates": completed,
            "rejected_update_count": rejected,
        },
        "next_sample_state": {
            "absolute_update_index": completed,
            "schedule_seed": seed,
        },
    }


class _FakeTrainer:
    def __init__(self) -> None:
        self.completed_updates = 0
        self.rejected_update_count = 0

    def checkpoint_payload(self) -> dict[str, Any]:
        return _payload(self.completed_updates, rejected=self.rejected_update_count)

    def load_checkpoint_payload(self, payload: dict[str, Any], *, restore_rng: bool = True) -> None:
        assert restore_rng
        self.completed_updates = payload["trainer_state"]["completed_updates"]
        self.rejected_update_count = payload["trainer_state"]["rejected_update_count"]

    def run_update(self) -> None:
        self.completed_updates += 1


class _RejectOnceTrainer(_FakeTrainer):
    def run_update(self) -> None:
        if self.rejected_update_count == 0:
            self.rejected_update_count += 1
            raise RuntimeError("synthetic rejected update")
        super().run_update()


class _LoadTrackingTrainer(_FakeTrainer):
    def __init__(self) -> None:
        super().__init__()
        self.load_count = 0

    def load_checkpoint_payload(self, payload: dict[str, Any], *, restore_rng: bool = True) -> None:
        self.load_count += 1
        super().load_checkpoint_payload(payload, restore_rng=restore_rng)


class _ExactResumeTrainer(_FakeTrainer):
    def __init__(self, *, interrupt_at: int | None = None) -> None:
        super().__init__()
        self.interrupt_at = interrupt_at
        self.optimizer_marker = 0.0
        self.scheduler_marker = 0
        self.adapter_marker = 0
        self.weight = 0.0

    def checkpoint_payload(self) -> dict[str, Any]:
        payload = _payload(
            self.completed_updates,
            rejected=self.rejected_update_count,
            weight=self.weight,
        )
        payload["synthetic_exact_state"] = {
            "optimizer": self.optimizer_marker,
            "scheduler": self.scheduler_marker,
            "adapter": self.adapter_marker,
            "torch_rng": torch.get_rng_state().clone(),
        }
        return payload

    def load_checkpoint_payload(self, payload: dict[str, Any], *, restore_rng: bool = True) -> None:
        super().load_checkpoint_payload(payload, restore_rng=restore_rng)
        state = payload["synthetic_exact_state"]
        self.optimizer_marker = state["optimizer"]
        self.scheduler_marker = state["scheduler"]
        self.adapter_marker = state["adapter"]
        self.weight = float(payload["model_state"]["weight"][0])
        if restore_rng:
            torch.set_rng_state(state["torch_rng"])

    def run_update(self) -> None:
        if self.interrupt_at == self.completed_updates:
            raise KeyboardInterrupt
        sample = float(torch.rand(()))
        self.weight = self.weight * 0.75 + sample
        self.optimizer_marker += sample * sample
        self.scheduler_marker += 1
        self.adapter_marker += 6
        self.completed_updates += 1


class _FakeQualification:
    def __init__(
        self,
        root: Path,
        config_sha256: str,
        *,
        screen_wall_seconds: float = 0.0,
        attempt_index: int = 1,
        architecture_choice: str = "base",
        resolved_config: OrpheusConfig | None = None,
        prior_attempt_cumulative_seconds: float = 0.0,
        architecture_attempt_sha256: str = "e" * 64,
    ) -> None:
        self._resolved_config = resolved_config or OrpheusConfig()
        resolved_config_sha256 = canonical_sha256(self._resolved_config.to_dict())
        base_config_sha256 = (
            resolved_config_sha256
            if attempt_index == 1
            else canonical_sha256(OrpheusConfig().to_dict())
        )
        self.protocol_sha256 = "a" * 64
        self.protocol = {
            "config_sha256": config_sha256,
            "base_config_payload_sha256": base_config_sha256,
            "source": dict(SOURCE),
        }
        self.artifacts = SimpleNamespace(root=root)
        self.candidates: list[dict[str, Any]] = []
        self._screen_wall_seconds = screen_wall_seconds
        self._architecture_binding = {
            "architecture_attempt_index": attempt_index,
            "architecture_choice": architecture_choice,
            "base_config_sha256": base_config_sha256,
            "resolved_config_sha256": resolved_config_sha256,
            "prior_attempt_cumulative_seconds": prior_attempt_cumulative_seconds,
            "architecture_attempt_sha256": architecture_attempt_sha256,
        }

    def _campaign(self) -> dict[str, Any]:
        return {
            "state": "training",
            "validation_candidates": list(self.candidates),
        }

    def screen_wall_seconds(self) -> float:
        return self._screen_wall_seconds

    def architecture_execution_binding(self) -> dict[str, Any]:
        return dict(self._architecture_binding)

    def architecture_resolved_config(self) -> OrpheusConfig:
        return self._resolved_config


def _patch_repository_execution(
    monkeypatch: pytest.MonkeyPatch,
    qualification: _FakeQualification,
    *,
    trainer_factory: Any = _FakeTrainer,
    validation_interval_updates: int = 4,
    maximum_training_seconds: float = 216_000.0,
    campaign_config: object | None = None,
) -> None:
    monkeypatch.setattr(
        execution.DynamicSetQualification,
        "attach",
        lambda _path: qualification,
    )
    monkeypatch.setattr(
        execution,
        "_load_captured_config",
        lambda _contents, *, source_path: OrpheusConfig(),
    )
    monkeypatch.setattr(execution, "_authenticate_current_source", lambda _source: {})
    monkeypatch.setattr(
        execution,
        "_fresh_repository_trainer",
        lambda **_kwargs: trainer_factory(),
    )
    configured = campaign_config or replace(
        DEFAULT_CAMPAIGN,
        validation_interval_updates=validation_interval_updates,
        maximum_training_hours=(maximum_training_seconds + 1.0) / 3600.0,
        reserved_audit_hours=1.0 / 3600.0,
    )
    configured.validate()
    monkeypatch.setattr(execution, "DEFAULT_CAMPAIGN", configured)


def _timing(
    *completed_update_seconds: float,
    discarded_attempt_seconds: float = 0.0,
) -> execution._UpdateTimingEvidence:
    return execution._UpdateTimingEvidence(
        architecture_binding=ARCHITECTURE_BINDING,
        completed_update_seconds=completed_update_seconds,
        discarded_attempt_seconds=discarded_attempt_seconds,
    ).validate(completed_updates=len(completed_update_seconds))


def _candidate(
    report: execution.DynamicSetCampaignExecutionReport,
    *,
    previous: tuple[dict[str, Any], ...] = (),
    validation_wall_seconds: float = 0.25,
) -> dict[str, Any]:
    sequence = len(previous)
    previous_timing_sha256 = "0" * 64 if not previous else previous[-1]["validation_timing_sha256"]
    cumulative = float(
        sum(item["validation_wall_seconds"] for item in previous) + validation_wall_seconds
    )
    callback_binding_sha256 = "f" * 64
    timing_body = {
        "schema": execution.VALIDATION_TIMING_EVIDENCE_SCHEMA,
        "sequence": sequence,
        "completed_updates": report.completed_updates,
        "checkpoint_sha256": report.checkpoint_sha256,
        "model_state_sha256": report.model_state_sha256,
        "execution_progress_record_sha256": report.progress_record_sha256,
        "callback_binding_sha256": callback_binding_sha256,
        "validation_wall_seconds": validation_wall_seconds,
        "validation_timing_count": sequence + 1,
        "cumulative_validation_wall_seconds": cumulative,
        "previous_validation_timing_sha256": previous_timing_sha256,
    }
    candidate = {
        "sequence": sequence,
        "completed_updates": report.completed_updates,
        "checkpoint_sha256": report.checkpoint_sha256,
        "model_state_sha256": report.model_state_sha256,
        "execution_progress_record_sha256": report.progress_record_sha256,
        "callback_binding_sha256": callback_binding_sha256,
        "validation_wall_seconds": validation_wall_seconds,
        "validation_timing_count": sequence + 1,
        "cumulative_validation_wall_seconds": cumulative,
        "previous_validation_timing_sha256": previous_timing_sha256,
        "validation_timing_sha256": canonical_sha256(timing_body),
    }
    candidate["candidate_sha256"] = canonical_sha256(candidate)
    return candidate


def test_two_slot_journal_ignores_an_uncommitted_inactive_slot(tmp_path: Path) -> None:
    protocol = "a" * 64
    config = "b" * 64
    source = canonical_sha256({"source": "frozen"})
    progress, _ = execution._journal_commit(
        work_directory=tmp_path,
        payload=_payload(3),
        previous_progress=None,
        protocol_sha256=protocol,
        config_sha256=config,
        source_sha256=source,
        timing=_timing(0.5, 0.5, 0.5),
        rejected_update_count=0,
    )
    inactive = "resume_b.pt" if progress["active_resume_name"] == "resume_a.pt" else "resume_a.pt"
    (tmp_path / inactive).write_bytes(b"interrupted-write")

    restored = execution._validated_resume(
        work_directory=tmp_path,
        protocol_sha256=protocol,
        config_sha256=config,
        source_sha256=source,
        architecture_binding=ARCHITECTURE_BINDING,
    )

    assert restored is not None
    committed, payload, _ = restored
    assert committed["completed_updates"] == 3
    assert payload["trainer_state"]["completed_updates"] == 3


def test_progress_tampering_fails_closed(tmp_path: Path) -> None:
    protocol = "a" * 64
    config = "b" * 64
    source = "c" * 64
    execution._journal_commit(
        work_directory=tmp_path,
        payload=_payload(1),
        previous_progress=None,
        protocol_sha256=protocol,
        config_sha256=config,
        source_sha256=source,
        timing=_timing(1.0),
        rejected_update_count=0,
    )
    progress_path = tmp_path / "progress.json"
    value = json.loads(progress_path.read_text(encoding="utf-8"))
    value["completed_updates"] = 2
    progress_path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="progress digest"):
        execution._validated_resume(
            work_directory=tmp_path,
            protocol_sha256=protocol,
            config_sha256=config,
            source_sha256=source,
            architecture_binding=ARCHITECTURE_BINDING,
        )


def test_executor_stops_at_each_exact_validation_boundary_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qualification_root = tmp_path / "qualification"
    qualification_root.mkdir()
    config_path = tmp_path / "profile.yaml"
    config_path.write_text("profile: frozen\n", encoding="utf-8")
    qualification = _FakeQualification(
        qualification_root,
        sha256_bytes(config_path.read_bytes()),
    )
    _patch_repository_execution(monkeypatch, qualification)
    work = tmp_path / "campaign-work"

    first = execution.execute_repository_training_block(
        run_directory=qualification_root,
        work_directory=work,
        config_path=config_path,
    )
    assert first.boundary_reached
    assert first.completed_updates == 4
    assert first.updates_executed == 4
    assert first.status == "continue"
    assert first.completed_update_timing_count == 4
    assert first.architecture_attempt_index == 1
    assert first.architecture_choice == "base"
    assert first.base_config_sha256 == first.resolved_config_sha256
    assert first.prior_attempt_cumulative_seconds == 0.0
    assert first.architecture_attempt_sha256 == "e" * 64
    assert len(first.timing_evidence_sha256) == 64
    assert first.limit_hit_reason == "none"
    assert Path(first.checkpoint_path).name == "update_000004.pt"
    first_payload = execution._safe_checkpoint_payload(Path(first.checkpoint_path).read_bytes())
    first_timing = execution._timing_from_payload(
        first_payload,
        completed_updates=4,
    )
    assert len(first_timing.completed_update_seconds) == 4
    assert first_timing.sha256 == first.timing_evidence_sha256
    assert first_timing.architecture_binding.fields() == {
        "architecture_attempt_index": first.architecture_attempt_index,
        "architecture_choice": first.architecture_choice,
        "base_config_sha256": first.base_config_sha256,
        "resolved_config_sha256": first.resolved_config_sha256,
        "prior_attempt_cumulative_seconds": first.prior_attempt_cumulative_seconds,
        "architecture_attempt_sha256": first.architecture_attempt_sha256,
    }

    waiting = execution.execute_repository_training_block(
        run_directory=qualification_root,
        work_directory=work,
        config_path=config_path,
    )
    assert waiting.boundary_reached
    assert waiting.updates_executed == 0
    assert waiting.checkpoint_sha256 == first.checkpoint_sha256

    qualification.candidates.append(_candidate(first))
    reconciled_first = execution.execute_repository_training_block(
        run_directory=qualification_root,
        work_directory=work,
        config_path=config_path,
    )
    assert reconciled_first.status == "validation_timing_reconciled"
    assert reconciled_first.validation_timing_reconciled
    assert reconciled_first.completed_updates == 4
    assert reconciled_first.updates_executed == 0
    assert reconciled_first.execution_validation_timing_count == 1
    assert reconciled_first.execution_cumulative_validation_seconds == 0.25
    assert (
        reconciled_first.execution_validation_timing_sha256
        == qualification.candidates[0]["validation_timing_sha256"]
    )

    second = execution.execute_repository_training_block(
        run_directory=qualification_root,
        work_directory=work,
        config_path=config_path,
    )
    assert second.boundary_reached
    assert second.completed_updates == 8
    assert second.updates_executed == 4
    assert Path(second.checkpoint_path).name == "update_000008.pt"
    assert second.validated_candidate_count == 1
    assert second.validated_boundary_updates == 4
    assert second.validated_candidate_sha256 == qualification.candidates[0]["candidate_sha256"]
    assert second.validated_checkpoint_sha256 == first.checkpoint_sha256
    assert second.validated_model_state_sha256 == first.model_state_sha256
    assert not second.validation_timing_reconciled
    assert execution._read_progress(Path(second.progress_path))["record_sha256"] == (
        second.progress_record_sha256
    )

    repeated = execution.execute_repository_training_block(
        run_directory=qualification_root,
        work_directory=work,
        config_path=config_path,
    )
    assert repeated.updates_executed == 0
    assert repeated.execution_validation_timing_count == 1
    assert repeated.execution_cumulative_validation_seconds == 0.25
    assert repeated.timing_evidence_sha256 == second.timing_evidence_sha256

    qualification.candidates.append(
        _candidate(
            second,
            previous=tuple(qualification.candidates),
            validation_wall_seconds=0.5,
        )
    )
    reconciled_second = execution.execute_repository_training_block(
        run_directory=qualification_root,
        work_directory=work,
        config_path=config_path,
    )
    assert reconciled_second.status == "validation_timing_reconciled"
    assert reconciled_second.completed_updates == 8
    assert reconciled_second.execution_validation_timing_count == 2
    assert reconciled_second.execution_cumulative_validation_seconds == 0.75

    third = execution.execute_repository_training_block(
        run_directory=qualification_root,
        work_directory=work,
        config_path=config_path,
    )
    assert third.completed_updates == 12
    assert third.execution_validation_timing_count == 2
    assert third.execution_cumulative_validation_seconds == 0.75
    assert (
        third.execution_validation_timing_sha256
        == qualification.candidates[-1]["validation_timing_sha256"]
    )


def test_second_architecture_attempt_starts_fresh_and_carries_time_with_shared_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qualification_root = tmp_path / "qualification"
    qualification_root.mkdir()
    config_path = tmp_path / "profile.yaml"
    config_path.write_text("profile: frozen\n", encoding="utf-8")
    base = OrpheusConfig()
    widened = replace(
        base,
        model=replace(
            base.model,
            rgbd=replace(base.model.rgbd, set_feature_dim=64),
        ),
    )
    qualification = _FakeQualification(
        qualification_root,
        sha256_bytes(config_path.read_bytes()),
        screen_wall_seconds=0.5,
        attempt_index=2,
        architecture_choice="widen_perception",
        resolved_config=widened,
        prior_attempt_cumulative_seconds=0.25,
        architecture_attempt_sha256="f" * 64,
    )
    trainers: list[_LoadTrackingTrainer] = []

    def trainer_factory() -> _LoadTrackingTrainer:
        trainer = _LoadTrackingTrainer()
        trainers.append(trainer)
        return trainer

    _patch_repository_execution(
        monkeypatch,
        qualification,
        trainer_factory=trainer_factory,
        validation_interval_updates=2,
    )
    work = tmp_path / "campaign-work"
    second = execution.execute_repository_training_block(
        run_directory=qualification_root,
        work_directory=work,
        config_path=config_path,
    )
    assert second.architecture_attempt_index == 2
    assert second.architecture_choice == "widen_perception"
    assert second.resolved_config_sha256 == canonical_sha256(widened.to_dict())
    assert second.prior_attempt_cumulative_seconds == 0.25
    assert second.completed_updates == 2
    assert second.completed_update_timing_count == 2
    assert second.screen_wall_seconds == 0.5
    assert second.cumulative_training_seconds >= 0.75
    assert Path(second.progress_path).parent.name == "attempt_02"
    assert Path(second.checkpoint_path).name == "update_000002.pt"
    assert trainers[-1].load_count == 0
    assert not (work / "attempt_01").exists()
    assert (work / "training_cache").is_dir()


def test_second_attempt_rejects_discarded_main_attempt_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qualification_root = tmp_path / "qualification"
    qualification_root.mkdir()
    config_path = tmp_path / "profile.yaml"
    config_path.write_text("profile: frozen\n", encoding="utf-8")
    base = OrpheusConfig()
    widened = replace(
        base,
        model=replace(
            base.model,
            dynamics=replace(base.model.dynamics, relation_hidden_dim=64),
        ),
    )
    qualification = _FakeQualification(
        qualification_root,
        sha256_bytes(config_path.read_bytes()),
        attempt_index=2,
        architecture_choice="widen_relation",
        resolved_config=widened,
        prior_attempt_cumulative_seconds=0.5,
        architecture_attempt_sha256="f" * 64,
    )
    _patch_repository_execution(
        monkeypatch,
        qualification,
        validation_interval_updates=2,
    )
    work = tmp_path / "campaign-work"
    (work / "attempt_01").mkdir(parents=True)
    with pytest.raises(OSError, match="discarded screen attempt"):
        execution.execute_repository_training_block(
            run_directory=qualification_root,
            work_directory=work,
            config_path=config_path,
        )
    assert not (work / "attempt_02").exists()


def test_second_attempt_carry_and_new_screen_can_exhaust_budget_at_update_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qualification_root = tmp_path / "qualification"
    qualification_root.mkdir()
    config_path = tmp_path / "profile.yaml"
    config_path.write_text("profile: frozen\n", encoding="utf-8")
    base = OrpheusConfig()
    widened = replace(
        base,
        model=replace(
            base.model,
            rgbd=replace(base.model.rgbd, set_feature_dim=64),
        ),
    )
    qualification = _FakeQualification(
        qualification_root,
        sha256_bytes(config_path.read_bytes()),
        screen_wall_seconds=0.3,
        attempt_index=2,
        architecture_choice="widen_perception",
        resolved_config=widened,
        prior_attempt_cumulative_seconds=0.8,
        architecture_attempt_sha256="f" * 64,
    )
    _patch_repository_execution(
        monkeypatch,
        qualification,
        validation_interval_updates=2,
        maximum_training_seconds=1.0,
    )
    monkeypatch.setattr(execution.time, "monotonic", lambda: 0.0)
    work = tmp_path / "campaign-work"
    limited = execution.execute_repository_training_block(
        run_directory=qualification_root,
        work_directory=work,
        config_path=config_path,
    )
    assert limited.status == "limit_hit"
    assert limited.limit_hit_reason == "training_reserve_boundary"
    assert limited.completed_updates == 0
    assert limited.updates_executed == 0
    assert limited.prior_attempt_cumulative_seconds == 0.8
    assert limited.screen_wall_seconds == 0.3
    assert limited.cumulative_training_seconds == pytest.approx(1.1)


def test_validation_timing_omission_and_tampering_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qualification_root = tmp_path / "qualification"
    qualification_root.mkdir()
    config_path = tmp_path / "profile.yaml"
    config_path.write_text("profile: frozen\n", encoding="utf-8")
    qualification = _FakeQualification(
        qualification_root,
        sha256_bytes(config_path.read_bytes()),
    )
    _patch_repository_execution(monkeypatch, qualification)
    work = tmp_path / "campaign-work"
    boundary = execution.execute_repository_training_block(
        run_directory=qualification_root,
        work_directory=work,
        config_path=config_path,
    )

    tampered = _candidate(boundary)
    tampered["validation_wall_seconds"] = 99.0
    qualification.candidates.append(tampered)
    with pytest.raises(ValueError, match="validation timing chain differs"):
        execution.execute_repository_training_block(
            run_directory=qualification_root,
            work_directory=work,
            config_path=config_path,
        )

    qualification.candidates[:] = [_candidate(boundary)]
    reconciled = execution.execute_repository_training_block(
        run_directory=qualification_root,
        work_directory=work,
        config_path=config_path,
    )
    assert reconciled.execution_validation_timing_count == 1
    qualification.candidates.clear()
    with pytest.raises(ValueError, match="unsealed validation timing"):
        execution.execute_repository_training_block(
            run_directory=qualification_root,
            work_directory=work,
            config_path=config_path,
        )


def test_validation_timing_reconciliation_crash_retries_without_double_charge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qualification_root = tmp_path / "qualification"
    qualification_root.mkdir()
    config_path = tmp_path / "profile.yaml"
    config_path.write_text("profile: frozen\n", encoding="utf-8")
    qualification = _FakeQualification(
        qualification_root,
        sha256_bytes(config_path.read_bytes()),
    )
    _patch_repository_execution(monkeypatch, qualification)
    work = tmp_path / "campaign-work"
    boundary = execution.execute_repository_training_block(
        run_directory=qualification_root,
        work_directory=work,
        config_path=config_path,
    )
    qualification.candidates.append(_candidate(boundary, validation_wall_seconds=0.75))

    write_progress = execution._write_progress
    interrupted = False

    def interrupt_commit(path: Path, body: dict[str, Any]) -> dict[str, Any]:
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        return write_progress(path, body)

    monkeypatch.setattr(execution, "_write_progress", interrupt_commit)
    with pytest.raises(KeyboardInterrupt):
        execution.execute_repository_training_block(
            run_directory=qualification_root,
            work_directory=work,
            config_path=config_path,
        )
    monkeypatch.setattr(execution, "_write_progress", write_progress)

    reconciled = execution.execute_repository_training_block(
        run_directory=qualification_root,
        work_directory=work,
        config_path=config_path,
    )
    assert reconciled.status == "validation_timing_reconciled"
    assert reconciled.completed_updates == boundary.completed_updates
    assert reconciled.execution_validation_timing_count == 1
    assert reconciled.execution_cumulative_validation_seconds == 0.75
    timing = execution._timing_from_payload(
        execution._safe_checkpoint_payload(Path(reconciled.checkpoint_path).read_bytes()),
        completed_updates=boundary.completed_updates,
    )
    assert timing.completed_validation_seconds == (0.75,)


def test_last_hard_cap_validation_can_be_reconciled_without_an_extension(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qualification_root = tmp_path / "qualification"
    qualification_root.mkdir()
    config_path = tmp_path / "profile.yaml"
    config_path.write_text("profile: frozen\n", encoding="utf-8")
    qualification = _FakeQualification(
        qualification_root,
        sha256_bytes(config_path.read_bytes()),
    )
    campaign = replace(
        DEFAULT_CAMPAIGN,
        minimum_updates=4,
        extension_updates=4,
        maximum_updates=8,
        validation_interval_updates=4,
        warmup_updates=4,
        minimum_timing_support_updates=2,
        timing_projection_window_updates=2,
        screen_maximum_updates=4,
    ).validate()
    _patch_repository_execution(
        monkeypatch,
        qualification,
        campaign_config=campaign,
    )
    work = tmp_path / "campaign-work"
    first = execution.execute_repository_training_block(
        run_directory=qualification_root,
        work_directory=work,
        config_path=config_path,
    )
    qualification.candidates.append(_candidate(first))
    assert (
        execution.execute_repository_training_block(
            run_directory=qualification_root,
            work_directory=work,
            config_path=config_path,
        ).status
        == "validation_timing_reconciled"
    )
    final_boundary = execution.execute_repository_training_block(
        run_directory=qualification_root,
        work_directory=work,
        config_path=config_path,
    )
    assert final_boundary.completed_updates == 8
    qualification.candidates.append(
        _candidate(final_boundary, previous=tuple(qualification.candidates))
    )
    final_reconciliation = execution.execute_repository_training_block(
        run_directory=qualification_root,
        work_directory=work,
        config_path=config_path,
    )
    assert final_reconciliation.status == "validation_timing_reconciled"
    assert final_reconciliation.completed_updates == 8
    assert final_reconciliation.updates_executed == 0
    assert final_reconciliation.execution_validation_timing_count == 2

    idempotent = execution.execute_repository_training_block(
        run_directory=qualification_root,
        work_directory=work,
        config_path=config_path,
    )
    assert idempotent.completed_updates == 8
    assert idempotent.updates_executed == 0
    assert idempotent.execution_validation_timing_count == 2


def test_screen_and_validation_time_can_exhaust_mutation_budget_without_an_update(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qualification_root = tmp_path / "qualification"
    qualification_root.mkdir()
    config_path = tmp_path / "profile.yaml"
    config_path.write_text("profile: frozen\n", encoding="utf-8")
    screen_limited = _FakeQualification(
        qualification_root,
        sha256_bytes(config_path.read_bytes()),
        screen_wall_seconds=1.0,
    )
    _patch_repository_execution(
        monkeypatch,
        screen_limited,
        maximum_training_seconds=0.5,
    )
    report = execution.execute_repository_training_block(
        run_directory=qualification_root,
        work_directory=tmp_path / "screen-limited-work",
        config_path=config_path,
    )
    assert report.status == "limit_hit"
    assert report.limit_hit_reason == "training_reserve_boundary"
    assert report.completed_updates == 0
    assert report.screen_wall_seconds == 1.0
    assert report.cumulative_training_seconds == 1.0

    validation_root = tmp_path / "validation-qualification"
    validation_root.mkdir()
    validation_limited = _FakeQualification(
        validation_root,
        sha256_bytes(config_path.read_bytes()),
    )
    _patch_repository_execution(
        monkeypatch,
        validation_limited,
        validation_interval_updates=2,
        maximum_training_seconds=1.0,
    )
    ticks = iter((0.0, 0.0, 0.2, 0.2, 0.4, 0.4, 0.4, 0.4))
    monkeypatch.setattr(execution.time, "monotonic", lambda: next(ticks))
    boundary = execution.execute_repository_training_block(
        run_directory=validation_root,
        work_directory=tmp_path / "validation-limited-work",
        config_path=config_path,
    )
    validation_limited.candidates.append(_candidate(boundary, validation_wall_seconds=0.7))
    limited = execution.execute_repository_training_block(
        run_directory=validation_root,
        work_directory=tmp_path / "validation-limited-work",
        config_path=config_path,
    )
    assert limited.status == "limit_hit"
    assert limited.limit_hit_reason == "training_reserve_boundary"
    assert limited.completed_updates == 2
    assert limited.updates_executed == 0
    assert limited.execution_validation_timing_count == 1
    assert limited.execution_cumulative_validation_seconds == 0.7
    assert limited.cumulative_training_seconds == pytest.approx(1.1)


def test_work_directory_cannot_overlap_sealed_qualification(tmp_path: Path) -> None:
    qualification = tmp_path / "qualification"
    qualification.mkdir()
    with pytest.raises(ValueError, match="must not overlap"):
        execution._assert_dedicated_work_directory(
            qualification / "training-work",
            qualification,
        )


@pytest.mark.parametrize("relation", ["equal", "ancestor", "descendant"])
def test_all_work_and_qualification_overlap_directions_fail(tmp_path: Path, relation: str) -> None:
    qualification = tmp_path / "qualification"
    qualification.mkdir()
    if relation == "equal":
        work = qualification
    elif relation == "ancestor":
        work = tmp_path
    else:
        work = qualification / "work"
    with pytest.raises(ValueError, match="must not overlap"):
        execution._assert_dedicated_work_directory(work, qualification)


def test_work_inventory_rejects_linked_artifacts(tmp_path: Path) -> None:
    qualification = tmp_path / "qualification"
    qualification.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    outside = tmp_path / "outside.lock"
    (work / execution._LOCK_NAME).symlink_to(outside)

    with pytest.raises(OSError, match="single-link regular file"):
        execution._assert_dedicated_work_directory(work, qualification)
    assert not outside.exists()


def test_execution_lock_is_exclusive_and_reusable(tmp_path: Path) -> None:
    first = execution._lock_execution(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="another dynamic-set campaign executor"):
            execution._lock_execution(tmp_path)
    finally:
        first.close()
    second = execution._lock_execution(tmp_path)
    second.close()


def test_stable_reader_rejects_a_hard_link(tmp_path: Path) -> None:
    original = tmp_path / "checkpoint.pt"
    linked = tmp_path / "linked.pt"
    original.write_bytes(b"checkpoint")
    os.link(original, linked)

    with pytest.raises(OSError, match="single-link regular file"):
        execution._stable_regular_bytes(linked)


def test_captured_config_loader_parses_the_already_bound_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = tmp_path / "profile.yaml"
    original.write_bytes(b"profile: original\n")
    captured = original.read_bytes()
    sentinel = OrpheusConfig()

    def load_captured(path: Path) -> object:
        original.write_bytes(b"profile: replaced\n")
        assert Path(path) != original
        assert Path(path).read_bytes() == captured
        return sentinel

    monkeypatch.setattr(execution, "load_config", load_captured)
    loaded = execution._load_captured_config(captured, source_path=original)
    assert loaded.source_path == str(original)
    assert loaded.project == sentinel.project


def test_source_authentication_requires_exact_clean_head_tree_and_upstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = {
        "HEAD": SOURCE["commit"],
        "HEAD^{tree}": SOURCE["tree"],
        "@{upstream}": SOURCE["upstream_commit"],
    }

    def git_bytes(_root: Path, arguments: tuple[str, ...], *, label: str) -> bytes:
        del label
        if arguments[0] == "status":
            return b""
        if arguments[0] == "rev-list":
            return b"0\t0\n"
        return (outputs[arguments[-1]] + "\n").encode("ascii")

    monkeypatch.setattr(execution, "_git_bytes", git_bytes)
    assert execution._authenticate_current_source(SOURCE)["tree"] == SOURCE["tree"]

    outputs["HEAD^{tree}"] = "f" * 40
    with pytest.raises(PermissionError, match="differs from"):
        execution._authenticate_current_source(SOURCE)


def test_checkpoint_schedule_seed_is_fixed() -> None:
    with pytest.raises(ValueError, match="schedule seed differs"):
        execution._checkpoint_cursor(_payload(0, seed=999))


def test_governed_execution_rejects_sparse_checkpoint_intervals() -> None:
    with pytest.raises(ValueError, match="per-update checkpoints"):
        execution.execute_repository_training_block(
            run_directory="unused",
            work_directory="unused",
            config_path="unused",
            checkpoint_interval_updates=2,
        )


def test_continuation_rejects_a_resume_not_matching_the_validated_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qualification_root = tmp_path / "qualification"
    qualification_root.mkdir()
    config_path = tmp_path / "profile.yaml"
    config_path.write_text("profile: frozen\n", encoding="utf-8")
    qualification = _FakeQualification(
        qualification_root,
        sha256_bytes(config_path.read_bytes()),
    )
    _patch_repository_execution(monkeypatch, qualification)
    work = tmp_path / "campaign-work"
    boundary = execution.execute_repository_training_block(
        run_directory=qualification_root,
        work_directory=work,
        config_path=config_path,
    )
    qualification.candidates.append(_candidate(boundary))

    source_sha = canonical_sha256(SOURCE)
    attempt_work = work / "attempt_01"
    architecture_binding = execution._ArchitectureExecutionBinding.from_qualification(
        qualification.architecture_execution_binding()
    )
    resumed = execution._validated_resume(
        work_directory=attempt_work,
        protocol_sha256=qualification.protocol_sha256,
        config_sha256=qualification.protocol["config_sha256"],
        source_sha256=source_sha,
        architecture_binding=architecture_binding,
    )
    assert resumed is not None
    progress, resume_payload, _ = resumed
    divergent = _payload(boundary.completed_updates, weight=999.0)
    execution._journal_commit(
        work_directory=attempt_work,
        payload=divergent,
        previous_progress=progress,
        protocol_sha256=qualification.protocol_sha256,
        config_sha256=qualification.protocol["config_sha256"],
        source_sha256=source_sha,
        timing=execution._timing_from_payload(
            resume_payload,
            completed_updates=boundary.completed_updates,
        ),
        rejected_update_count=0,
    )

    with pytest.raises(ValueError, match="not descended"):
        execution.execute_repository_training_block(
            run_directory=qualification_root,
            work_directory=work,
            config_path=config_path,
        )


def test_rejected_attempt_is_durably_resumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qualification_root = tmp_path / "qualification"
    qualification_root.mkdir()
    config_path = tmp_path / "profile.yaml"
    config_path.write_text("profile: frozen\n", encoding="utf-8")
    qualification = _FakeQualification(
        qualification_root,
        sha256_bytes(config_path.read_bytes()),
    )
    _patch_repository_execution(
        monkeypatch,
        qualification,
        trainer_factory=_RejectOnceTrainer,
        validation_interval_updates=2,
    )
    work = tmp_path / "campaign-work"

    with pytest.raises(RuntimeError, match="synthetic rejected update"):
        execution.execute_repository_training_block(
            run_directory=qualification_root,
            work_directory=work,
            config_path=config_path,
        )
    progress = execution._read_progress(work / "attempt_01" / execution._PROGRESS_NAME)
    assert progress["completed_updates"] == 0
    assert progress["rejected_update_count"] == 1
    assert progress["discarded_attempt_seconds"] >= 0.0

    report = execution.execute_repository_training_block(
        run_directory=qualification_root,
        work_directory=work,
        config_path=config_path,
    )
    assert report.boundary_reached
    assert report.completed_updates == 2
    assert report.rejected_update_count == 1


def test_interrupt_resume_matches_uninterrupted_rng_and_mutable_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qualification_root = tmp_path / "qualification"
    qualification_root.mkdir()
    config_path = tmp_path / "profile.yaml"
    config_path.write_text("profile: frozen\n", encoding="utf-8")
    qualification = _FakeQualification(
        qualification_root,
        sha256_bytes(config_path.read_bytes()),
    )
    factories = iter(
        (
            lambda: _ExactResumeTrainer(interrupt_at=2),
            lambda: _ExactResumeTrainer(),
        )
    )
    _patch_repository_execution(
        monkeypatch,
        qualification,
        trainer_factory=lambda: next(factories)(),
    )
    work = tmp_path / "campaign-work"
    with pytest.raises(KeyboardInterrupt):
        execution.execute_repository_training_block(
            run_directory=qualification_root,
            work_directory=work,
            config_path=config_path,
        )
    resumed = execution.execute_repository_training_block(
        run_directory=qualification_root,
        work_directory=work,
        config_path=config_path,
    )
    actual = execution._safe_checkpoint_payload(Path(resumed.checkpoint_path).read_bytes())

    execution._seed_fresh_process(execution.DEFAULT_SCHEDULE_SEED)
    uninterrupted = _ExactResumeTrainer()
    for _ in range(4):
        uninterrupted.run_update()
    expected = uninterrupted.checkpoint_payload()
    assert (
        actual["synthetic_exact_state"]["optimizer"]
        == expected["synthetic_exact_state"]["optimizer"]
    )
    assert actual["synthetic_exact_state"]["scheduler"] == 4
    assert actual["synthetic_exact_state"]["adapter"] == 24
    torch.testing.assert_close(
        actual["model_state"]["weight"], expected["model_state"]["weight"], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        actual["synthetic_exact_state"]["torch_rng"],
        expected["synthetic_exact_state"]["torch_rng"],
        rtol=0.0,
        atol=0.0,
    )
    timing = execution._timing_from_payload(actual, completed_updates=4)
    assert len(timing.completed_update_seconds) == 4
    assert resumed.timing_evidence_sha256 == timing.sha256


def test_update_crossing_time_limit_is_rolled_back_and_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qualification_root = tmp_path / "qualification"
    qualification_root.mkdir()
    config_path = tmp_path / "profile.yaml"
    config_path.write_text("profile: frozen\n", encoding="utf-8")
    qualification = _FakeQualification(
        qualification_root,
        sha256_bytes(config_path.read_bytes()),
    )
    _patch_repository_execution(
        monkeypatch,
        qualification,
        maximum_training_seconds=1.0,
    )
    ticks = iter((0.0, 0.0, 0.4, 0.4, 1.2, 1.2))
    monkeypatch.setattr(execution.time, "monotonic", lambda: next(ticks))

    report = execution.execute_repository_training_block(
        run_directory=qualification_root,
        work_directory=tmp_path / "campaign-work",
        config_path=config_path,
    )
    assert report.stopped_for_time_limit
    assert not report.boundary_reached
    assert report.completed_updates == 1
    assert report.updates_executed == 1
    assert report.cumulative_training_seconds == pytest.approx(1.2)
    assert report.status == "limit_hit"
    assert report.limit_hit_reason == "training_reserve_boundary"
    assert report.completed_update_timing_count == 1
    assert report.discarded_attempt_seconds == pytest.approx(0.8)
    progress = execution._read_progress(
        tmp_path / "campaign-work" / "attempt_01" / execution._PROGRESS_NAME
    )
    assert progress["training_limit_reached"] is True
    assert progress["completed_updates"] == 1


def test_update_ending_exactly_at_reserve_boundary_is_not_rolled_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qualification_root = tmp_path / "qualification"
    qualification_root.mkdir()
    config_path = tmp_path / "profile.yaml"
    config_path.write_text("profile: frozen\n", encoding="utf-8")
    qualification = _FakeQualification(
        qualification_root,
        sha256_bytes(config_path.read_bytes()),
    )
    _patch_repository_execution(
        monkeypatch,
        qualification,
        maximum_training_seconds=1.0,
    )
    ticks = iter((0.0, 0.0, 0.25, 0.25, 1.0, 1.0))
    monkeypatch.setattr(execution.time, "monotonic", lambda: next(ticks))

    report = execution.execute_repository_training_block(
        run_directory=qualification_root,
        work_directory=tmp_path / "campaign-work",
        config_path=config_path,
    )
    assert report.status == "limit_hit"
    assert report.completed_updates == 2
    assert report.updates_executed == 2
    assert report.completed_update_timing_count == 2
    assert report.discarded_attempt_seconds == 0.0
    assert report.cumulative_training_seconds == 1.0


def test_measured_projection_stops_after_support_without_rolling_back_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qualification_root = tmp_path / "qualification"
    qualification_root.mkdir()
    config_path = tmp_path / "profile.yaml"
    config_path.write_text("profile: frozen\n", encoding="utf-8")
    qualification = _FakeQualification(
        qualification_root,
        sha256_bytes(config_path.read_bytes()),
    )
    campaign = replace(
        DEFAULT_CAMPAIGN,
        minimum_updates=8,
        extension_updates=4,
        maximum_updates=12,
        validation_interval_updates=4,
        warmup_updates=4,
        maximum_training_hours=11.0 / 3600.0,
        reserved_audit_hours=1.0 / 3600.0,
        minimum_timing_support_updates=6,
        timing_projection_window_updates=2,
        timing_projection_confidence_z=1.0,
        timing_projection_safety_factor=1.10,
        screen_maximum_updates=4,
    ).validate()
    _patch_repository_execution(
        monkeypatch,
        qualification,
        campaign_config=campaign,
    )
    ticks = iter(
        (
            0.0,
            0.0,
            1.0,
            1.0,
            2.0,
            2.0,
            3.0,
            3.0,
            4.0,
            4.0,
            4.0,
            4.0,
            5.0,
            5.0,
            6.0,
            6.0,
            7.0,
            7.0,
        )
    )
    monkeypatch.setattr(execution.time, "monotonic", lambda: next(ticks))

    boundary = execution.execute_repository_training_block(
        run_directory=qualification_root,
        work_directory=tmp_path / "campaign-work",
        config_path=config_path,
    )
    assert boundary.completed_updates == 4
    qualification.candidates.append(_candidate(boundary, validation_wall_seconds=1.0))
    reconciled = execution.execute_repository_training_block(
        run_directory=qualification_root,
        work_directory=tmp_path / "campaign-work",
        config_path=config_path,
    )
    assert reconciled.status == "validation_timing_reconciled"
    assert reconciled.completed_updates == 4
    report = execution.execute_repository_training_block(
        run_directory=qualification_root,
        work_directory=tmp_path / "campaign-work",
        config_path=config_path,
    )
    assert report.status == "limit_hit"
    assert report.limit_hit_reason == "minimum_update_projection_infeasible"
    assert report.completed_updates == 6
    assert report.updates_executed == 2
    assert report.projection_support_satisfied
    assert report.minimum_update_feasible is False
    assert report.execution_validation_timing_count == 1
    assert report.execution_cumulative_validation_seconds == 1.0
    assert report.conservative_update_seconds == pytest.approx(1.1)
    assert report.conservative_validation_seconds == pytest.approx(1.1)
    assert report.projected_remaining_validation_seconds == pytest.approx(1.1)
    assert report.projected_minimum_training_seconds == pytest.approx(10.3)
    assert report.projected_minimum_envelope_seconds == pytest.approx(11.3)
    assert report.campaign_envelope_limit_seconds == pytest.approx(11.0)
    assert report.training_mutation_limit_seconds == pytest.approx(10.0)
    assert report.discarded_attempt_seconds == 0.0


def test_checkpoint_timing_tampering_fails_after_outer_digest_is_rebound(
    tmp_path: Path,
) -> None:
    progress, _ = execution._journal_commit(
        work_directory=tmp_path,
        payload=_payload(2),
        previous_progress=None,
        protocol_sha256="a" * 64,
        config_sha256="b" * 64,
        source_sha256="c" * 64,
        timing=_timing(0.4, 0.6),
        rejected_update_count=0,
    )
    checkpoint_path = tmp_path / progress["active_resume_name"]
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    payload["execution_timing"]["completed_update_seconds"][0] = 0.5
    contents = execution._checkpoint_bytes(payload)
    checkpoint_path.write_bytes(contents)
    progress_body = {key: value for key, value in progress.items() if key != "record_sha256"}
    progress_body["checkpoint_sha256"] = sha256_bytes(contents)
    execution._write_progress(tmp_path / execution._PROGRESS_NAME, progress_body)

    with pytest.raises(ValueError, match="timing binding"):
        execution._validated_resume(
            work_directory=tmp_path,
            protocol_sha256="a" * 64,
            config_sha256="b" * 64,
            source_sha256="c" * 64,
            architecture_binding=ARCHITECTURE_BINDING,
        )


def test_journal_rejects_rewriting_already_committed_update_timings(
    tmp_path: Path,
) -> None:
    progress, _ = execution._journal_commit(
        work_directory=tmp_path,
        payload=_payload(2),
        previous_progress=None,
        protocol_sha256="a" * 64,
        config_sha256="b" * 64,
        source_sha256="c" * 64,
        timing=_timing(0.4, 0.6),
        rejected_update_count=0,
    )
    with pytest.raises(ValueError, match="append-only continuation"):
        execution._journal_commit(
            work_directory=tmp_path,
            payload=_payload(2),
            previous_progress=progress,
            protocol_sha256="a" * 64,
            config_sha256="b" * 64,
            source_sha256="c" * 64,
            timing=_timing(0.5, 0.5),
            rejected_update_count=0,
        )


def test_update_timing_evidence_is_finite_and_campaign_bounded() -> None:
    with pytest.raises(ValueError, match="campaign bound"):
        execution._UpdateTimingEvidence(
            architecture_binding=ARCHITECTURE_BINDING,
            completed_update_seconds=(0.0,) * (DEFAULT_CAMPAIGN.maximum_updates + 1),
        ).validate()
    with pytest.raises(ValueError, match="completed-update timing"):
        _timing(float("nan"))
    with pytest.raises(ValueError, match="checkpoint cursor"):
        _timing(0.1).validate(completed_updates=2)


def test_elapsed_time_cannot_be_supplied_to_journal_or_cli() -> None:
    assert (
        "cumulative_training_seconds" not in inspect.signature(execution._journal_commit).parameters
    )
    with pytest.raises(SystemExit):
        campaign_cli.arguments(
            [
                "--run-directory",
                "run",
                "--work-directory",
                "work",
                "--config",
                "config",
                "--elapsed-training-hours",
                "1",
            ]
        )


def test_cli_returns_nonzero_when_training_limit_is_reached(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    report = SimpleNamespace(
        status="limit_hit",
        stopped_for_time_limit=True,
        to_dict=lambda: {"schema": execution.EXECUTION_SCHEMA, "stopped_for_time_limit": True},
    )
    monkeypatch.setattr(
        campaign_cli,
        "execute_repository_training_block",
        lambda **_kwargs: report,
    )
    result = campaign_cli.main(
        ["--run-directory", "run", "--work-directory", "work", "--config", "config"]
    )
    assert result == 2
    assert json.loads(capsys.readouterr().out)["stopped_for_time_limit"] is True
