from __future__ import annotations

import json

import pytest

import train
from world_model.training.trainer import InteractionGradientRetentionError


def test_training_lock_rejects_concurrent_owner(tmp_path) -> None:
    path = tmp_path / ".training.lock"
    first = train._acquire_training_lock(path)
    try:
        with pytest.raises(RuntimeError, match="already owns"):
            train._acquire_training_lock(path)
    finally:
        first.close()
    second = train._acquire_training_lock(path)
    second.close()


def test_fresh_cli_failure_persists_terminal_diagnostic(tmp_path, monkeypatch) -> None:
    def fail_training(*_args, **_kwargs):
        raise RuntimeError("deliberate initialization failure")

    monkeypatch.setattr(
        "world_model.training.trainer.train_from_config",
        fail_training,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--config",
            "configs/tiny_overfit.yaml",
            "--run-name",
            "failure-state",
            "--set",
            f"project.output_dir={tmp_path / 'runs'}",
        ],
    )

    with pytest.raises(RuntimeError, match="deliberate initialization failure"):
        train.main()

    run_directories = list((tmp_path / "runs").glob("*-failure-state"))
    assert len(run_directories) == 1
    state = json.loads((run_directories[0] / "training_state.json").read_text(encoding="utf-8"))
    failure = json.loads((run_directories[0] / "training_failure.json").read_text(encoding="utf-8"))
    assert state == failure
    assert failure["state"] == "failed"
    assert failure["exception_type"] == "RuntimeError"
    assert failure["message"] == "deliberate initialization failure"
    assert "RuntimeError: deliberate initialization failure" in failure["traceback"]
    history = [
        json.loads(line)
        for line in (run_directories[0] / "training_failures.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert history == [failure]


def test_cli_failure_persists_structured_optimizer_diagnostics(tmp_path, monkeypatch) -> None:
    def fail_training(*_args, **_kwargs):
        raise InteractionGradientRetentionError(
            "complete interaction gradient was starved",
            {
                "optimizer_step_attempted": 60.0,
                "interaction_gradient_clip_coefficient": 0.085,
                "episode_seeds": "1,2,3,4,5,6,7,8",
            },
        )

    monkeypatch.setattr(
        "world_model.training.trainer.train_from_config",
        fail_training,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--config",
            "configs/tiny_overfit.yaml",
            "--run-name",
            "structured-failure-state",
            "--set",
            f"project.output_dir={tmp_path / 'runs'}",
        ],
    )

    with pytest.raises(
        InteractionGradientRetentionError,
        match="complete interaction gradient was starved",
    ):
        train.main()

    run_directory = next((tmp_path / "runs").glob("*-structured-failure-state"))
    failure = json.loads((run_directory / "training_failure.json").read_text())
    assert failure["diagnostics"] == {
        "optimizer_step_attempted": 60.0,
        "interaction_gradient_clip_coefficient": 0.085,
        "episode_seeds": "1,2,3,4,5,6,7,8",
    }


def test_fresh_cli_retry_cannot_overwrite_early_failure_directory(
    tmp_path,
    monkeypatch,
) -> None:
    def fail_training(*_args, **_kwargs):
        raise RuntimeError("first failure remains authoritative")

    monkeypatch.setattr(
        "world_model.training.trainer.train_from_config",
        fail_training,
    )
    arguments = [
        "train.py",
        "--config",
        "configs/tiny_overfit.yaml",
        "--run-name",
        "exclusive-failure",
        "--set",
        f"project.output_dir={tmp_path / 'runs'}",
    ]
    monkeypatch.setattr("sys.argv", arguments)
    with pytest.raises(RuntimeError, match="first failure remains authoritative"):
        train.main()

    run_directory = next((tmp_path / "runs").glob("*-exclusive-failure"))
    original_state = (run_directory / "training_state.json").read_bytes()
    original_failure = (run_directory / "training_failure.json").read_bytes()
    original_history = (run_directory / "training_failures.jsonl").read_bytes()
    arguments[arguments.index("exclusive-failure")] = run_directory.name
    monkeypatch.setattr("sys.argv", arguments)

    with pytest.raises(FileExistsError):
        train.main()

    assert (run_directory / "training_state.json").read_bytes() == original_state
    assert (run_directory / "training_failure.json").read_bytes() == original_failure
    assert (run_directory / "training_failures.jsonl").read_bytes() == original_history


def test_in_place_resume_failure_replaces_stale_completed_state(
    tmp_path,
    monkeypatch,
) -> None:
    run_directory = tmp_path / "resume-run"
    checkpoint = run_directory / "checkpoints" / "last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"trainer is monkeypatched before reading this")
    prior_state = {
        "state": "completed",
        "updated_utc": "2026-08-03T00:00:00+00:00",
        "run_directory": str(run_directory),
        "completed_steps": 12,
        "best_checkpoint": str(checkpoint),
        "best_checkpoint_kind": "last",
    }
    (run_directory / "training_state.json").write_text(
        json.dumps(prior_state),
        encoding="utf-8",
    )

    def fail_resume(*_args, **_kwargs):
        raise RuntimeError("resume extension failed")

    monkeypatch.setattr(
        "world_model.training.trainer.train_from_config",
        fail_resume,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--config",
            "configs/tiny_overfit.yaml",
            "--resume",
            str(checkpoint),
        ],
    )

    with pytest.raises(RuntimeError, match="resume extension failed"):
        train.main()

    state = json.loads((run_directory / "training_state.json").read_text(encoding="utf-8"))
    failure = json.loads((run_directory / "training_failure.json").read_text(encoding="utf-8"))
    assert state == failure
    assert state["state"] == "failed"
    assert state["resume"] == str(checkpoint)
    assert state["previous_state"]["state"] == "completed"
    assert state["previous_state"]["completed_steps"] == 12
    history = [
        json.loads(line)
        for line in (run_directory / "training_failures.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert history == [failure]


def test_successful_resume_clears_current_failure_but_retains_history(
    tmp_path,
    monkeypatch,
) -> None:
    run_directory = tmp_path / "resume-run"
    checkpoint = run_directory / "checkpoints" / "last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"trainer is monkeypatched before reading this")
    old_failure = {
        "state": "failed",
        "updated_utc": "2026-08-03T00:00:00+00:00",
        "run_directory": str(run_directory),
        "exception_type": "RuntimeError",
        "message": "old failure",
    }
    (run_directory / "training_state.json").write_text(
        json.dumps(old_failure),
        encoding="utf-8",
    )
    (run_directory / "training_failure.json").write_text(
        json.dumps(old_failure),
        encoding="utf-8",
    )
    (run_directory / "training_failures.jsonl").write_text(
        json.dumps(old_failure) + "\n",
        encoding="utf-8",
    )

    def complete_resume(*_args, **_kwargs):
        assert not (run_directory / "training_failure.json").exists()
        live_state = json.loads((run_directory / "training_state.json").read_text(encoding="utf-8"))
        assert live_state["state"] == "running"
        assert live_state["previous_state"]["state"] == "failed"
        return {
            "run_directory": str(run_directory),
            "completed_steps": 16,
            "best_checkpoint": str(checkpoint),
            "best_checkpoint_kind": "best_rollout",
        }

    monkeypatch.setattr(
        "world_model.training.trainer.train_from_config",
        complete_resume,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--config",
            "configs/tiny_overfit.yaml",
            "--resume",
            str(checkpoint),
        ],
    )

    assert train.main() == 0

    state = json.loads((run_directory / "training_state.json").read_text(encoding="utf-8"))
    assert state["state"] == "completed"
    assert state["completed_steps"] == 16
    assert state["previous_state"]["state"] == "failed"
    assert not (run_directory / "training_failure.json").exists()
    history = [
        json.loads(line)
        for line in (run_directory / "training_failures.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert history == [old_failure]


def test_keyboard_interrupt_is_recorded_as_terminal_failure(
    tmp_path,
    monkeypatch,
) -> None:
    def interrupt_training(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(
        "world_model.training.trainer.train_from_config",
        interrupt_training,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--config",
            "configs/tiny_overfit.yaml",
            "--run-name",
            "interrupted",
            "--set",
            f"project.output_dir={tmp_path / 'runs'}",
        ],
    )

    with pytest.raises(KeyboardInterrupt):
        train.main()

    run_directory = next((tmp_path / "runs").glob("*-interrupted"))
    state = json.loads((run_directory / "training_state.json").read_text(encoding="utf-8"))
    assert state["state"] == "failed"
    assert state["exception_type"] == "KeyboardInterrupt"
    assert "KeyboardInterrupt" in state["traceback"]
    assert (run_directory / "training_failure.json").is_file()
