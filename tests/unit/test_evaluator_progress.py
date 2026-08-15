from __future__ import annotations

import json

import pytest

from world_model.evaluation import evaluator


def test_evaluate_checkpoint_persists_terminal_failure_progress(
    tmp_path,
    monkeypatch,
) -> None:
    progress_path = tmp_path / "evaluation_progress.json"
    callback_events: list[dict[str, object]] = []

    def fail_after_progress(*_args, progress_sink, **_kwargs):
        progress_sink.last_event = {
            "stage": "anchor_complete",
            "updated_utc": "2026-08-15T12:00:00+00:00",
            "pid": 123,
            "split": "test",
            "device": "mps",
            "episodes": 4,
            "batch": 1,
            "batches": 2,
            "frame": 8,
            "total_frames": 40,
        }
        raise RuntimeError("forced evaluator failure")

    monkeypatch.setattr(evaluator, "_evaluate_checkpoint_impl", fail_after_progress)

    with pytest.raises(RuntimeError, match="forced evaluator failure"):
        evaluator.evaluate_checkpoint(
            object(),
            tmp_path / "checkpoint.pt",
            progress_callback=callback_events.append,
            progress_path=progress_path,
        )

    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["stage"] == "failed"
    assert progress["exception_type"] == "RuntimeError"
    assert progress["message"] == "forced evaluator failure"
    assert progress["last_stage"] == "anchor_complete"
    assert progress["split"] == "test"
    assert progress["device"] == "mps"
    assert progress["batch"] == 1
    assert progress["total_frames"] == 40
    assert callback_events[0]["stage"] == "initializing"
    assert callback_events[-1] == progress


def test_default_progress_path_captures_prestart_failure(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "run" / "checkpoints" / "last.pt"

    def fail_before_start(*_args, **_kwargs):
        raise ValueError("invalid checkpoint before start")

    monkeypatch.setattr(evaluator, "_evaluate_checkpoint_impl", fail_before_start)

    with pytest.raises(ValueError, match="invalid checkpoint before start"):
        evaluator.evaluate_checkpoint(object(), checkpoint)

    progress_paths = list((tmp_path / "run" / "evaluation").glob("*-test/evaluation_progress.json"))
    assert len(progress_paths) == 1
    progress = json.loads(progress_paths[0].read_text(encoding="utf-8"))
    assert progress["stage"] == "failed"
    assert progress["last_stage"] == "initializing"
    assert progress["exception_type"] == "ValueError"
    assert progress["message"] == "invalid checkpoint before start"
    assert progress["checkpoint"] == str(checkpoint.resolve())
    assert progress["output_directory"] == str(progress_paths[0].parent.resolve())


def test_initial_progress_callback_failure_is_persisted(tmp_path, monkeypatch) -> None:
    progress_path = tmp_path / "evaluation_progress.json"

    def should_not_start(*_args, **_kwargs):
        pytest.fail("evaluation implementation must not start after callback failure")

    def broken_callback(_event):
        raise BrokenPipeError("progress consumer closed")

    monkeypatch.setattr(evaluator, "_evaluate_checkpoint_impl", should_not_start)

    with pytest.raises(BrokenPipeError, match="progress consumer closed"):
        evaluator.evaluate_checkpoint(
            object(),
            tmp_path / "checkpoint.pt",
            progress_callback=broken_callback,
            progress_path=progress_path,
        )

    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["stage"] == "failed"
    assert progress["last_stage"] == "initializing"
    assert progress["exception_type"] == "BrokenPipeError"
    assert progress["message"] == "progress consumer closed"
