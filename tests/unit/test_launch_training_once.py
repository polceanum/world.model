from __future__ import annotations

from pathlib import Path

import pytest

from scripts.launch_training_once import build_launchd_payload


def test_launch_payload_is_persistent_but_never_keepalive(tmp_path) -> None:
    repository = tmp_path / "repo"
    payload = build_launchd_payload(
        label="com.example.orpheus.run-20260803",
        repository_root=repository,
        python_executable=Path("/conda/orpheus/bin/python"),
        config_path=repository / "configs" / "run.yaml",
        run_name="20260803-120000-run",
        device="mps",
        initialize_from=repository / "source.pt",
        resume=None,
        overrides=["training.steps=32"],
        stdout_path=Path("/private/tmp/run.stdout.log"),
        stderr_path=Path("/private/tmp/run.stderr.log"),
    )

    assert payload["RunAtLoad"] is True
    assert payload["KeepAlive"] is False
    assert payload["ProgramArguments"][:4] == [
        "/usr/bin/caffeinate",
        "-dimsu",
        "/conda/orpheus/bin/python",
        str(repository / "train.py"),
    ]
    assert payload["ProgramArguments"][-2:] == ["--set", "training.steps=32"]
    assert payload["EnvironmentVariables"]["PYTHONPATH"] == str(repository)


def test_launch_payload_rejects_unsafe_label(tmp_path) -> None:
    with pytest.raises(ValueError, match="unsupported characters"):
        build_launchd_payload(
            label="com.example;rm",
            repository_root=tmp_path,
            python_executable=Path("/python"),
            config_path=tmp_path / "config.yaml",
            run_name="run",
            device="cpu",
            initialize_from=None,
            resume=None,
            overrides=[],
            stdout_path=tmp_path / "stdout.log",
            stderr_path=tmp_path / "stderr.log",
        )
