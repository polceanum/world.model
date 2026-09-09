from __future__ import annotations

import json

import pytest
import torch

import world_model.evaluation.capability_factor_runner as runner
from scripts.run_capability_factor import arguments, main
from world_model.evaluation.capability_factor_runner import (
    SUPPORTED_CAPABILITY_FACTORS,
    factor_physical_rows,
    materialize_capability_physical_episode,
)
from world_model.evaluation.general_capability import manifest_sha256
from world_model.training.dynamic_set_evaluation import DynamicSetEvaluationError
from world_model.training.dynamic_set_protocol import PHYSICAL_CELLS


def test_executable_capability_factor_rows_cover_all_physical_cells() -> None:
    for factor in SUPPORTED_CAPABILITY_FACTORS:
        first = factor_physical_rows(factor)
        second = factor_physical_rows(factor)
        assert first == second
        assert len(first) == len(PHYSICAL_CELLS) == 22
        assert {item[1].cell_index for item in first} == set(range(22))
        assert all(item[0].seed == item[1].seed for item in first)
        assert manifest_sha256(tuple(item[0] for item in first)) == manifest_sha256(
            tuple(item[0] for item in second)
        )


def test_sensor_noise_is_deterministic_observable_only_and_truth_isolated() -> None:
    capability_row, physical_row = factor_physical_rows("sensor_noise")[0]
    clean = materialize_capability_physical_episode(
        capability_row,
        physical_row,
        apply_factor=False,
    )
    first = materialize_capability_physical_episode(
        capability_row,
        physical_row,
        apply_factor=True,
    )
    second = materialize_capability_physical_episode(
        capability_row,
        physical_row,
        apply_factor=True,
    )

    assert torch.equal(first.episode["rgb"], second.episode["rgb"])
    assert torch.equal(first.episode["depth"], second.episode["depth"])
    assert not torch.equal(first.episode["rgb"], clean.episode["rgb"])
    assert not torch.equal(first.episode["depth"], clean.episode["depth"])
    for root in ("objects", "labels", "events"):
        assert first.episode[root].keys() == clean.episode[root].keys()
        assert all(
            torch.equal(first.episode[root][name], clean.episode[root][name])
            for name in first.episode[root]
        )
    _frames, boundary = first.public_frames_with_boundary()
    assert boundary.truth_leakage_count == 0


def test_physical_parameter_factor_changes_simulation_and_remains_truth_isolated() -> None:
    capability_row, physical_row = factor_physical_rows("physical_parameters")[0]
    clean = materialize_capability_physical_episode(
        capability_row,
        physical_row,
        apply_factor=False,
    )
    varied = materialize_capability_physical_episode(
        capability_row,
        physical_row,
        apply_factor=True,
    )
    controls = capability_row.controls
    active = varied.episode["objects"]["active"]
    for name, expected in (
        ("radius", controls.radius_m),
        ("mass", controls.mass_kg),
        ("drag", controls.drag_per_second),
        ("restitution", controls.restitution),
        ("friction", controls.friction),
    ):
        values = varied.episode["objects"][name][active]
        assert torch.equal(values, values.new_full(values.shape, expected))
    assert not torch.equal(
        varied.episode["objects"]["position"],
        clean.episode["objects"]["position"],
    )
    assert not torch.equal(varied.episode["rgb"], clean.episode["rgb"])
    _frames, boundary = varied.public_frames_with_boundary()
    assert boundary.truth_leakage_count == 0


def test_partial_visibility_has_bounded_window_and_exact_recovery() -> None:
    capability_row, physical_row = factor_physical_rows("partial_visibility")[0]
    clean = materialize_capability_physical_episode(
        capability_row,
        physical_row,
        apply_factor=False,
    )
    occluded = materialize_capability_physical_episode(
        capability_row,
        physical_row,
        apply_factor=True,
    )
    details = occluded.episode["metadata"]["capability_observation_details"]
    start = details["start_frame"]
    end = details["end_frame_exclusive"]
    assert details["removed_pixels"] > 0
    assert end - start == capability_row.controls.occlusion_frames
    assert end + capability_row.controls.recovery_observation_frames <= 15
    assert not torch.equal(occluded.episode["rgb"][start:end], clean.episode["rgb"][start:end])
    assert torch.equal(occluded.episode["rgb"][end:], clean.episode["rgb"][end:])
    assert torch.equal(occluded.episode["depth"][end:], clean.episode["depth"][end:])
    _frames, boundary = occluded.public_frames_with_boundary()
    assert boundary.truth_leakage_count == 0


def test_camera_motion_is_calibrated_deterministic_and_truth_isolated() -> None:
    capability_row, physical_row = next(
        item
        for item in factor_physical_rows("camera_motion")
        if item[0].controls.camera_motion != "static"
    )
    clean = materialize_capability_physical_episode(
        capability_row,
        physical_row,
        apply_factor=False,
    )
    first = materialize_capability_physical_episode(
        capability_row,
        physical_row,
        apply_factor=True,
    )
    second = materialize_capability_physical_episode(
        capability_row,
        physical_row,
        apply_factor=True,
    )

    assert torch.equal(first.episode["rgb"], second.episode["rgb"])
    assert torch.equal(
        first.episode["camera"]["world_from_camera"],
        second.episode["camera"]["world_from_camera"],
    )
    assert not torch.equal(first.episode["rgb"], clean.episode["rgb"])
    moving_transform = first.episode["camera"]["world_from_camera"]
    assert not torch.equal(moving_transform[0], moving_transform[-1])
    assert bool(first.episode["camera"]["calibrated"].all())
    for name in (
        "id",
        "active",
        "position",
        "velocity",
        "radius",
        "mass",
        "restitution",
        "drag",
        "friction",
    ):
        assert torch.equal(first.episode["objects"][name], clean.episode["objects"][name])
    for name in first.episode["events"]:
        assert torch.equal(first.episode["events"][name], clean.episode["events"][name])
    _frames, boundary = first.public_frames_with_boundary()
    assert boundary.truth_leakage_count == 0


def test_known_action_factor_is_public_exactly_once_and_broad() -> None:
    rows = factor_physical_rows("known_actions")
    enabled = [item for item in rows if item[0].controls.known_action_enabled]
    assert enabled
    assert len({item[0].controls.impulse_phase for item in enabled}) >= 2
    assert len({item[0].controls.impulse_direction_world for item in enabled}) == len(enabled)
    capability_row, physical_row = enabled[0]
    clean = materialize_capability_physical_episode(
        capability_row,
        physical_row,
        apply_factor=False,
    )
    acted = materialize_capability_physical_episode(
        capability_row,
        physical_row,
        apply_factor=True,
    )
    event_mask = acted.episode["events"]["known_action_observed"]
    frame = {"early": 10, "middle": 34, "late": 44}[capability_row.controls.impulse_phase]
    assert int(event_mask.sum()) == 1
    assert bool(event_mask[frame, capability_row.controls.impulse_target_rank])
    impulse = acted.episode["events"]["known_impulse_world"]
    assert torch.count_nonzero(impulse, dim=-1).gt(0).sum() == 1
    assert torch.isclose(
        torch.linalg.vector_norm(impulse[frame]),
        torch.tensor(capability_row.controls.impulse_magnitude),
        atol=1.0e-6,
    )
    assert torch.equal(acted.episode["rgb"][:frame], clean.episode["rgb"][:frame])
    assert not torch.equal(
        acted.episode["objects"]["velocity"][frame:], clean.episode["objects"]["velocity"][frame:]
    )
    frames, boundary = acted.public_frames_with_boundary()
    assert sum(bool(item.known_action_observed[0]) for item in frames) == 1
    assert boundary.truth_leakage_count == 0


def test_compositional_holdout_combines_controls_on_disjoint_seeds() -> None:
    capability_row, physical_row = next(
        item
        for item in factor_physical_rows("compositional_holdout")
        if item[0].controls.camera_motion != "static"
    )
    assert capability_row.split == "compositional_holdout"
    assert capability_row.seed >= 92_000_000
    assert physical_row.split == "development"
    clean = materialize_capability_physical_episode(
        capability_row,
        physical_row,
        apply_factor=False,
    )
    composed = materialize_capability_physical_episode(
        capability_row,
        physical_row,
        apply_factor=True,
    )
    details = composed.episode["metadata"]["capability_observation_details"]
    assert set(details) == {
        "physical_parameters",
        "known_action",
        "camera",
        "partial_visibility",
        "sensor_noise",
    }
    assert not torch.equal(composed.episode["rgb"], clean.episode["rgb"])
    assert not torch.equal(
        composed.episode["camera"]["world_from_camera"],
        clean.episode["camera"]["world_from_camera"],
    )
    assert int(composed.episode["events"]["known_action_observed"].sum()) == 1
    active = composed.episode["objects"]["active"]
    radius = composed.episode["objects"]["radius"][active]
    assert torch.equal(
        radius,
        radius.new_full(radius.shape, capability_row.controls.radius_m),
    )
    _frames, boundary = composed.public_frames_with_boundary()
    assert boundary.truth_leakage_count == 0


def test_factor_cli_dry_run_is_read_only(capsys) -> None:
    parsed = arguments(["--factor", "sensor_noise", "--dry-run"])
    assert parsed.factor == "sensor_noise"
    assert main(["--factor", "sensor_noise", "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert '"physical_episodes": 22' in output
    assert '"generated_episodes_retained": false' in output


def test_runtime_refusal_writes_compact_failed_summary(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "_seed_process", lambda _seed, _threads: None)
    monkeypatch.setattr(runner, "load_config", lambda _path: object())
    monkeypatch.setattr(
        runner,
        "_model_from_workbench_checkpoint",
        lambda _config, _path: (object(), {"checkpoint_sha256": "a" * 64}),
    )
    monkeypatch.setattr(
        runner,
        "materialize_capability_physical_episode",
        lambda *_args, **_kwargs: object(),
    )

    def refuse(_model, materializations, **_kwargs):
        next(iter(materializations))
        raise DynamicSetEvaluationError("public action target could not be resolved at frame 34")

    monkeypatch.setattr(runner, "evaluate_dynamic_set_materializations", refuse)
    run = tmp_path / "runs" / "failed-composition"
    report = runner.run_incumbent_factor_evaluation(
        factor="compositional_holdout",
        model_config_path="unused.yaml",
        checkpoint_path="unused.pt",
        run_directory=run,
        progress=None,
    )

    assert report["status"] == "failed"
    summary = json.loads((run / "capability_summary.json").read_text(encoding="utf-8"))
    assert summary["lifecycle_status"] == "failed"
    assert summary["failure_attribution"]["ablation_owner"] == "appearance_target_resolution"
    assert summary["factor_metrics"]["compositional_holdout"][
        "observable_target_handle_resolution_upper_bound"
    ] == pytest.approx(21 / 22)
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert not any(path.suffix == ".pt" for path in run.iterdir())
