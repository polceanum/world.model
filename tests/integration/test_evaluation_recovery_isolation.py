from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import torch

from world_model.evaluation.evaluator import evaluate_checkpoint
from world_model.runtime import OnlineWorldModel
from world_model.training.checkpointing import save_checkpoint
from world_model.utils.config import load_config
from world_model.utils.device import select_device
from world_model.utils.version import SIMULATOR_VERSION, SPECIFICATION_VERSION

_RECOVERY_METRICS = {
    "perturbation_prior_position_error_m",
    "perturbation_posterior_position_error_m",
    "perturbation_correction_improvement_m",
    "perturbation_correction_improvement_fraction",
    "perturbation_positive_correction_rate",
    "perturbation_evaluated_object_horizons",
    "injected_perturbation_batch_updates",
    "recovery_probe_evaluated_episodes",
    "recovery_probe_nonfinite_output_count",
    "recovery_probe_post_observation_std_contraction_mean_m",
    "post_observation_std_contraction_mean_m",
}


def _evaluation_config(*, recovery_probe_enabled: bool):
    source = load_config("configs/tiny_overfit.yaml")
    config = replace(
        source,
        device=replace(source.device, preference="cpu"),
        simulator=replace(
            source.simulator,
            image_size=(24, 24),
            sequence_frames=4,
            min_objects=1,
            max_objects=1,
            camera_motion="fixed",
            render_noise_std=0.0,
        ),
        model=replace(
            source.model,
            rgb=replace(
                source.model.rgb,
                backbone_channels=(8, 16, 24, 32),
                feature_dim=16,
                proposal_queries=3,
                roi_size=8,
            ),
            dynamics=replace(source.model.dynamics, hidden_dim=24),
            filter=replace(source.model.filter, hidden_dim=32),
            lifecycle=replace(
                source.model.lifecycle,
                birth_confidence=0.0,
                birth_confirmations=1,
            ),
        ),
        training=replace(
            source.training,
            batch_size=1,
            validation_episodes=1,
            num_workers=0,
            horizon_weights=(1.0,),
        ),
        evaluation=replace(
            source.evaluation,
            horizons_seconds=(0.05,),
            episodes=1,
            recovery_probe_enabled=recovery_probe_enabled,
            perturbation_position_std=(0.9 if recovery_probe_enabled else 0.01),
            perturbation_velocity_std=(1.2 if recovery_probe_enabled else 0.02),
        ),
    )
    config.validate()
    return config


def _report(path: str) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_recovery_probe_cannot_change_primary_metrics_or_posterior_trace(tmp_path) -> None:
    clean_config = _evaluation_config(recovery_probe_enabled=False)
    probe_config = _evaluation_config(recovery_probe_enabled=True)
    assert (
        clean_config.evaluation.perturbation_position_std
        != probe_config.evaluation.perturbation_position_std
    )
    assert (
        clean_config.evaluation.perturbation_velocity_std
        != probe_config.evaluation.perturbation_velocity_std
    )
    model = OnlineWorldModel.from_config(clean_config, device="cpu")
    checkpoint = save_checkpoint(
        tmp_path / "checkpoint.pt",
        model=model,
        optimizer=None,
        config=clean_config,
        step=0,
        device="cpu",
    )
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_payload["simulator_version"] = "sphere_world_v4"
    checkpoint_payload["specification_version"] = "1.12"
    checkpoint_payload["git"] = {
        "commit": "prior-simulator-source",
        "dirty": False,
        "worktree_fingerprint": "prior-worktree",
        "runtime_source_fingerprint": "prior-runtime",
    }
    torch.save(checkpoint_payload, checkpoint)
    device = select_device("cpu")
    probe_progress: list[dict[str, object]] = []

    clean_result = evaluate_checkpoint(
        clean_config,
        checkpoint,
        output_dir=tmp_path / "clean",
        device_info=device,
    )
    clean = _report(clean_result["json_report"])
    probed_result = evaluate_checkpoint(
        probe_config,
        checkpoint,
        output_dir=tmp_path / "probed",
        device_info=device,
        progress_callback=probe_progress.append,
    )
    probed = _report(probed_result["json_report"])

    clean_metadata = clean["metadata"]
    probed_metadata = probed["metadata"]
    assert clean_metadata["checkpoint_simulator_version"] == "sphere_world_v4"
    assert clean_metadata["evaluation_simulator_version"] == SIMULATOR_VERSION
    assert clean_metadata["checkpoint_specification_version"] == "1.12"
    assert clean_metadata["evaluation_specification_version"] == SPECIFICATION_VERSION
    assert clean_metadata["checkpoint_source_provenance"] == checkpoint_payload["git"]
    evaluation_source = clean_metadata["evaluation_source_provenance"]
    assert isinstance(evaluation_source, dict)
    assert set(evaluation_source) >= {
        "commit",
        "dirty",
        "worktree_fingerprint",
        "runtime_source_fingerprint",
    }
    report_text = Path(clean_result["markdown_report"]).read_text(encoding="utf-8")
    assert "- checkpoint_simulator_version: `sphere_world_v4`" in report_text
    assert f"- evaluation_simulator_version: `{SIMULATOR_VERSION}`" in report_text
    assert "- checkpoint_specification_version: `1.12`" in report_text
    assert f"- evaluation_specification_version: `{SPECIFICATION_VERSION}`" in report_text
    assert "- checkpoint_source_provenance:" in report_text
    assert "- evaluation_source_provenance:" in report_text
    assert clean_metadata["primary_online_pass_evaluator_state_perturbation_free"] is True
    assert probed_metadata["primary_online_pass_evaluator_state_perturbation_free"] is True
    assert clean_metadata["primary_online_pass_intervention_free_scope"] == (
        "evaluator_injected_state_perturbations_only"
    )
    assert clean_metadata["recovery_probe_enabled"] is False
    assert probed_metadata["recovery_probe_enabled"] is True
    protocol_bytes = json.dumps(
        clean_metadata["resolved_evaluation_protocol"],
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    assert (
        clean_metadata["resolved_evaluation_protocol_sha256"]
        == hashlib.sha256(protocol_bytes).hexdigest()
    )
    assert clean_metadata["resolved_evaluation_protocol"]["seed_manifest"] == [200000]
    assert (
        clean_metadata["primary_online_pass_simulator_external_actuation_object_event_count"]
        == (clean["metrics"]["simulator_external_actuation_object_event_count"])
    )
    assert (
        clean_metadata["primary_posterior_trace_sha256"]
        == probed_metadata["primary_posterior_trace_sha256"]
    )
    assert (
        clean_metadata["primary_physical_metrics_sha256"]
        == probed_metadata["primary_physical_metrics_sha256"]
    )

    clean_primary = {
        name: value
        for name, value in clean["metrics"].items()
        if name not in _RECOVERY_METRICS and "latency" not in name
    }
    probed_primary = {
        name: value
        for name, value in probed["metrics"].items()
        if name not in _RECOVERY_METRICS and "latency" not in name
    }
    assert clean_primary == probed_primary
    expected_hashed_keys = sorted(clean_primary)
    expected_primary_hash = hashlib.sha256(
        json.dumps(
            clean_primary,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    expected_hash_exclusions = {
        "latency_metric_name_substring": "latency",
        "recovery_only_metric_names": sorted(_RECOVERY_METRICS),
    }
    for metadata in (clean_metadata, probed_metadata):
        assert metadata["primary_physical_metrics_hashed_keys"] == expected_hashed_keys
        assert metadata["primary_physical_metrics_scope"] == (
            "clean_primary_metrics_before_isolated_recovery_probe_append"
        )
        assert metadata["primary_physical_metrics_hash_excludes"] == expected_hash_exclusions
        assert metadata["primary_physical_metrics_sha256"] == expected_primary_hash
    assert "posterior_current_velocity_sse" in clean["metrics"]
    for axis in ("x", "y", "z"):
        assert f"posterior_current_velocity_{axis}_sse" in clean["metrics"]
        assert f"posterior_current_position_{axis}_gaussian_nll_sum" in clean["metrics"]
    for horizon in clean_metadata["per_scenario_metrics_horizons"]:
        assert f"model@{horizon}_velocity_sse" in clean["metrics"]
        assert f"collision@{horizon}_true_positive_count" in clean["metrics"]
        assert f"model@{horizon}_position_gaussian_nll_sum" in clean["metrics"]
        assert f"model@{horizon}_position_calibration_error90" in clean["metrics"]
        assert f"forecast_identity@{horizon}_association_count" in clean["metrics"]
        for axis in ("x", "y", "z"):
            assert f"model@{horizon}_velocity_{axis}_sse" in clean["metrics"]
            assert f"model@{horizon}_position_{axis}_gaussian_nll_sum" in clean["metrics"]
    for prefix in ("rgb_global_update", "rgb_fast_update", "future_rollout"):
        count = clean["metrics"][f"{prefix}_latency_sample_count"]
        assert count >= 0.0
        assert f"{prefix}_latency_sum_ms" in clean["metrics"]
    assert clean["metrics"]["injected_perturbation_batch_updates"] == 0.0
    assert clean["metrics"]["recovery_probe_evaluated_episodes"] == 0.0
    assert probed["metrics"]["injected_perturbation_batch_updates"] == 1.0
    assert probed["metrics"]["recovery_probe_evaluated_episodes"] == 1.0
    progress_stages = [event["stage"] for event in probe_progress]
    assert progress_stages[0] == "initializing"
    assert (
        probe_progress[0]["evaluation_source_provenance"]
        == probed_metadata["evaluation_source_provenance"]
    )
    assert "recovery_probe_started" in progress_stages
    assert "recovery_probe_batch_complete" in progress_stages
    assert progress_stages[-1] == "completed"


def test_mutable_checkpoint_path_cannot_split_primary_and_recovery_weights(tmp_path) -> None:
    config = _evaluation_config(recovery_probe_enabled=True)
    model = OnlineWorldModel.from_config(config, device="cpu")
    checkpoint = save_checkpoint(
        tmp_path / "last.pt",
        model=model,
        optimizer=None,
        config=config,
        step=0,
        device="cpu",
    )
    expected_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    progress: list[dict[str, object]] = []

    def replace_mutable_source(event: dict[str, object]) -> None:
        progress.append(event)
        if event["stage"] == "initializing":
            checkpoint.write_bytes(b"concurrent trainer replacement")

    result = evaluate_checkpoint(
        config,
        checkpoint,
        output_dir=tmp_path / "snapshot-evaluation",
        device_info=select_device("cpu"),
        progress_callback=replace_mutable_source,
    )
    report = _report(result["json_report"])

    assert report["metadata"]["checkpoint_sha256"] == expected_sha256
    assert report["metrics"]["recovery_probe_evaluated_episodes"] == 1.0
    assert progress[0]["checkpoint_sha256"] == expected_sha256
