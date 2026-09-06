import math
from dataclasses import replace
from pathlib import Path

import pytest

from world_model.runtime.online_world_model import OnlineWorldModel
from world_model.training.dynamic_set_campaign import (
    SecondAttemptAdmissionEvidence,
    configure_second_attempt,
)
from world_model.training.dynamic_set_config import load_config
from world_model.training.dynamic_set_optimization import dynamic_set_capacity
from world_model.training.qualification_core import canonical_sha256

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROFILE = REPOSITORY_ROOT / "configs" / "rgbd_dynamic_set_planning_cpu.yaml"


def _capacity_admission_evidence(
    *,
    architecture_choice: str,
    base_config_sha256: str,
    resolved_config_sha256: str,
) -> SecondAttemptAdmissionEvidence:
    completed_update_seconds = [1.0] * 3_584
    completed_validation_seconds = [1.0] * 7
    timing_body = {
        "schema": "dynamic_set_update_timing_v3",
        "architecture_attempt_index": 2,
        "architecture_choice": architecture_choice,
        "base_config_sha256": base_config_sha256,
        "resolved_config_sha256": resolved_config_sha256,
        "prior_attempt_cumulative_seconds": 1.0,
        "architecture_attempt_sha256": "8" * 64,
        "completed_update_seconds": completed_update_seconds,
        "discarded_attempt_seconds": 0.0,
        "screen_wall_seconds": 1.0,
        "completed_validation_seconds": completed_validation_seconds,
        "validation_timing_sha256": "9" * 64,
        "cumulative_training_seconds": float(
            math.fsum(
                (
                    1.0,
                    1.0,
                    *completed_update_seconds,
                    *completed_validation_seconds,
                )
            )
        ),
    }
    return SecondAttemptAdmissionEvidence.from_sealed_execution_timing(
        execution_timing={
            **timing_body,
            "evidence_sha256": canonical_sha256(timing_body),
        },
        validation_boundary_updates=tuple(range(512, 3_585, 512)),
        protocol_sha256="1" * 64,
        source_sha256="2" * 64,
        first_attempt_sha256="3" * 64,
        first_screen_result_sha256="4" * 64,
        first_screen_wall_seconds=1.0,
        second_screen_result_sha256="5" * 64,
        execution_progress_record_sha256="6" * 64,
    )


def test_dynamic_set_profile_loads_with_exact_specification_contract() -> None:
    config = load_config(PROFILE)

    assert config.device.preference == "cpu"
    assert config.simulator.image_size == (64, 64)
    assert config.simulator.frame_rate == 20
    assert config.simulator.physics_rate == 120
    assert config.simulator.sequence_frames == 56
    assert (config.simulator.min_objects, config.simulator.max_objects) == (1, 6)
    assert config.simulator.world_bounds == ((-30.0, 30.0),) * 3
    assert config.simulator.radius_range == (0.21, 0.21)
    assert config.simulator.mass_range == (1.0, 1.0)
    assert config.simulator.drag_range == (0.05, 0.05)
    assert config.simulator.restitution_range == (0.7, 0.7)
    assert config.simulator.friction_range == (0.2, 0.2)
    assert config.simulator.external_impulse_probability == 0.0
    assert config.model.max_objects == 6
    assert config.model.state.fast_log_variance_min == -32.0
    assert config.model.rgbd.observation_mode == "set"
    assert config.model.rgbd.proposal_count == 8
    assert config.model.rgbd.set_log_variance_residual_limit == 20.0
    assert config.model.rgbd.temporal_history_size == 16
    assert config.model.rgbd.temporal_min_samples == 3
    assert config.model.rgbd.temporal_velocity_variance_floor == 1.0e-14
    assert config.model.lifecycle.birth_confirmations == 2
    assert config.model.lifecycle.max_missed_steps == 1
    assert not config.model.dynamics.analytic_free_motion_only
    assert not config.model.dynamics.modal_dynamics_enabled
    assert not config.model.dynamics.continuous_pair_force_enabled
    assert not config.model.dynamics.node_acceleration_enabled
    assert config.model.dynamics.event_driven_state_only_enabled
    assert config.model.dynamics.relation_process_uncertainty_enabled
    assert config.model.dynamics.process_noise_position == 1.0e-14
    assert config.model.dynamics.process_noise_velocity == 1.0e-14
    assert not config.model.dynamics.attention_residual_enabled
    assert config.model.filter.min_log_variance == -32.0
    assert config.training.batch_size == 4
    assert config.training.steps == 32_768
    assert config.training.checkpoint_every == 512
    assert config.evaluation.horizons_seconds == (0.05, 0.10, 0.25, 0.50, 1.0, 2.0)


def test_dynamic_set_profile_instantiates_below_every_capacity_limit() -> None:
    model = OnlineWorldModel.from_config(load_config(PROFILE))
    perception = model.observation_modules["rgbd"].set_proposer
    relation = model.dynamics.interactions.edge_network
    legacy = (
        model.dynamics.interactions.node_network,
        model.dynamics.uncertainty.process_network,
    )

    capacity = dynamic_set_capacity(
        model,
        perception=perception,
        relation=relation,
        legacy_modules=legacy,
    )

    assert perception.log_variance_residual_limit == 20.0
    assert model.updater.config.minimum_log_variance == -32.0
    assert model.dynamics.uncertainty.log_variance_bounds == (-32.0, 6.0)
    assert capacity.perception_parameters == 20_236
    assert capacity.relation_parameters == 647
    assert capacity.legacy_parameters == 976
    assert capacity.total_new_parameters == 20_883
    assert capacity.complete_model_weight_bytes == 87_436


@pytest.mark.parametrize(
    ("perception_passed", "expected_perception_width", "expected_relation_width"),
    [(False, 64, 16), (True, 32, 64)],
)
def test_evidence_localized_second_attempts_remain_inside_capacity(
    perception_passed: bool,
    expected_perception_width: int,
    expected_relation_width: int,
) -> None:
    base = load_config(PROFILE)
    if perception_passed:
        architecture_choice = "widen_relation"
        expected_config = replace(
            base,
            model=replace(
                base.model,
                dynamics=replace(base.model.dynamics, relation_hidden_dim=64),
            ),
        )
    else:
        architecture_choice = "widen_perception"
        expected_config = replace(
            base,
            model=replace(
                base.model,
                rgbd=replace(base.model.rgbd, set_feature_dim=64),
            ),
        )
    admission = _capacity_admission_evidence(
        architecture_choice=architecture_choice,
        base_config_sha256=canonical_sha256(base.to_dict()),
        resolved_config_sha256=canonical_sha256(expected_config.to_dict()),
    )
    choice, configured = configure_second_attempt(
        base,
        first_attempt_failed_early=True,
        oracle_state_dynamics_passed=True,
        perception_gates_passed=perception_passed,
        truth_state_contact_owns_error=perception_passed,
        admission_evidence=admission,
        admission_evidence_sha256=admission.evidence_sha256,
    )
    assert choice == architecture_choice
    assert configured is not None
    model = OnlineWorldModel.from_config(configured, device="cpu")
    perception = model.observation_modules["rgbd"].set_proposer
    assert perception is not None
    relation = model.dynamics.interactions.edge_network
    capacity = dynamic_set_capacity(
        model,
        perception=perception,
        relation=relation,
        legacy_modules=(
            model.dynamics.interactions.node_network,
            model.dynamics.uncertainty.process_network,
        ),
    )
    assert perception.feature_dim == expected_perception_width
    assert relation.layers[0].out_features == expected_relation_width
    assert capacity.perception_parameters <= 100_000
    assert capacity.relation_parameters <= 50_000
    assert capacity.total_new_parameters <= 250_000
    assert capacity.complete_model_weight_bytes <= 1 << 20
