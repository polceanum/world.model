"""Fast synthetic execution tests for the specification-1.61 trainer."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor, nn

from world_model.runtime.online_world_model import OnlineWorldModel
from world_model.training.dynamic_set_config import load_config
from world_model.training.dynamic_set_objectives import (
    DynamicsLossInputs,
    PerceptionLossInputs,
)
from world_model.training.dynamic_set_optimization import learning_rate_multiplier
from world_model.training.dynamic_set_protocol import (
    PHYSICAL_CELLS,
    canonical_sha256,
    physical_manifest,
)
from world_model.training.dynamic_set_trainer import (
    SCREEN_EXAMPLE_COUNT,
    SCREEN_MANIFEST_SHA256,
    SCREEN_ROWS,
    DynamicSetCausalSupport,
    DynamicSetObjectiveInputs,
    DynamicSetScreenSnapshot,
    DynamicSetTrainer,
    DynamicSetTrainingMicrobatch,
    DynamicSetUpdateRejected,
    causal_scene_predictable_mask,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROFILE = REPOSITORY_ROOT / "configs" / "rgbd_dynamic_set_planning_cpu.yaml"
TRAINING_ROWS = physical_manifest("training")[:44]
SOURCE = {
    "commit": "synthetic-test",
    "dirty": False,
    "runtime_source_fingerprint": "a" * 64,
}


class _TinyDynamicSetModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.perception = nn.Linear(1, 1)
        self.relation = nn.Linear(1, 1)
        self.frozen_other = nn.Linear(1, 1)


@dataclass(frozen=True)
class _FakeMaterialization:
    episode: Mapping[str, Any]


class _SyntheticObjectiveAdapter:
    def __init__(self, *, bad_causal_mask: bool = False, nan_gradient: bool = False) -> None:
        self.calls = 0
        self.bad_causal_mask = bad_causal_mask
        self.nan_gradient = nan_gradient
        self.seen_cells: list[tuple[int, ...]] = []

    def state_dict(self) -> Mapping[str, Any]:
        return {"calls": self.calls}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if set(state) != {"calls"} or type(state["calls"]) is not int:
            raise ValueError("synthetic adapter state differs")
        self.calls = state["calls"]

    def build_objective_inputs(
        self,
        model: nn.Module,
        microbatch: DynamicSetTrainingMicrobatch,
    ) -> DynamicSetObjectiveInputs:
        assert isinstance(model, _TinyDynamicSetModel)
        self.calls += 1
        self.seen_cells.append(tuple(row.cell_index for row in microbatch.rows))
        # Consume global Torch RNG so the resume test proves RNG restoration,
        # rather than only deterministic schedule reconstruction.
        perception_value = model.perception(torch.rand(1, 1)).reshape(()) * 0.1
        relation_value = model.relation(torch.rand(1, 1)).reshape(()) * 0.1
        if self.nan_gradient:
            perception_value.register_hook(lambda gradient: torch.full_like(gradient, float("nan")))

        batch, proposals, objects, steps = 4, 8, 6, 2
        target_exists = torch.ones(batch, proposals, dtype=torch.bool)
        target_masks = torch.zeros(batch, proposals + 1, 1, 1)
        target_masks[:, 1:] = 1.0
        perception = PerceptionLossInputs(
            mask_logits=perception_value.expand(batch, proposals + 1, 1, 1),
            target_masks=target_masks,
            existence_logits=perception_value.expand(batch, proposals),
            target_exists=target_exists,
            metric_position=perception_value.expand(batch, proposals, 3),
            target_position=torch.zeros(batch, proposals, 3),
            position_log_variance=(0.1 * perception_value).expand(batch, proposals, 3),
            appearance=perception_value.expand(batch, proposals, 8),
            target_appearance=torch.ones(batch, proposals, 8),
        )

        external = torch.stack(
            [
                materialization.episode["events"]["externally_actuated"]
                for materialization in microbatch.materializations
            ]
        )
        known = torch.stack(
            [
                materialization.episode["events"]["known_action_observed"]
                for materialization in microbatch.materializations
            ]
        )
        causal_support = DynamicSetCausalSupport(
            externally_actuated=external,
            known_action_observed=known,
            anchor_frame_index=torch.zeros(batch, dtype=torch.int64),
            target_frame_index=torch.tensor([[1, 2]] * batch, dtype=torch.int64),
        )
        predictable = causal_scene_predictable_mask(causal_support)
        state_allowed = predictable[:, :, None].expand(batch, steps, objects)
        pair_allowed = predictable[:, :, None, None].expand(
            batch,
            steps,
            objects,
            objects,
        )
        if self.bad_causal_mask:
            state_allowed = torch.ones_like(state_allowed)
            pair_allowed = torch.ones_like(pair_allowed)
        known_scene = known.any(dim=(1, 2))
        known_action_mask = state_allowed & known_scene[:, None, None]
        diagonal = torch.eye(objects, dtype=torch.bool).view(1, 1, objects, objects)
        dynamics = DynamicsLossInputs(
            predicted_state=(0.1 * relation_value).expand(batch, steps, objects, 6),
            target_state=torch.zeros(batch, steps, objects, 6),
            process_log_variance=(0.05 * relation_value).expand(
                batch,
                steps,
                objects,
                6,
            ),
            contact_window_mask=state_allowed,
            collision_logits=(0.1 * relation_value).expand(
                batch,
                steps,
                objects,
                objects,
            ),
            collision_target=torch.zeros(
                batch,
                steps,
                objects,
                objects,
                dtype=torch.bool,
            ),
            collision_support=pair_allowed & ~diagonal,
            known_action_predictable_mask=known_action_mask,
            relation_residual=(0.1 * relation_value).expand(
                batch,
                steps,
                objects,
                3,
            ),
            unaffected_object_mask=state_allowed & ~known_action_mask,
        )
        return DynamicSetObjectiveInputs(
            perception=perception,
            dynamics=dynamics,
            causal_support=causal_support,
        )


def _materializer(*, hidden_impulse: bool = False):
    def materialize(row):
        externally_actuated = torch.zeros(3, 6, dtype=torch.bool)
        known_action_observed = torch.zeros_like(externally_actuated)
        if row.known_action:
            assert row.action_target_rank is not None
            externally_actuated[1, row.action_target_rank] = True
            known_action_observed[1, row.action_target_rank] = True
        if hidden_impulse:
            hidden_slot = (row.action_target_rank or 0) + 1
            externally_actuated[1, hidden_slot % 6] = True
        return _FakeMaterialization(
            episode={
                "events": {
                    "externally_actuated": externally_actuated,
                    "known_action_observed": known_action_observed,
                }
            }
        )

    return materialize


def _trainer(
    *,
    model: _TinyDynamicSetModel | None = None,
    adapter: _SyntheticObjectiveAdapter | None = None,
    hidden_impulse: bool = False,
    maximum_gradient_norm: float = 1.0,
    source: Mapping[str, Any] = SOURCE,
    validation_hook=None,
    training_rows=TRAINING_ROWS,
    screen_only: bool = False,
) -> DynamicSetTrainer:
    resolved_model = _TinyDynamicSetModel() if model is None else model
    resolved_adapter = _SyntheticObjectiveAdapter() if adapter is None else adapter
    return DynamicSetTrainer(
        model=resolved_model,
        perception_owner=resolved_model.perception,
        relation_owner=resolved_model.relation,
        legacy_owners=(resolved_model.frozen_other,),
        training_rows=training_rows,
        objective_adapter=resolved_adapter,
        resolved_config={"profile": "synthetic-dynamic-set-test"},
        source_provenance=source,
        schedule_seed=1_610,
        materializer=_materializer(hidden_impulse=hidden_impulse),
        validation_hook=validation_hook,
        maximum_gradient_norm=maximum_gradient_norm,
        screen_only=screen_only,
        test_only_synthetic_manifest=True,
    )


def _assert_model_state_equal(left: nn.Module, right: Mapping[str, Tensor]) -> None:
    assert set(left.state_dict()) == set(right)
    for name, value in left.state_dict().items():
        torch.testing.assert_close(value, right[name], rtol=0.0, atol=0.0)


def _assert_tensor_tree_equal(left: Any, right: Any) -> None:
    if isinstance(left, Tensor):
        assert isinstance(right, Tensor)
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
    elif isinstance(left, Mapping):
        assert isinstance(right, Mapping)
        assert set(left) == set(right)
        for key in left:
            _assert_tensor_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_tensor_tree_equal(left_item, right_item)
    else:
        assert left == right


def test_one_update_executes_six_b4_batches_over_all_22_cells_and_only_two_owners() -> None:
    torch.manual_seed(3)
    model = _TinyDynamicSetModel()
    adapter = _SyntheticObjectiveAdapter()
    trainer = _trainer(model=model, adapter=adapter)
    frozen_before = deepcopy(model.frozen_other.state_dict())

    report = trainer.run_update()

    assert report.completed_updates == 1
    assert report.next_update_index == 1
    assert report.materialized_examples == 24
    assert len(adapter.seen_cells) == 6
    assert all(len(cell_group) == 4 for cell_group in adapter.seen_cells)
    assert set(cell for group in adapter.seen_cells for cell in group) == set(range(22))
    assert sorted(report.cell_counts).count(2) == 2
    assert sorted(report.cell_counts).count(1) == 20
    assert [group["name"] for group in trainer.optimizer.param_groups] == [
        "perception",
        "relation",
    ]
    assert not any(parameter.requires_grad for parameter in model.frozen_other.parameters())
    for name, value in model.frozen_other.state_dict().items():
        torch.testing.assert_close(value, frozen_before[name], rtol=0.0, atol=0.0)
    assert report.finite_owner_gradients
    assert report.complete_gradient_retention >= 0.10
    assert not any(
        token in name
        for name in report.loss_terms
        for token in ("planning", "winner", "regret", "ranking", "task_success")
    )


def test_known_actions_remain_predictable_and_hidden_impulses_censor_the_coupled_scene() -> None:
    external = torch.zeros(2, 3, 2, dtype=torch.bool)
    known = torch.zeros_like(external)
    external[0, 1, 0] = True
    known[0, 1, 0] = True
    external[1, 1, 1] = True
    support = DynamicSetCausalSupport(
        externally_actuated=external,
        known_action_observed=known,
        anchor_frame_index=torch.zeros(2, dtype=torch.int64),
        target_frame_index=torch.tensor([[1, 2], [1, 2]], dtype=torch.int64),
    )

    assert causal_scene_predictable_mask(support).tolist() == [
        [True, True],
        [False, False],
    ]


def test_incomplete_hidden_impulse_censoring_rejects_before_any_mutation() -> None:
    torch.manual_seed(5)
    model = _TinyDynamicSetModel()
    adapter = _SyntheticObjectiveAdapter(bad_causal_mask=True)
    trainer = _trainer(model=model, adapter=adapter, hidden_impulse=True)
    model_before = deepcopy(model.state_dict())
    rng_before = torch.get_rng_state().clone()
    next_before = trainer.checkpoint_payload()["next_sample_state"]

    with pytest.raises(DynamicSetUpdateRejected, match="future hidden impulse"):
        trainer.run_update()

    _assert_model_state_equal(model, model_before)
    torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0.0, atol=0.0)
    assert trainer.completed_updates == 0
    assert trainer.rejected_update_count == 1
    assert adapter.calls == 0
    assert trainer.optimizer.state_dict()["state"] == {}
    assert trainer.scheduler.state_dict()["last_epoch"] == 0
    assert trainer.checkpoint_payload()["next_sample_state"] == next_before


@pytest.mark.parametrize("failure", ["nan_gradient", "retention"])
def test_numerical_and_sub_ten_percent_gradient_failures_are_pre_mutation(failure: str) -> None:
    torch.manual_seed(7)
    model = _TinyDynamicSetModel()
    adapter = _SyntheticObjectiveAdapter(nan_gradient=failure == "nan_gradient")
    trainer = _trainer(
        model=model,
        adapter=adapter,
        maximum_gradient_norm=1.0e-12 if failure == "retention" else 1.0,
    )
    before = deepcopy(model.state_dict())

    expected = "NaN or Inf" if failure == "nan_gradient" else "complete-gradient retention"
    with pytest.raises(DynamicSetUpdateRejected, match=expected):
        trainer.run_update()

    _assert_model_state_equal(model, before)
    assert trainer.completed_updates == 0
    assert trainer.rejected_update_count == 1
    assert trainer.optimizer.state_dict()["state"] == {}


def test_checkpoint_round_trip_restores_exact_optimizer_scheduler_rng_adapter_and_next_draw() -> (
    None
):
    torch.manual_seed(11)
    first_model = _TinyDynamicSetModel()
    first_adapter = _SyntheticObjectiveAdapter()
    first = _trainer(model=first_model, adapter=first_adapter)
    first.run_update()
    checkpoint = first.checkpoint_payload()
    checkpoint_rng = checkpoint["rng_state"]["torch_cpu"].clone()
    assert checkpoint["trainer_state"]["minimum_complete_gradient_retention"] == (
        first.observed_minimum_complete_gradient_retention
    )
    assert checkpoint["trainer_state"]["rejected_optimizer_mutation_count"] == 0

    expected_report = first.run_update()
    expected = first.checkpoint_payload()

    second_model = _TinyDynamicSetModel()
    second_adapter = _SyntheticObjectiveAdapter()
    second = _trainer(model=second_model, adapter=second_adapter)
    second.load_checkpoint_payload(checkpoint)
    torch.testing.assert_close(torch.get_rng_state(), checkpoint_rng, rtol=0.0, atol=0.0)
    actual_report = second.run_update()
    actual = second.checkpoint_payload()

    assert actual_report.objective == expected_report.objective
    assert actual["next_sample_state"] == expected["next_sample_state"]
    assert actual["scheduler_state"] == expected["scheduler_state"]
    assert actual["adapter_state"] == expected["adapter_state"]
    assert second.observed_minimum_complete_gradient_retention == (
        first.observed_minimum_complete_gradient_retention
    )
    assert second.rejected_optimizer_mutation_count == 0
    _assert_tensor_tree_equal(actual["model_state"], expected["model_state"])
    _assert_tensor_tree_equal(actual["optimizer_state"], expected["optimizer_state"])


def test_failed_exact_rollback_poisons_checkpoint_emission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _TinyDynamicSetModel()
    trainer = _trainer(
        model=model,
        adapter=_SyntheticObjectiveAdapter(nan_gradient=True),
    )
    restore = trainer._restore_transaction

    def corrupt_after_restore(snapshot: Mapping[str, Any]) -> None:
        restore(snapshot)
        with torch.no_grad():
            model.perception.weight.add_(1.0)

    monkeypatch.setattr(trainer, "_restore_transaction", corrupt_after_restore)

    with pytest.raises(RuntimeError, match="rollback failed"):
        trainer.run_update()
    assert trainer.rejected_optimizer_mutation_count == 1
    with pytest.raises(RuntimeError, match="integrity-poisoned"):
        trainer.checkpoint_payload()


def test_checkpoint_is_loadable_through_safe_weights_only_boundary() -> None:
    trainer = _trainer()
    trainer.run_update()
    stream = BytesIO()
    torch.save(trainer.checkpoint_payload(), stream)
    stream.seek(0)

    payload = torch.load(stream, map_location="cpu", weights_only=True)
    restored = _trainer()
    restored.load_checkpoint_payload(payload, restore_rng=False)

    assert restored.completed_updates == 1


def test_checkpoint_binding_mismatch_rejects_without_destination_mutation() -> None:
    torch.manual_seed(13)
    source = _trainer()
    source.run_update()
    payload = source.checkpoint_payload()
    destination_model = _TinyDynamicSetModel()
    destination = _trainer(
        model=destination_model,
        source={**SOURCE, "runtime_source_fingerprint": "b" * 64},
    )
    before = deepcopy(destination_model.state_dict())

    with pytest.raises(ValueError, match="binding differs"):
        destination.load_checkpoint_payload(payload)

    _assert_model_state_equal(destination_model, before)
    assert destination.completed_updates == 0
    assert destination.optimizer.state_dict()["state"] == {}


@pytest.mark.parametrize("corruption", ["moment_shape", "scheduler_rate"])
def test_checkpoint_optimizer_and_scheduler_corruption_is_rejected_atomically(
    corruption: str,
) -> None:
    torch.manual_seed(17)
    source = _trainer()
    source.run_update()
    payload = source.checkpoint_payload()
    if corruption == "moment_shape":
        first_state = next(iter(payload["optimizer_state"]["state"].values()))
        first_state["exp_avg"] = torch.zeros(7)
        expected = "exp_avg layout differs"
    else:
        payload["scheduler_state"]["_last_lr"][0] *= 2.0
        expected = "learning rates are not exact"

    destination_model = _TinyDynamicSetModel()
    destination_adapter = _SyntheticObjectiveAdapter()
    destination = _trainer(model=destination_model, adapter=destination_adapter)
    before = destination.checkpoint_payload()

    with pytest.raises(ValueError, match=expected):
        destination.load_checkpoint_payload(payload)

    after = destination.checkpoint_payload()
    _assert_tensor_tree_equal(after["model_state"], before["model_state"])
    _assert_tensor_tree_equal(after["optimizer_state"], before["optimizer_state"])
    assert after["scheduler_state"] == before["scheduler_state"]
    assert after["adapter_state"] == before["adapter_state"]
    assert after["next_sample_state"] == before["next_sample_state"]


def test_validation_hook_runs_only_on_the_exact_512_update_boundary() -> None:
    validation_steps: list[int] = []

    def validate(_model: nn.Module, step: int) -> Mapping[str, float]:
        validation_steps.append(step)
        return {"score": float(step)}

    trainer = _trainer(validation_hook=validate)
    assert trainer.run_update().validation is None
    # Fast-forward only the synthetic fixture's counters/moments; the next
    # real update still executes all six B4 microbatches and AdamW exactly.
    trainer._completed_updates = 511
    for state in trainer.optimizer.state.values():
        state["step"].fill_(511)
    multiplier = learning_rate_multiplier(511, trainer.optimizer_config)
    expected_rates = [
        trainer.optimizer_config.perception_learning_rate * multiplier,
        trainer.optimizer_config.relation_learning_rate * multiplier,
    ]
    for group, learning_rate in zip(
        trainer.optimizer.param_groups,
        expected_rates,
        strict=True,
    ):
        group["lr"] = learning_rate
    trainer.scheduler.last_epoch = 511
    trainer.scheduler._step_count = 512
    trainer.scheduler._last_lr = expected_rates

    boundary = trainer.run_update()
    after = trainer.run_update()

    assert boundary.validation is not None
    assert boundary.validation.completed_updates == 512
    assert after.validation is None
    assert validation_steps == [512]


def test_disposable_screen_manifest_has_frozen_all_regime_certificate() -> None:
    assert len(SCREEN_ROWS) == SCREEN_EXAMPLE_COUNT == 64
    assert SCREEN_MANIFEST_SHA256 == (
        "b6c3deb2c8fb2fa9d58d5cf29d38f695de3bc93430a02dd47cdcca38add3db0b"
    )
    assert canonical_sha256([asdict(row) for row in SCREEN_ROWS]) == SCREEN_MANIFEST_SHA256
    for index, cell in enumerate(PHYSICAL_CELLS):
        rows = tuple(row for row in SCREEN_ROWS if row.cell_index == index)
        assert {row.known_action for row in rows} == {False, True}
        if cell.contact:
            assert {(row.known_action, row.contact_origin) for row in rows} == {
                (False, "natural"),
                (True, "action_induced"),
            }
            assert {row.contact_geometry for row in rows} == {"head_on", "glancing"}
        if cell.dynamic_membership:
            assert {row.lifecycle_schedule for row in rows} == {
                "birth",
                "removal",
                "remove_then_birth",
            }


def test_disposable_screen_requires_exactly_64_examples_and_at_most_512_updates() -> None:
    trainer = _trainer(training_rows=SCREEN_ROWS, screen_only=True)

    def screen(_model: nn.Module, step: int) -> DynamicSetScreenSnapshot:
        return DynamicSetScreenSnapshot(
            example_count=SCREEN_EXAMPLE_COUNT,
            optimization_objective=-0.446 if step == 0 else -1.40,
            proposal_f1=0.95,
            collision_f1=0.95,
        )

    result = trainer.run_disposable_screen(updates=1, evaluation_hook=screen)
    assert result.passed
    assert result.example_count == 64
    assert result.metrics.example_count == 64
    assert result.metrics.completed_updates == 1

    with pytest.raises(ValueError, match=r"\[1,512\]"):
        _trainer(training_rows=SCREEN_ROWS, screen_only=True).run_disposable_screen(
            updates=513,
            evaluation_hook=screen,
        )

    def wrong_count(_model: nn.Module, _step: int) -> DynamicSetScreenSnapshot:
        return DynamicSetScreenSnapshot(
            example_count=63,
            optimization_objective=-0.446,
            proposal_f1=0.95,
            collision_f1=0.95,
        )

    with pytest.raises(ValueError, match="exactly 64"):
        _trainer(training_rows=SCREEN_ROWS, screen_only=True).run_disposable_screen(
            updates=1,
            evaluation_hook=wrong_count,
        )


def test_disposable_screen_requires_dedicated_frozen_rows_and_no_development_hook() -> None:
    def screen(_model: nn.Module, step: int) -> DynamicSetScreenSnapshot:
        return DynamicSetScreenSnapshot(
            example_count=SCREEN_EXAMPLE_COUNT,
            optimization_objective=-0.446 if step == 0 else -0.57,
            proposal_f1=0.95,
            collision_f1=0.95,
        )

    with pytest.raises(RuntimeError, match="dedicated exact-64-row"):
        _trainer().run_disposable_screen(updates=1, evaluation_hook=screen)
    with pytest.raises(ValueError, match="exact frozen 64-row cover"):
        _trainer(training_rows=SCREEN_ROWS[:-1], screen_only=True)

    development_calls: list[int] = []
    trainer = _trainer(
        training_rows=SCREEN_ROWS,
        screen_only=True,
        validation_hook=lambda _model, step: development_calls.append(step) or {},
    )
    with pytest.raises(RuntimeError, match="development validation hook"):
        trainer.run_disposable_screen(updates=1, evaluation_hook=screen)
    assert development_calls == []


def test_online_world_model_factory_resolves_the_set_proposer_and_edge_relation_only() -> None:
    config = load_config(PROFILE)
    model = OnlineWorldModel.from_config(config)
    adapter = _SyntheticObjectiveAdapter()

    trainer = DynamicSetTrainer.from_online_world_model(
        model=model,
        training_rows=TRAINING_ROWS,
        objective_adapter=adapter,
        resolved_config=config,
        source_provenance=SOURCE,
        schedule_seed=1_610,
        materializer=_materializer(),
        test_only_synthetic_manifest=True,
    )

    assert trainer.perception_owner is model.observation_modules["rgbd"].set_proposer
    assert trainer.relation_owner is model.dynamics.interactions.edge_network
    assert trainer.capacity.perception_parameters == 20_236
    assert trainer.capacity.relation_parameters == 647
    assert tuple(item.name for item in fields(DynamicSetObjectiveInputs)) == (
        "perception",
        "dynamics",
        "causal_support",
    )


@pytest.mark.parametrize(
    ("section", "name", "value"),
    (
        ("simulator", "gravity", [0.0, -9.81, 0.0]),
        ("simulator", "world_bounds", [[-3.0, 3.0]] * 3),
        ("model.state", "fast_log_variance_min", -12.0),
        ("model.dynamics", "node_acceleration_enabled", True),
        ("model.dynamics", "process_noise_position", 1.0e-4),
        ("model.dynamics", "process_noise_velocity", 2.0e-3),
        ("model.filter", "min_log_variance", -12.0),
        ("model.rgbd", "set_log_variance_residual_limit", 4.0),
        ("model.rgbd", "temporal_min_samples", 2),
        ("model.rgbd", "temporal_velocity_variance_floor", 1.0e-6),
        ("training", "train_episodes", 65_999),
    ),
)
def test_online_factory_rejects_nearby_but_non_specification_profiles(
    section: str,
    name: str,
    value: object,
) -> None:
    config = load_config(PROFILE)
    model = OnlineWorldModel.from_config(config)
    payload = config.to_dict()
    owner: dict[str, Any] = payload
    for component in section.split("."):
        child = owner[component]
        assert isinstance(child, dict)
        owner = child
    owner[name] = value

    with pytest.raises(ValueError, match="specification-1.61 CPU set profile"):
        DynamicSetTrainer.from_online_world_model(
            model=model,
            training_rows=TRAINING_ROWS,
            objective_adapter=_SyntheticObjectiveAdapter(),
            resolved_config=payload,
            source_provenance=SOURCE,
            schedule_seed=1_610,
            materializer=_materializer(),
            test_only_synthetic_manifest=True,
        )
