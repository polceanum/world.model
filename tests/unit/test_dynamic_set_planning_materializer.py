from __future__ import annotations

import math
from dataclasses import asdict, fields, replace
from functools import cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import world_model.training.dynamic_set_planning_materializer as planning_materializer
from world_model.belief import BeliefFactory, MotionMode
from world_model.dynamics import AnalyticFreeMotionDynamics
from world_model.observations import ObservationPacket
from world_model.observations.rgbd.temporal import RGBDTemporalPositionHistory
from world_model.runtime.state import RuntimeState, runtime_stream_key
from world_model.training.dynamic_set_planning import PlanningTaskOutcome
from world_model.training.dynamic_set_planning_materializer import (
    PLANNING_DYNAMIC_BIRTH_FRAME,
    PLANNING_HISTORY_FRAMES,
    PLANNING_HISTORY_SENSOR_ID,
    PlanningPairedPopulationEvaluationResult,
    PlanningPopulationEvaluationConfig,
    PlanningTaskMaterialization,
    evaluate_authorized_paired_planning_rows,
    evaluate_authorized_planning_rows,
    evaluate_development_planning_materializations,
    evaluate_paired_development_planning_materializations,
    evaluate_planning_task_materialization,
    infer_and_bind_public_planning_task,
    materialize_planning_task,
    planning_error,
    validate_paired_planning_population_evaluation_result,
    validate_planning_population_evaluation_result,
    validate_public_planning_history,
)
from world_model.training.dynamic_set_protocol import PlanningManifestRow, canonical_sha256
from world_model.training.qualification_core import (
    OrderedSplitLedger,
    QualificationArtifactDirectory,
)


def _row(
    *,
    ordinal: int,
    object_count: int,
    candidate_count: int,
    previously_dynamic: bool,
    candidate_induced_contact: bool,
    split: str = "development",
) -> PlanningManifestRow:
    return PlanningManifestRow(
        split=split,  # type: ignore[arg-type]
        ordinal=ordinal,
        seed={
            "development": 76_900_000,
            "selector": 77_900_000,
            "confirmation": 78_900_000,
            "final_test": 79_900_000,
            "compositional_ood": 80_900_000,
        }[split]
        + ordinal,
        object_count=object_count,
        previously_dynamic=previously_dynamic,
        candidate_induced_contact=candidate_induced_contact,
        target_rank=(ordinal // 3) % object_count,
        action_time_stratum=ordinal % 4,
        camera_stratum=ordinal % 8,
        goal_direction=ordinal % 6,
        candidate_count=candidate_count,  # type: ignore[arg-type]
        minimum_normalized_winner_margin=0.05,
        distribution=("compositional_ood" if split == "compositional_ood" else "in_distribution"),
    )


@cache
def _materialization(row: PlanningManifestRow) -> PlanningTaskMaterialization:
    if row.split in {"selector", "confirmation", "final_test", "compositional_ood"}:
        capability = planning_materializer._mint_protected_planning_materialization_capability(
            split=row.split,
            rows=(row,),
            protocol_sha256="a" * 64,
            manifest_sha256=canonical_sha256([asdict(row)]),
            permit_index={
                "selector": 0,
                "confirmation": 1,
                "final_test": 2,
                "compositional_ood": 3,
            }[row.split],
            permit_nonce="b" * 64,
        )
        result = planning_materializer._materialize_protected_planning_task(
            row,
            capability=capability,
        )
        capability.finish()
        return result
    return materialize_planning_task(row)


def _component_masks(mask: torch.Tensor) -> list[torch.Tensor]:
    """Small observable-only 4-connected component finder for the test runtime."""

    remaining = mask.clone()
    height, width = remaining.shape
    components: list[torch.Tensor] = []
    while bool(remaining.any()):
        start = torch.nonzero(remaining, as_tuple=False)[0].tolist()
        stack = [(int(start[0]), int(start[1]))]
        remaining[start[0], start[1]] = False
        pixels: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            pixels.append((y, x))
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < height and 0 <= xx < width and bool(remaining[yy, xx]):
                    remaining[yy, xx] = False
                    stack.append((yy, xx))
        component = torch.zeros_like(mask)
        for y, x in pixels:
            component[y, x] = True
        components.append(component)
    return components


def _appearance(rgb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(rgb.dtype)
    epsilon = torch.finfo(rgb.dtype).eps
    mass = weight.sum()
    mean_rgb = torch.einsum("hw,chw->c", weight, rgb) / mass
    second_rgb = torch.einsum("hw,chw->c", weight, rgb.square()) / mass
    std_rgb = (second_rgb - mean_rgb.square() + epsilon).clamp_min(epsilon).sqrt()
    intensity = rgb.mean(dim=0)
    mean_intensity = torch.einsum("hw,hw->", weight, intensity) / mass
    second_intensity = torch.einsum("hw,hw->", weight, intensity.square()) / mass
    std_intensity = (second_intensity - mean_intensity.square() + epsilon).clamp_min(epsilon).sqrt()
    return F.normalize(
        torch.cat((mean_rgb, std_rgb, mean_intensity[None], std_intensity[None])),
        dim=0,
    )


class _ObservableHistoryModel(nn.Module):
    """A public-only checkpoint double with the real temporal-history class."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.dynamics = AnalyticFreeMotionDynamics()
        self.factory = BeliefFactory(max_objects=6, appearance_dim=8, initial_radius=0.21)
        self.state = RuntimeState()
        self._tracks: dict[int, tuple[torch.Tensor, int, int]] = {}
        self._pending: dict[tuple[float, ...], int] = {}
        self._next_id = 700

    @property
    def belief(self):
        return self.state.belief

    def reset(self, batch_size: int = 1) -> None:
        assert batch_size == 1
        self.state = RuntimeState(batch_size=1)
        self._tracks.clear()
        self._pending.clear()
        self._next_id = 700

    @staticmethod
    def _world_position(
        depth: torch.Tensor,
        mask: torch.Tensor,
        intrinsics: torch.Tensor,
        world_from_camera: torch.Tensor,
    ) -> torch.Tensor:
        pixels = torch.nonzero(mask, as_tuple=False).to(torch.float32)
        y, x = pixels.mean(dim=0)
        surface = depth[mask].median()
        ray = torch.stack(
            (
                (x - intrinsics[0, 2]) / intrinsics[0, 0],
                (y - intrinsics[1, 2]) / intrinsics[1, 1],
                torch.ones_like(surface),
            )
        )
        camera_position = surface * ray + 0.21 * ray / torch.linalg.vector_norm(ray)
        homogeneous = torch.cat((camera_position, torch.ones(1)))
        return (world_from_camera @ homogeneous)[:3]

    def ingest(self, packet: ObservationPacket):
        assert packet.modality == "rgbd"
        assert set(packet.payload) == {"rgb", "depth"}
        assert set(packet.calibration) == {"world_from_camera", "intrinsics"}
        assert set(packet.metadata) == {"image_size"}
        rgb = packet.payload["rgb"][0]
        depth = packet.payload["depth"][0, 0]
        intrinsics = packet.calibration["intrinsics"][0]
        world_from_camera = packet.calibration["world_from_camera"][0]
        observed = []
        for mask in _component_masks(depth > 0.0):
            descriptor = _appearance(rgb, mask)
            position = self._world_position(depth, mask, intrinsics, world_from_camera)
            observed.append((descriptor, position))

        matched_ids: set[int] = set()
        current: list[tuple[int, torch.Tensor, torch.Tensor, int]] = []
        for descriptor, position in observed:
            candidates = [
                (float(F.cosine_similarity(descriptor, value[0], dim=0)), object_id)
                for object_id, value in self._tracks.items()
                if object_id not in matched_ids
            ]
            similarity, object_id = max(candidates, default=(-1.0, -1))
            key = tuple(round(float(value), 6) for value in descriptor)
            if similarity >= 0.99:
                old_descriptor, age, _misses = self._tracks[object_id]
                self._tracks[object_id] = (old_descriptor, age + 1, 0)
                matched_ids.add(object_id)
                current.append((object_id, descriptor, position, age + 1))
            else:
                count = self._pending.get(key, 0) + 1
                self._pending[key] = count
                if count >= 2:
                    object_id = self._next_id
                    self._next_id += 1
                    self._tracks[object_id] = (descriptor.clone(), count, 0)
                    matched_ids.add(object_id)
                    current.append((object_id, descriptor, position, count))
        for object_id in list(self._tracks):
            if object_id not in matched_ids:
                descriptor, age, misses = self._tracks[object_id]
                misses += 1
                if misses >= 2:
                    del self._tracks[object_id]
                else:
                    self._tracks[object_id] = (descriptor, age, misses)

        current.sort(key=lambda item: item[0])
        belief = self.factory.create(
            timestamp=packet.timestamp,
            gravity=(0.0, 0.0, 0.0),
            intrinsics=intrinsics.unsqueeze(0),
            world_from_camera=world_from_camera.unsqueeze(0),
            active_modalities=("rgbd",),
        )
        objects = belief.objects.clone()
        for slot, (object_id, descriptor, position, age) in enumerate(current):
            objects.active[0, slot] = True
            objects.object_id[0, slot] = object_id
            objects.position[0, slot] = position
            objects.appearance[0, slot] = descriptor
            objects.age_steps[0, slot] = age
            objects.existence_logit[0, slot] = 8.0
            objects.visibility_logit[0, slot] = 8.0
            objects.motion_mode_logits[0, slot].zero_()
            objects.motion_mode_logits[0, slot, MotionMode.FREE] = 8.0
        belief = replace(
            belief,
            objects=objects,
            next_object_id=torch.tensor([self._next_id], dtype=torch.int64),
            metadata={},
        ).validate()
        stream_key = runtime_stream_key("rgbd", PLANNING_HISTORY_SENSOR_ID)
        temporal = self.state.temporal_histories.get(stream_key)
        if not isinstance(temporal, RGBDTemporalPositionHistory):
            temporal = RGBDTemporalPositionHistory.empty(
                object_ids=belief.objects.object_id,
                active_mask=belief.objects.active,
                history_size=16,
                dtype=torch.float32,
            )
        temporal = temporal.append(
            object_ids=belief.objects.object_id,
            active_mask=belief.objects.active,
            append_mask=belief.objects.active,
            timestamp=belief.timestamp,
            positions=belief.objects.position,
            valid_mask=belief.objects.active,
            minimum_dt=1.0e-4,
        )
        self.state.temporal_histories[stream_key] = temporal
        self.state.belief = belief
        self.state.ingest_count += 1
        return belief


_FACTOR_ROWS = tuple(
    _row(
        ordinal=100 + 2 * (object_count - 1) + k_index,
        object_count=object_count,
        candidate_count=candidate_count,
        previously_dynamic=bool(k_index),
        candidate_induced_contact=bool(k_index and object_count > 1),
    )
    for object_count in range(1, 7)
    for k_index, candidate_count in enumerate((8, 32))
)

_FAST_PAIRED_CONFIG = PlanningPopulationEvaluationConfig(
    require_complete_slices=False,
    evaluate_invariants=False,
    latency_warmup_runs=0,
    latency_measured_runs=1,
)


def test_invariant_and_latency_populations_cover_every_required_public_stratum() -> None:
    rows = (
        replace(
            _row(
                ordinal=300,
                object_count=6,
                candidate_count=8,
                previously_dynamic=False,
                candidate_induced_contact=False,
            ),
            target_rank=0,
            action_time_stratum=0,
            camera_stratum=0,
            goal_direction=0,
        ),
        replace(
            _row(
                ordinal=301,
                object_count=6,
                candidate_count=8,
                previously_dynamic=True,
                candidate_induced_contact=True,
            ),
            target_rank=5,
            action_time_stratum=3,
            camera_stratum=7,
            goal_direction=5,
        ),
        _row(
            ordinal=302,
            object_count=6,
            candidate_count=32,
            previously_dynamic=True,
            candidate_induced_contact=True,
        ),
        _row(
            ordinal=303,
            object_count=5,
            candidate_count=8,
            previously_dynamic=False,
            candidate_induced_contact=False,
        ),
    )
    tasks = tuple(SimpleNamespace(row=row) for row in rows)

    features = set().union(*(planning_materializer._invariant_features(row) for row in rows))
    covered = planning_materializer._stratified_invariant_cover(tasks)
    covered_features = set().union(
        *(planning_materializer._invariant_features(task.row) for task in covered)
    )
    assert features == covered_features
    assert {
        ("target_rank", 5),
        ("action_time_stratum", 3),
        ("camera_stratum", 7),
        ("goal_direction", 5),
    } <= features
    assert (
        planning_materializer._state_only_latency_population(
            tasks,
            candidate_count=8,
        )
        == tasks[:2]
    )
    assert (
        planning_materializer._state_only_latency_population(
            tasks,
            candidate_count=32,
        )
        == tasks[2:3]
    )


@pytest.mark.parametrize("row", _FACTOR_ROWS)
def test_real_materialization_covers_all_cardinalities_and_k_strata(row) -> None:
    item = _materialization(row)
    history = validate_public_planning_history(item.public_history)
    counts = [len(_component_masks(frame.depth[0] > 0.0)) for frame in history.frames]
    expected_before = row.object_count - int(row.previously_dynamic)
    assert counts[:PLANNING_DYNAMIC_BIRTH_FRAME] == [expected_before] * 4
    assert counts[PLANNING_DYNAMIC_BIRTH_FRAME:] == [row.object_count] * (
        PLANNING_HISTORY_FRAMES - PLANNING_DYNAMIC_BIRTH_FRAME
    )
    assert item.template.appearance_handle.shape == (1, 8)
    assert float(history.frames[0].intrinsics[0, 0]) > 100.0
    assert item.private_oracle.candidate_costs.shape == (row.candidate_count,)
    assert item.private_oracle.certificate.normalized_winner_margin >= 0.05
    assert item.private_oracle.certificate.winner_succeeds
    assert bool(item.private_oracle.candidate_induced_contact.any()) is (
        row.candidate_induced_contact
    )
    assert item.attempt_count <= 8


def _mapping_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for child in value.values() for key in _mapping_keys(child)}
    if isinstance(value, (tuple, list)):
        return {key for child in value for key in _mapping_keys(child)}
    return set()


def test_public_history_and_template_expose_no_truth_id_or_slot() -> None:
    item = _materialization(_FACTOR_ROWS[-1])
    frame_fields = {field.name for field in fields(item.public_history.frames[0])}
    assert frame_fields == {
        "frame_index",
        "timestamp",
        "rgb",
        "depth",
        "world_from_camera",
        "intrinsics",
    }
    public_keys = _mapping_keys(asdict(item.public_history)) | _mapping_keys(asdict(item.template))
    assert "object_id" not in public_keys
    assert "target_slot" not in public_keys
    assert "simulator_id" not in public_keys
    packet = item.public_history.frames[-1].packet()
    assert set(packet.payload) == {"rgb", "depth"}
    assert set(packet.calibration) == {"world_from_camera", "intrinsics"}
    assert packet.metadata == {"image_size": (64, 64)}


def test_public_history_tamper_and_determinism() -> None:
    row = _FACTOR_ROWS[0]
    first = _materialization(row)
    second = materialize_planning_task(row)
    assert first.materialization_sha256 == second.materialization_sha256
    assert first.public_history.history_sha256 == second.public_history.history_sha256
    frames = list(first.public_history.frames)
    rgb = frames[0].rgb.clone()
    rgb[0, 0, 0] += 0.01
    frames[0] = replace(frames[0], rgb=rgb)
    tampered = replace(first.public_history, frames=tuple(frames))
    with pytest.raises(ValueError, match="digest"):
        validate_public_planning_history(tampered)


@pytest.mark.parametrize("previously_dynamic", [False, True])
def test_public_checkpoint_binding_uses_real_temporal_history(previously_dynamic: bool) -> None:
    row = _row(
        ordinal=400 + int(previously_dynamic),
        object_count=3,
        candidate_count=8,
        previously_dynamic=previously_dynamic,
        candidate_induced_contact=False,
    )
    item = _materialization(row)
    model = _ObservableHistoryModel()
    task = infer_and_bind_public_planning_task(model, item.public_history, item.template)
    assert task.history_evidence.previously_dynamic is previously_dynamic
    assert torch.all(task.history_evidence.valid_sample_count[task.frozen_active_mask] == 16)
    assert int(task.frozen_active_mask.sum()) == row.object_count
    assert task.target_object_id.item() >= 700


def test_unresolved_maturity_becomes_outcome_without_opening_oracle() -> None:
    item = _materialization(_FACTOR_ROWS[0])

    class _NoTrackModel(_ObservableHistoryModel):
        def ingest(self, packet):
            belief = self.factory.create(
                timestamp=packet.timestamp,
                gravity=(0.0, 0.0, 0.0),
                active_modalities=("rgbd",),
            )
            self.state.belief = belief
            return belief

    poisoned_oracle = replace(
        item.private_oracle,
        candidate_costs=torch.full_like(item.private_oracle.candidate_costs, torch.nan),
    )
    outcome = evaluate_planning_task_materialization(
        _NoTrackModel(), replace(item, private_oracle=poisoned_oracle)
    )
    assert isinstance(outcome, PlanningTaskOutcome)
    assert not outcome.handle_resolved
    assert outcome.evaluation is None
    assert "cardinality" in outcome.failure_reason


def test_real_public_binding_and_private_oracle_evaluation() -> None:
    item = _materialization(_FACTOR_ROWS[4])
    outcome = evaluate_planning_task_materialization(_ObservableHistoryModel(), item)
    assert outcome.handle_resolved
    assert outcome.evaluation is not None
    assert outcome.evaluation.serial_vectorized_winner_parity
    assert outcome.evaluation.maximum_cost_difference <= 1.0e-6
    assert math.isfinite(outcome.evaluation.normalized_regret)
    assert 0.0 <= planning_error((outcome,)) <= 1.0


def test_combined_materialization_digest_is_checked_after_model_plans() -> None:
    item = _materialization(_FACTOR_ROWS[2])
    with pytest.raises(ValueError, match="combined digest"):
        evaluate_planning_task_materialization(
            _ObservableHistoryModel(),
            replace(item, materialization_sha256="0" * 64),
        )

    retagged_seed = planning_materializer._candidate_seed(item.row.seed, 1)
    retagged_rejections = ("ValueError: deterministic rejected candidate",)
    retagged = replace(
        item,
        accepted_seed=retagged_seed,
        attempt_count=2,
        rejection_reasons=retagged_rejections,
        materialization_sha256=planning_materializer._materialization_digest(
            item.public_history,
            item.template,
            item.private_oracle,
            retagged_seed,
            2,
            retagged_rejections,
        ),
    )
    with pytest.raises(ValueError, match="combined digest"):
        evaluate_planning_task_materialization(
            _ObservableHistoryModel(),
            replace(
                retagged,
                rejection_reasons=("ValueError: altered rejection audit",),
            ),
        )


def test_compositional_ood_row_has_real_bound_oracle() -> None:
    row = _row(
        ordinal=37,
        object_count=4,
        candidate_count=32,
        previously_dynamic=True,
        candidate_induced_contact=True,
        split="compositional_ood",
    )
    item = _materialization(row)
    assert item.row.distribution == "compositional_ood"
    assert item.private_oracle.certificate.normalized_winner_margin >= 0.05
    assert item.private_oracle.certificate.winner_succeeds
    assert bool(item.private_oracle.candidate_induced_contact.any())


def test_private_oracle_is_independent_of_production_rollout_and_cost_faults(
    monkeypatch,
) -> None:
    import world_model.planning.counterfactual as production_cost
    import world_model.training.dynamic_set_planning as production_evaluator
    from world_model.dynamics import DynamicsModel

    row = _row(
        ordinal=213,
        object_count=2,
        candidate_count=8,
        previously_dynamic=False,
        candidate_induced_contact=True,
    )
    reference = materialize_planning_task(row)

    def production_fault(*_args, **_kwargs):
        raise AssertionError("fault-injected production rollout/cost path was reached")

    monkeypatch.setattr(DynamicsModel, "rollout", production_fault)
    monkeypatch.setattr(production_cost, "plan_counterfactual_actions", production_fault)
    monkeypatch.setattr(production_evaluator, "plan_counterfactual_actions", production_fault)
    repeated = materialize_planning_task(row)

    assert repeated.materialization_sha256 == reference.materialization_sha256
    assert repeated.private_oracle.evidence_sha256 == reference.private_oracle.evidence_sha256
    assert torch.equal(
        repeated.private_oracle.candidate_costs,
        reference.private_oracle.candidate_costs,
    )
    assert torch.equal(
        repeated.private_oracle.candidate_terminal_goal_distance_m,
        reference.private_oracle.candidate_terminal_goal_distance_m,
    )
    assert torch.equal(
        repeated.private_oracle.candidate_induced_contact,
        reference.private_oracle.candidate_induced_contact,
    )


def test_development_population_aggregates_all_slices_and_invariants() -> None:
    result = evaluate_development_planning_materializations(
        _ObservableHistoryModel(),
        (_materialization(row) for row in _FACTOR_ROWS),
        config=PlanningPopulationEvaluationConfig(
            require_complete_slices=True,
            evaluate_invariants=True,
            latency_warmup_runs=0,
            latency_measured_runs=1,
        ),
    )
    assert {
        (item.object_count, item.candidate_count, item.distribution)
        for item in result.reduction.slices
    } == {
        (object_count, candidate_count, "in_distribution")
        for object_count in range(1, 7)
        for candidate_count in (8, 32)
    }
    assert result.invariants.serial_vectorized_winner_parity
    assert result.invariants.maximum_cost_difference <= 1.0e-6
    assert result.invariants.pre_action_invariance
    assert result.invariants.exactly_once_impulse
    assert result.invariants.action_target_isolation
    assert result.invariants.conservation
    assert result.invariants.batch_independence
    assert result.invariants.source_belief_unchanged
    assert math.isfinite(result.invariants.latency_k8_seconds)
    assert math.isfinite(result.invariants.latency_k32_seconds)
    assert result.invariants.latency_k8_seconds >= 0.0
    assert result.invariants.latency_k32_seconds >= 0.0


def test_paired_development_materializes_once_and_aligns_identical_models(
    monkeypatch,
) -> None:
    row = _FACTOR_ROWS[0]
    item = _materialization(row)
    calls = 0

    def counting_materializer(requested: PlanningManifestRow) -> PlanningTaskMaterialization:
        nonlocal calls
        calls += 1
        assert requested == row
        return item

    monkeypatch.setattr(
        planning_materializer,
        "materialize_planning_task",
        counting_materializer,
    )
    result = evaluate_paired_development_planning_materializations(
        _ObservableHistoryModel(),
        _ObservableHistoryModel(),
        planning_materializer.iter_planning_task_materializations((row,)),
        config=_FAST_PAIRED_CONFIG,
    )

    assert isinstance(result, PlanningPairedPopulationEvaluationResult)
    assert calls == 1
    assert result.candidate.outcomes == result.reference.outcomes
    assert result.candidate.result_sha256 == result.reference.result_sha256
    assert result.candidate.population_binding == result.reference.population_binding
    assert len(result.provenance) == 1
    assert result.provenance[0].row_sha256 == canonical_sha256(asdict(row))
    assert result.provenance[0].public_history_sha256 == item.public_history.history_sha256
    assert result.provenance[0].public_template_sha256 == item.template.template_sha256
    assert result.provenance[0].public_materialization_sha256 != item.materialization_sha256
    assert result.provenance[0].private_oracle_sha256 == item.private_oracle.evidence_sha256
    assert result.provenance[0].complete_materialization_sha256 == item.materialization_sha256
    with pytest.raises(ValueError, match="combined digest"):
        planning_materializer.PlanningPairProvenance.create(
            replace(
                item,
                private_oracle=replace(
                    item.private_oracle,
                    evidence_sha256="f" * 64,
                ),
                materialization_sha256="0" * 64,
            )
        )
    assert validate_paired_planning_population_evaluation_result(result) is result

    tampered = replace(
        result,
        provenance=(replace(result.provenance[0], public_history_sha256="f" * 64),),
    )
    with pytest.raises(ValueError, match="provenance digest differs"):
        validate_paired_planning_population_evaluation_result(tampered)


def test_paired_private_oracle_opens_after_both_serial_and_vectorized_plans(
    monkeypatch,
) -> None:
    import world_model.training.dynamic_set_planning as planning

    item = _materialization(_FACTOR_ROWS[2])
    candidate = _ObservableHistoryModel()
    reference = _ObservableHistoryModel()
    plan_counts = {"candidate": 0, "reference": 0}
    original_plan = planning.plan_counterfactual_actions
    original_validate_oracle = planning._validate_private_oracle
    oracle_open_counts: list[tuple[int, int]] = []

    def counted_plan(dynamics, *args, **kwargs):
        if dynamics is candidate.dynamics:
            plan_counts["candidate"] += 1
        elif dynamics is reference.dynamics:
            plan_counts["reference"] += 1
        else:
            raise AssertionError("an unexpected dynamics object reached paired planning")
        return original_plan(dynamics, *args, **kwargs)

    def guarded_oracle(task, oracle):
        oracle_open_counts.append((plan_counts["candidate"], plan_counts["reference"]))
        assert candidate.state.ingest_count == PLANNING_HISTORY_FRAMES
        assert reference.state.ingest_count == PLANNING_HISTORY_FRAMES
        assert oracle_open_counts[-1] == (2, 2)
        return original_validate_oracle(task, oracle)

    monkeypatch.setattr(planning, "plan_counterfactual_actions", counted_plan)
    monkeypatch.setattr(planning, "_validate_private_oracle", guarded_oracle)
    result = evaluate_paired_development_planning_materializations(
        candidate,
        reference,
        (item,),
        config=_FAST_PAIRED_CONFIG,
    )

    assert oracle_open_counts == [(2, 2)]
    assert result.candidate.outcomes[0].handle_resolved
    assert result.reference.outcomes[0].handle_resolved


def test_paired_unresolved_models_never_open_private_oracle(monkeypatch) -> None:
    import world_model.training.dynamic_set_planning as planning

    item = _materialization(_FACTOR_ROWS[0])

    class _NoTrackModel(_ObservableHistoryModel):
        def ingest(self, packet):
            belief = self.factory.create(
                timestamp=packet.timestamp,
                gravity=(0.0, 0.0, 0.0),
                active_modalities=("rgbd",),
            )
            self.state.belief = belief
            return belief

    def forbidden_oracle(*_args, **_kwargs):
        raise AssertionError("private oracle was opened for an unresolved pair")

    monkeypatch.setattr(planning, "_validate_private_oracle", forbidden_oracle)
    poisoned_oracle = replace(
        item.private_oracle,
        candidate_costs=torch.full_like(item.private_oracle.candidate_costs, torch.nan),
    )
    result = evaluate_paired_development_planning_materializations(
        _NoTrackModel(),
        _NoTrackModel(),
        (replace(item, private_oracle=poisoned_oracle),),
        config=_FAST_PAIRED_CONFIG,
    )

    assert not result.candidate.outcomes[0].handle_resolved
    assert not result.reference.outcomes[0].handle_resolved
    assert result.candidate.outcomes == result.reference.outcomes


def test_paired_evaluator_rejects_public_evidence_mutation(monkeypatch) -> None:
    item = _materialization(_FACTOR_ROWS[0])
    frames = list(item.public_history.frames)
    frames[0] = replace(frames[0], rgb=frames[0].rgb.clone())
    isolated = replace(
        item,
        public_history=replace(item.public_history, frames=tuple(frames)),
    )
    original = planning_materializer.infer_and_bind_public_planning_task
    calls = 0

    def mutating_inference(model, public_history, template):
        nonlocal calls
        calls += 1
        task = original(model, public_history, template)
        if calls == 1:
            public_history.frames[0].rgb.add_(0.01)
        return task

    monkeypatch.setattr(
        planning_materializer,
        "infer_and_bind_public_planning_task",
        mutating_inference,
    )
    with pytest.raises(
        planning_materializer.PlanningTaskMaterializationError,
        match="candidate mutated paired public planning evidence",
    ):
        evaluate_paired_development_planning_materializations(
            _ObservableHistoryModel(),
            _ObservableHistoryModel(),
            (isolated,),
            config=_FAST_PAIRED_CONFIG,
        )


def test_public_planning_materializers_reject_protected_before_oracle_generation(
    monkeypatch,
) -> None:
    row = _row(
        ordinal=0,
        object_count=1,
        candidate_count=8,
        previously_dynamic=False,
        candidate_induced_contact=False,
        split="selector",
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("protected oracle was generated before authorization")

    monkeypatch.setattr(planning_materializer, "_materialize_planning_task", forbidden)
    with pytest.raises(PermissionError, match="governed evaluator"):
        materialize_planning_task(row)
    with pytest.raises(PermissionError, match="governed evaluator"):
        tuple(planning_materializer.iter_planning_task_materializations((row,)))


def test_protected_planning_capability_is_ordered_split_bound_and_one_shot(
    monkeypatch,
) -> None:
    rows = tuple(
        _row(
            ordinal=index,
            object_count=1,
            candidate_count=8,
            previously_dynamic=False,
            candidate_induced_contact=False,
            split="selector",
        )
        for index in range(2)
    )
    generated: list[int] = []

    def counted(row, **kwargs):
        del kwargs
        generated.append(row.ordinal)
        return object()

    monkeypatch.setattr(planning_materializer, "_materialize_planning_task_core", counted)

    def capability():
        return planning_materializer._mint_protected_planning_materialization_capability(
            split="selector",
            rows=rows,
            protocol_sha256="a" * 64,
            manifest_sha256="b" * 64,
            permit_index=0,
            permit_nonce="c" * 64,
        )

    wrong_order = capability()
    with pytest.raises(PermissionError, match="row order"):
        planning_materializer._materialize_protected_planning_task(rows[1], capability=wrong_order)
    with pytest.raises(PermissionError, match="closed"):
        planning_materializer._materialize_protected_planning_task(rows[0], capability=wrong_order)
    assert generated == []

    wrong_split = capability()
    with pytest.raises(PermissionError, match="public row"):
        planning_materializer._materialize_protected_planning_task(
            _FACTOR_ROWS[0], capability=wrong_split
        )
    assert generated == []

    one_row = planning_materializer._mint_protected_planning_materialization_capability(
        split="selector",
        rows=rows[:1],
        protocol_sha256="a" * 64,
        manifest_sha256="b" * 64,
        permit_index=0,
        permit_nonce="c" * 64,
    )
    planning_materializer._materialize_protected_planning_task(rows[0], capability=one_row)
    with pytest.raises(PermissionError, match="row order"):
        planning_materializer._materialize_protected_planning_task(rows[0], capability=one_row)
    assert generated == [0]


def test_development_stream_rejects_protected_before_inference() -> None:
    protected = _materialization(
        _row(
            ordinal=0,
            object_count=1,
            candidate_count=8,
            previously_dynamic=False,
            candidate_induced_contact=False,
            split="selector",
        )
    )

    with pytest.raises(PermissionError, match="claimed capability"):
        planning_materializer._materialize_planning_task(protected.row)
    with pytest.raises(PermissionError, match="governed"):
        evaluate_development_planning_materializations(
            _ObservableHistoryModel(),
            (protected,),
            config=PlanningPopulationEvaluationConfig(
                require_complete_slices=False,
                evaluate_invariants=False,
            ),
        )


def _authorized_fixture(monkeypatch, tmp_path: Path):
    rows = tuple(
        _row(
            ordinal=index,
            object_count=index // 2 + 1,
            candidate_count=8 if index % 2 == 0 else 32,
            previously_dynamic=bool(index % 2),
            candidate_induced_contact=bool(index % 2 and index // 2 + 1 > 1),
            split="selector",
        )
        for index in range(12)
    )
    digest = canonical_sha256([asdict(row) for row in rows])
    monkeypatch.setitem(planning_materializer.PLANNING_SPLIT_SIZES, "selector", len(rows))
    monkeypatch.setitem(planning_materializer.FROZEN_PLANNING_MANIFEST_SHA256, "selector", digest)
    artifacts = QualificationArtifactDirectory.create_fresh(
        tmp_path / "protected",
        allowed_names=("ledger.json",),
    )
    ledger = OrderedSplitLedger(
        artifacts,
        artifact_name="ledger.json",
        protocol_sha256="a" * 64,
        split_order=("selector",),
    )
    ledger.create_fresh()
    return rows, digest, ledger, ledger.begin("selector")


def test_authorized_paired_evaluator_rejects_bad_permit_before_materialization(
    monkeypatch,
    tmp_path: Path,
) -> None:
    rows, digest, ledger, permit = _authorized_fixture(monkeypatch, tmp_path)
    materialization_count = 0

    def materialize(*args, **kwargs):
        nonlocal materialization_count
        materialization_count += 1
        raise AssertionError("bad permit reached protected materialization")

    monkeypatch.setattr(planning_materializer, "_materialize_planning_task_core", materialize)

    with pytest.raises(PermissionError, match="index"):
        evaluate_authorized_paired_planning_rows(
            _ObservableHistoryModel(),
            _ObservableHistoryModel(),
            ledger=ledger,
            permit=replace(permit, index=1),
            expected_protocol_sha256="a" * 64,
            expected_manifest_sha256=digest,
            expected_rows=rows,
        )
    assert materialization_count == 0


def test_authorized_paired_planning_rows_materialize_each_task_once(
    monkeypatch,
    tmp_path: Path,
) -> None:
    row = _row(
        ordinal=0,
        object_count=1,
        candidate_count=8,
        previously_dynamic=False,
        candidate_induced_contact=False,
        split="selector",
    )
    item = _materialization(row)
    digest = canonical_sha256([asdict(row)])
    artifacts = QualificationArtifactDirectory.create_fresh(
        tmp_path / "protected",
        allowed_names=("ledger.json",),
    )
    ledger = OrderedSplitLedger(
        artifacts,
        artifact_name="ledger.json",
        protocol_sha256="a" * 64,
        split_order=("selector",),
    )
    ledger.create_fresh()
    permit = ledger.begin("selector")
    calls = 0

    def materialize(requested, **kwargs):
        nonlocal calls
        del kwargs
        calls += 1
        assert requested == row
        return item

    monkeypatch.setitem(planning_materializer.PLANNING_SPLIT_SIZES, "selector", 1)
    monkeypatch.setitem(
        planning_materializer.FROZEN_PLANNING_MANIFEST_SHA256,
        "selector",
        digest,
    )
    monkeypatch.setattr(
        planning_materializer,
        "DEFAULT_PLANNING_POPULATION_EVALUATION_CONFIG",
        _FAST_PAIRED_CONFIG,
    )
    monkeypatch.setattr(planning_materializer, "_materialize_planning_task_core", materialize)

    result = evaluate_authorized_paired_planning_rows(
        _ObservableHistoryModel(),
        _ObservableHistoryModel(),
        ledger=ledger,
        permit=permit,
        expected_protocol_sha256="a" * 64,
        expected_manifest_sha256=digest,
        expected_rows=(row,),
        config=_FAST_PAIRED_CONFIG,
    )

    assert calls == 1
    assert result.candidate.outcomes == result.reference.outcomes
    assert result.candidate.population_binding == result.reference.population_binding
    assert result.provenance[0].row_sha256 == canonical_sha256(asdict(row))
    reopened = OrderedSplitLedger(
        ledger.artifacts,
        artifact_name=ledger.artifact_name,
        protocol_sha256=ledger.protocol_sha256,
        split_order=ledger.split_order,
    )
    with pytest.raises(RuntimeError, match="already claimed"):
        evaluate_authorized_paired_planning_rows(
            _ObservableHistoryModel(),
            _ObservableHistoryModel(),
            ledger=reopened,
            permit=permit,
            expected_protocol_sha256="a" * 64,
            expected_manifest_sha256=digest,
            expected_rows=(row,),
            config=_FAST_PAIRED_CONFIG,
        )
    assert calls == 1


def test_obsolete_protected_planning_stream_is_not_iterated() -> None:
    opened = False

    def stream():
        nonlocal opened
        opened = True
        yield _materialization(_FACTOR_ROWS[0])

    with pytest.raises(PermissionError, match="caller-supplied"):
        planning_materializer.evaluate_authorized_planning_materializations(
            _ObservableHistoryModel(),
            stream(),
        )
    assert not opened


def test_authorized_evaluator_rejects_order_and_permit_tamper(
    monkeypatch,
    tmp_path: Path,
) -> None:
    rows, digest, ledger, permit = _authorized_fixture(monkeypatch, tmp_path)
    materialization_count = 0

    def materialize(*args, **kwargs):
        nonlocal materialization_count
        materialization_count += 1
        raise AssertionError("invalid binding reached protected materialization")

    monkeypatch.setattr(planning_materializer, "_materialize_planning_task_core", materialize)
    with pytest.raises(ValueError, match="digest"):
        evaluate_authorized_planning_rows(
            _ObservableHistoryModel(),
            ledger=ledger,
            permit=permit,
            expected_protocol_sha256="a" * 64,
            expected_manifest_sha256=digest,
            expected_rows=tuple(reversed(rows)),
        )
    with pytest.raises(ValueError, match="count"):
        evaluate_authorized_planning_rows(
            _ObservableHistoryModel(),
            ledger=ledger,
            permit=permit,
            expected_protocol_sha256="a" * 64,
            expected_manifest_sha256=digest,
            expected_rows=rows[:-1],
        )
    with pytest.raises(PermissionError, match="index"):
        evaluate_authorized_planning_rows(
            _ObservableHistoryModel(),
            ledger=ledger,
            permit=replace(permit, index=1),
            expected_protocol_sha256="a" * 64,
            expected_manifest_sha256=digest,
            expected_rows=rows,
        )
    with pytest.raises(ValueError, match="digest"):
        evaluate_authorized_planning_rows(
            _ObservableHistoryModel(),
            ledger=ledger,
            permit=permit,
            expected_protocol_sha256="a" * 64,
            expected_manifest_sha256="c" * 64,
            expected_rows=rows,
        )
    assert materialization_count == 0


def test_authorized_evaluator_returns_bound_nonnegative_score(
    monkeypatch,
    tmp_path: Path,
) -> None:
    rows, digest, ledger, permit = _authorized_fixture(monkeypatch, tmp_path)
    items = {row: _materialization(row) for row in rows}
    materialization_count = 0

    def materialize(row, **kwargs):
        nonlocal materialization_count
        del kwargs
        materialization_count += 1
        return items[row]

    monkeypatch.setattr(planning_materializer, "_materialize_planning_task_core", materialize)
    result = evaluate_authorized_planning_rows(
        _ObservableHistoryModel(),
        ledger=ledger,
        permit=permit,
        expected_protocol_sha256="a" * 64,
        expected_manifest_sha256=digest,
        expected_rows=rows,
    )
    assert result.population_binding.row_count == len(rows)
    assert result.population_binding.manifest_sha256 == digest
    assert 0.0 <= result.planning_error <= 1.0
    assert len(result.outcomes) == len(rows)
    assert materialization_count == len(rows)
    assert len(result.result_sha256) == 64
    assert validate_planning_population_evaluation_result(result) is result

    with pytest.raises(ValueError, match="error differs"):
        validate_planning_population_evaluation_result(
            replace(result, planning_error=result.planning_error + 0.01)
        )
    with pytest.raises(ValueError, match="reduction differs"):
        validate_planning_population_evaluation_result(
            replace(
                result,
                reduction=replace(
                    result.reduction,
                    source_belief_unchanged=not result.reduction.source_belief_unchanged,
                ),
            )
        )
    with pytest.raises(ValueError, match="row count"):
        validate_planning_population_evaluation_result(
            replace(
                result,
                population_binding=replace(
                    result.population_binding,
                    row_count=result.population_binding.row_count + 1,
                ),
            )
        )


def test_protected_evaluator_rejects_weakened_diagnostics(
    monkeypatch,
    tmp_path: Path,
) -> None:
    rows, digest, ledger, permit = _authorized_fixture(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="frozen"):
        evaluate_authorized_planning_rows(
            _ObservableHistoryModel(),
            ledger=ledger,
            permit=permit,
            expected_protocol_sha256="a" * 64,
            expected_manifest_sha256=digest,
            expected_rows=rows,
            config=PlanningPopulationEvaluationConfig(
                require_complete_slices=False,
                evaluate_invariants=False,
            ),
        )
