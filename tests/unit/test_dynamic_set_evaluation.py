from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import world_model.training.dynamic_set_evaluation as evaluation
from world_model.belief import BeliefFactory, BeliefTrajectory, MotionMode
from world_model.observations import MeasurementSet, ObservationPacket
from world_model.runtime.online_world_model import OnlineWorldModel
from world_model.training.dynamic_set_config import load_config
from world_model.training.dynamic_set_evaluation import (
    BeliefFrameTrace,
    DynamicSetCellAccumulator,
    DynamicSetEpisodeTrace,
    DynamicSetEvaluationResult,
    DynamicSetExampleScoreEvidence,
    HorizonTrace,
    ProposalFrameTrace,
    SelectionScoreEvidence,
    SquaredErrorSum,
    run_public_dynamic_set_episode,
    score_dynamic_set_trace,
)
from world_model.training.dynamic_set_gates import (
    ResourceMetrics,
    physical_cell_gate_failures,
)
from world_model.training.dynamic_set_materializer import (
    DynamicSetMaterialization,
    DynamicSetPublicFrame,
    certify_dynamic_set_public_boundary,
    inspect_dynamic_set_public_boundary,
    materialize_dynamic_set_episode,
)
from world_model.training.dynamic_set_protocol import (
    PHYSICAL_CELLS,
    PhysicalCell,
    PhysicalManifestRow,
    physical_manifest,
)
from world_model.training.dynamic_set_scene import DynamicSetSceneCertificate
from world_model.training.qualification_core import (
    OrderedSplitLedger,
    QualificationArtifactDirectory,
    SplitPermit,
    canonical_sha256,
)

_PROFILE = Path(__file__).parents[2] / "configs" / "rgbd_dynamic_set_planning_cpu.yaml"

_PUBLIC_ACTION_HANDLE = torch.tensor(
    [[0.8, 0.1, 0.2, 0.01, 0.01, 0.01, 0.4, 0.01]],
    dtype=torch.float32,
)


def test_collision_probability_transform_preserves_exact_symmetry() -> None:
    logits = torch.tensor(
        [
            [0.0, -4.00001, 3.25],
            [-4.00001, 0.0, -17.75],
            [3.25, -17.75, 0.0],
        ],
        dtype=torch.float32,
    )

    probability = evaluation._exact_symmetric_probability(logits)

    assert torch.equal(probability, probability.transpose(0, 1))
    torch.testing.assert_close(
        torch.triu(probability),
        torch.triu(logits.sigmoid()),
        rtol=0.0,
        atol=0.0,
    )


def test_unresolved_lifecycle_predictions_have_a_total_order() -> None:
    assert evaluation._ordered_lifecycle_predictions(((7, None), (5, None), (7, 3), (7, 1))) == (
        (5, None),
        (7, 1),
        (7, 3),
        (7, None),
    )


def _protected_ledger(
    tmp_path: Path, protocol_sha256: str
) -> tuple[OrderedSplitLedger, SplitPermit]:
    artifacts = QualificationArtifactDirectory.create_fresh(
        tmp_path / "protected",
        allowed_names=("ledger.json",),
    )
    ledger = OrderedSplitLedger(
        artifacts,
        artifact_name="ledger.json",
        protocol_sha256=protocol_sha256,
        split_order=("selector",),
    )
    ledger.create_fresh()
    return ledger, ledger.begin("selector")


def _public_frames(*, action_frame: int | None = None) -> tuple[DynamicSetPublicFrame, ...]:
    result: list[DynamicSetPublicFrame] = []
    intrinsics = torch.tensor(
        [[40.0, 0.0, 31.5], [0.0, 40.0, 31.5], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    for frame_index in range(56):
        observed = torch.zeros(1, dtype=torch.bool)
        timestamps = torch.full((1,), -1.0, dtype=torch.float32)
        appearance_handle = torch.zeros(1, 8, dtype=torch.float32)
        impulses = torch.zeros(1, 3, dtype=torch.float32)
        if frame_index == action_frame:
            observed[0] = True
            timestamps[0] = frame_index / 20.0
            appearance_handle.copy_(_PUBLIC_ACTION_HANDLE)
            impulses[0] = torch.tensor([0.25, 0.0, 0.0])
        result.append(
            DynamicSetPublicFrame(
                frame_index=frame_index,
                timestamp=frame_index / 20.0,
                rgb=torch.zeros(3, 64, 64, dtype=torch.float32),
                depth=torch.ones(1, 64, 64, dtype=torch.float32),
                world_from_camera=torch.eye(4, dtype=torch.float32),
                intrinsics=intrinsics.clone(),
                known_action_observed=observed,
                known_action_timestamp=timestamps,
                known_action_appearance_handle=appearance_handle,
                known_impulse_world=impulses,
            )
        )
    return tuple(result)


def _public_frames_with_boundary(
    materialization: DynamicSetMaterialization,
) -> tuple[tuple[DynamicSetPublicFrame, ...], Any]:
    frames = _public_frames()
    return frames, certify_dynamic_set_public_boundary(materialization, frames)


class _PublicOnlySpyModel(nn.Module):
    def __init__(
        self,
        *,
        replacement_schedule: bool = False,
        appearance_handle: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.factory = BeliefFactory(max_objects=6, appearance_dim=8)
        self._belief = None
        self._last_measurements = None
        self._last_interval_collision_mask = None
        self._last_interval_pair_collision_logits = None
        self.last_direct_velocity_evidence = None
        self.state = None
        self.diagnostics = None
        self.actions: list[Any] = []
        self.packet_count = 0
        self.replacement_schedule = replacement_schedule
        self.appearance_handle = (
            (_PUBLIC_ACTION_HANDLE if appearance_handle is None else appearance_handle)
            .detach()
            .clone()
        )

    @property
    def belief(self) -> Any:
        return self._belief

    @property
    def last_measurements(self) -> MeasurementSet | None:
        return self._last_measurements

    @property
    def last_interval_collision_mask(self) -> torch.Tensor | None:
        return self._last_interval_collision_mask

    @property
    def last_interval_pair_collision_logits(self) -> torch.Tensor | None:
        return self._last_interval_pair_collision_logits

    @property
    def dynamics(self) -> _PublicOnlySpyModel:
        return self

    def reset(self, batch_size: int = 1) -> None:
        assert batch_size == 1
        self._belief = None
        self._last_measurements = None
        self._last_interval_collision_mask = None
        self._last_interval_pair_collision_logits = None
        self.actions.clear()
        self.packet_count = 0

    def ingest(self, packet: ObservationPacket, *, action: Any = None) -> Any:
        assert isinstance(packet, ObservationPacket)
        assert packet.modality == "rgbd"
        assert set(packet.payload) == {"rgb", "depth"}
        assert set(packet.calibration) == {"world_from_camera", "intrinsics"}
        assert set(packet.metadata) == {"image_size"}
        assert not hasattr(packet, "objects")
        self.packet_count += 1
        self._last_interval_collision_mask = torch.zeros(1, 6, dtype=torch.bool)
        self._last_interval_pair_collision_logits = torch.full((1, 6, 6), -4.0)
        if action is not None:
            self.actions.append(action)
        belief = self.factory.create(
            timestamp=packet.timestamp,
            gravity=(0.0, 0.0, 0.0),
        )
        objects = belief.objects
        active = objects.active.clone()
        object_id = objects.object_id.clone()
        position = objects.position.clone()
        velocity = objects.velocity.clone()
        appearance = objects.appearance.clone()
        frame_index = int(packet.frame_id.rsplit(":", maxsplit=1)[1])
        replacement_absent = self.replacement_schedule and 3 <= frame_index < 7
        if not replacement_absent:
            active[0, 0] = True
            object_id[0, 0] = 8_801 if self.replacement_schedule and frame_index >= 7 else 7
            position[0, 0] = torch.tensor([0.2, -0.1, 4.0])
            appearance[0, 0] = self.appearance_handle[0]
        self._belief = belief.replace(
            objects=objects.replace(
                active=active,
                object_id=object_id,
                position=position,
                velocity=velocity,
                appearance=appearance,
            )
        )
        values = torch.zeros(1, 8, 3)
        values[0, 0] = position[0, 0]
        valid = torch.zeros(1, 8, dtype=torch.bool)
        valid[0, 0] = not replacement_absent
        logits = torch.full((1, 8), -12.0)
        if not replacement_absent:
            logits[0, 0] = 12.0
        self._last_measurements = MeasurementSet(
            modality="rgbd",
            sensor_id=packet.sensor_id,
            timestamp=torch.tensor([packet.timestamp], dtype=torch.float32),
            values=values,
            log_variance=torch.zeros(1, 8, 3),
            existence_logits=logits,
            measurement_mask=valid,
            appearance=torch.zeros(1, 8, 8),
            class_logits=None,
            frame_id=packet.frame_id,
            supported_state_fields=("position",),
            auxiliary={"world_position": values, "world_log_variance": torch.zeros_like(values)},
        )
        return self._belief

    def rollout(
        self,
        belief: Any,
        query_times: Any,
        *,
        return_events: bool,
        return_auxiliary: bool,
    ) -> BeliefTrajectory:
        assert belief is self._belief
        assert return_events is False
        assert return_auxiliary is False
        assert self._belief is not None
        offsets = torch.as_tensor(query_times, dtype=torch.float32)
        objects = self._belief.objects
        steps = offsets.numel()
        positions = (
            objects.position[:, None] + objects.velocity[:, None] * offsets[None, :, None, None]
        )
        return BeliefTrajectory(
            timestamps=self._belief.timestamp[:, None] + offsets[None],
            positions=positions,
            velocities=objects.velocity[:, None].expand(-1, steps, -1, -1).clone(),
            orientations=objects.orientation[:, None].expand(-1, steps, -1, -1).clone(),
            motion_mode_logits=objects.motion_mode_logits[:, None]
            .expand(-1, steps, -1, -1)
            .clone(),
            fast_log_variance=objects.fast_log_variance[:, None].expand(-1, steps, -1, -1).clone(),
            active_mask=objects.active[:, None].expand(-1, steps, -1).clone(),
        )


def _cell_row(
    cell: PhysicalCell, *, schedule: str = "none", split: str = "development"
) -> PhysicalManifestRow:
    return PhysicalManifestRow(
        split=split,  # type: ignore[arg-type]
        ordinal=0,
        seed=123,
        cell_index=PHYSICAL_CELLS.index(cell),
        object_count=cell.object_count,
        contact=cell.contact,
        dynamic_membership=cell.dynamic_membership,
        lifecycle_schedule=schedule,  # type: ignore[arg-type]
        known_action=False,
        contact_origin="natural" if cell.contact else "none",
        action_target_rank=None,
        action_time_stratum=None,
        camera_stratum=0,
        contact_geometry="head_on" if cell.contact else "none",
        distribution="in_distribution",
    )


def _certificate(
    cell: PhysicalCell, *, births: int = 0, removals: int = 0
) -> DynamicSetSceneCertificate:
    return DynamicSetSceneCertificate(
        peak_object_count=cell.object_count,
        natural_pair_collision_count=int(cell.contact),
        action_induced_pair_collision_count=0,
        lifecycle_birth_count=births,
        lifecycle_removal_count=removals,
        known_action_count=0,
        minimum_visible_fraction=1.0,
        minimum_image_clearance_pixels=1.0,
    )


def _manual_materialization(
    cell: PhysicalCell,
    *,
    schedule: str = "none",
    removal_frame: int | None = None,
    birth_frame: int | None = None,
    collision_frame: int | None = None,
    split: str = "development",
) -> DynamicSetMaterialization:
    active = torch.zeros(56, 6, dtype=torch.bool)
    object_id = torch.full((56, 6), -1, dtype=torch.int64)
    position = torch.zeros(56, 6, 3)
    velocity = torch.zeros_like(position)
    created = torch.zeros_like(active)
    removed = torch.zeros_like(active)
    for slot in range(cell.object_count):
        active[:, slot] = True
        object_id[:, slot] = 10 + slot
        position[:, slot] = torch.tensor([0.5 * slot, 0.0, 4.0])
        created[0, slot] = True
    if removal_frame is not None:
        active[removal_frame:, 0] = False
        object_id[removal_frame:, 0] = -1
        removed[removal_frame, 0] = True
    if birth_frame is not None:
        active[birth_frame:, 0] = True
        object_id[birth_frame:, 0] = 99
        position[birth_frame:, 0] = torch.tensor([0.0, 0.0, 4.0])
        created[birth_frame, 0] = True
    collision = torch.zeros_like(active)
    pair_collision = torch.zeros(56, 6, 6, dtype=torch.bool)
    if collision_frame is not None:
        collision[collision_frame, :2] = True
        pair_collision[collision_frame, 0, 1] = True
        pair_collision[collision_frame, 1, 0] = True
    events = {
        "created": created,
        "removed": removed,
        "collision": collision,
        "pair_collision": pair_collision,
        "known_action_observed": torch.zeros_like(active),
    }
    episode = {
        "timestamps": torch.arange(56, dtype=torch.float32) / 20.0,
        "objects": {
            "active": active,
            "id": object_id,
            "position": position,
            "velocity": velocity,
        },
        "events": events,
        "labels": {},
    }
    return DynamicSetMaterialization(
        row=_cell_row(cell, schedule=schedule, split=split),
        episode=episode,
        known_action_observed=events["known_action_observed"],
        certificate=_certificate(
            cell,
            births=int(birth_frame is not None),
            removals=int(removal_frame is not None),
        ),
        accepted_seed=123,
        attempt_count=1,
        rejection_reasons=(),
    )


def _manual_trace(
    materialization: DynamicSetMaterialization,
    *,
    retirement_delay: int = 1,
    birth_confirmation_delay: int = 1,
    predicted_collision_frame: int | None = None,
) -> DynamicSetEpisodeTrace:
    objects = materialization.episode["objects"]
    events = materialization.episode["events"]
    frames: list[BeliefFrameTrace] = []
    truth_birth = torch.nonzero(events["created"][1:], as_tuple=False)
    truth_removal = torch.nonzero(events["removed"], as_tuple=False)
    birth_frame = int(truth_birth[0, 0] + 1) if truth_birth.numel() else None
    removal_frame = int(truth_removal[0, 0]) if truth_removal.numel() else None
    for frame_index in range(56):
        runtime_active = objects["active"][frame_index].clone()
        runtime_id = objects["id"][frame_index].clone()
        runtime_position = objects["position"][frame_index].clone()
        if (
            removal_frame is not None
            and removal_frame <= frame_index < removal_frame + retirement_delay
        ):
            runtime_active[0] = True
            runtime_id[0] = 100 + int(objects["id"][removal_frame - 1, 0])
        if (
            birth_frame is not None
            and birth_frame <= frame_index < birth_frame + birth_confirmation_delay
        ):
            runtime_active[0] = False
            runtime_id[0] = -1
        for slot in torch.nonzero(runtime_active, as_tuple=False).flatten().tolist():
            truth_identity = int(objects["id"][frame_index, slot])
            if truth_identity >= 0:
                runtime_id[slot] = 100 + truth_identity
        modes = torch.full((6,), int(MotionMode.FREE), dtype=torch.int64)
        interval_collision = torch.zeros(6, dtype=torch.bool)
        pair_collision_probability = torch.zeros(6, 6)
        if frame_index == predicted_collision_frame:
            interval_collision[:2] = True
            pair_collision_probability[0, 1] = 1.0
            pair_collision_probability[1, 0] = 1.0
        proposal_position = torch.zeros(8, 3)
        proposal_valid = torch.zeros(8, dtype=torch.bool)
        proposal_probability = torch.zeros(8)
        for proposal, slot in enumerate(
            torch.nonzero(objects["active"][frame_index], as_tuple=False).flatten().tolist()
        ):
            proposal_position[proposal] = objects["position"][frame_index, slot]
            proposal_valid[proposal] = True
            proposal_probability[proposal] = 1.0
        frames.append(
            BeliefFrameTrace(
                frame_index=frame_index,
                timestamp=frame_index / 20.0,
                object_id=runtime_id,
                active=runtime_active,
                position=runtime_position,
                velocity=torch.zeros(6, 3),
                motion_mode=modes,
                interval_collision=interval_collision,
                interval_pair_collision_probability=pair_collision_probability,
                fast_log_variance=torch.zeros(6, 21),
                proposal=ProposalFrameTrace(
                    position=proposal_position,
                    log_variance=torch.zeros(8, 3),
                    existence_probability=proposal_probability,
                    valid_mask=proposal_valid,
                ),
            )
        )
    anchor = 15
    offsets = torch.tensor(evaluation.HORIZONS_SECONDS)
    source = frames[anchor]
    horizon = HorizonTrace(
        anchor_frame=anchor,
        source_object_id=source.object_id.clone(),
        source_active=source.active.clone(),
        timestamps=torch.tensor(anchor / 20.0) + offsets,
        positions=source.position[None].expand(6, -1, -1).clone(),
        velocities=source.velocity[None].expand(6, -1, -1).clone(),
        fast_log_variance=source.fast_log_variance[None].expand(6, -1, -1).clone(),
        active_mask=source.active[None].expand(6, -1).clone(),
    )
    return DynamicSetEpisodeTrace(
        frames=tuple(frames),
        horizon=horizon,
        perception_latencies_seconds=(0.001,) * 56,
        six_horizon_latency_seconds=0.002,
        public_action_count=0,
        persistent_tensor_bytes=1024,
        process_rss_bytes=2048,
    )


def test_public_pass_receives_only_rgbd_calibration_and_action_once() -> None:
    model = _PublicOnlySpyModel()
    trace = run_public_dynamic_set_episode(model, _public_frames(action_frame=10))

    assert model.packet_count == 56
    assert len(model.actions) == 1
    assert model.actions[0].object_id.tolist() == [7]
    assert model.actions[0].timestamp.tolist() == pytest.approx([0.5])
    assert trace.public_action_count == 1
    assert len(trace.frames) == 56
    assert trace.horizon.positions.shape == (6, 6, 3)
    assert all(not hasattr(frame, "known_action_object_id") for frame in _public_frames())


def test_single_entry_batch_delegates_to_exact_b1_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _PublicOnlySpyModel()
    expected = _manual_trace(
        _manual_materialization(
            PhysicalCell(object_count=6, contact=False, dynamic_membership=False)
        )
    )
    calls = 0

    def b1(*args: Any, **kwargs: Any) -> DynamicSetEpisodeTrace:
        nonlocal calls
        calls += 1
        return expected

    monkeypatch.setattr(evaluation, "run_public_dynamic_set_episode", b1)
    result = evaluation.run_public_dynamic_set_batch(model, (_public_frames(),))

    assert len(result) == 1 and result[0] is expected
    assert calls == 1


def test_batched_public_action_rejects_mixed_action_schedules() -> None:
    action = _public_frames(action_frame=10)[10]
    no_action = _public_frames()[10]

    with pytest.raises(evaluation.DynamicSetEvaluationError, match="may not mix"):
        evaluation._batched_public_action((action, no_action), None)


def test_n1_replacement_runtime_resolves_new_checkpoint_id_before_ingest() -> None:
    row = physical_manifest("development")[111]
    assert row.object_count == 1
    assert row.lifecycle_schedule == "remove_then_birth"
    materialization = materialize_dynamic_set_episode(row)
    public_frames = tuple(materialization.public_frames())
    action_frame = next(frame for frame in public_frames if bool(frame.known_action_observed[0]))
    model = _PublicOnlySpyModel(
        replacement_schedule=True,
        appearance_handle=action_frame.known_action_appearance_handle,
    )

    trace = run_public_dynamic_set_episode(model, public_frames)

    assert trace.public_action_count == 1
    assert len(model.actions) == 1
    assert model.actions[0].object_id.tolist() == [8_801]
    private_id = int(
        materialization.episode["events"]["known_action_object_id"][
            materialization.episode["events"]["known_action_observed"]
        ].item()
    )
    assert private_id != int(model.actions[0].object_id[0])
    assert len(trace.frames) == 56


def test_real_b3_action_lifecycle_and_contact_trace_matches_b1_within_gate_tolerance() -> None:
    manifest = physical_manifest("development")
    rows = (
        next(
            row
            for row in manifest
            if row.object_count == 1
            and not row.dynamic_membership
            and row.known_action
            and row.action_time_stratum == 0
        ),
        next(
            row
            for row in manifest
            if row.object_count == 1
            and row.lifecycle_schedule == "removal"
            and row.known_action
            and row.action_time_stratum == 0
        ),
        next(
            row
            for row in manifest
            if row.contact_origin == "action_induced"
            and row.known_action
            and row.action_time_stratum == 0
        ),
    )
    assert all(row.known_action and row.action_time_stratum == 0 for row in rows)
    assert rows[0].lifecycle_schedule == "none"
    assert rows[1].lifecycle_schedule == "removal"
    assert rows[2].contact and rows[2].contact_origin == "action_induced"
    materializations = tuple(materialize_dynamic_set_episode(row) for row in rows)
    public_episodes = tuple(tuple(item.public_frames()) for item in materializations)

    torch.manual_seed(61)
    b1_model = OnlineWorldModel.from_config(load_config(_PROFILE), device="cpu")
    batched_model = OnlineWorldModel.from_config(load_config(_PROFILE), device="cpu")
    batched_model.load_state_dict(b1_model.state_dict(), strict=True)
    b1 = tuple(
        run_public_dynamic_set_episode(b1_model, public_frames) for public_frames in public_episodes
    )
    batched_traces = evaluation.run_public_dynamic_set_batch(batched_model, public_episodes)

    assert [trace.public_action_count for trace in b1] == [1, 1, 1]
    assert [trace.public_action_count for trace in batched_traces] == [1, 1, 1]
    assert all(trace.runtime_batch_size == 3 for trace in batched_traces)
    for serial, batched in zip(b1, batched_traces, strict=True):
        for serial_frame, batched_frame in zip(serial.frames, batched.frames, strict=True):
            assert torch.equal(serial_frame.object_id, batched_frame.object_id)
            assert torch.equal(serial_frame.active, batched_frame.active)
            assert torch.equal(serial_frame.motion_mode, batched_frame.motion_mode)
            assert torch.equal(serial_frame.interval_collision, batched_frame.interval_collision)
            torch.testing.assert_close(
                serial_frame.interval_pair_collision_probability,
                batched_frame.interval_pair_collision_probability,
                rtol=0.0,
                atol=0.0,
            )
            for serial_value, batched_value in (
                (serial_frame.position, batched_frame.position),
                (serial_frame.velocity, batched_frame.velocity),
                (serial_frame.fast_log_variance, batched_frame.fast_log_variance),
                (serial_frame.proposal.position, batched_frame.proposal.position),
                (
                    serial_frame.proposal.log_variance,
                    batched_frame.proposal.log_variance,
                ),
                (
                    serial_frame.proposal.existence_probability,
                    batched_frame.proposal.existence_probability,
                ),
            ):
                torch.testing.assert_close(
                    serial_value,
                    batched_value,
                    rtol=1.0e-5,
                    atol=2.0e-6,
                )
            assert torch.equal(serial_frame.proposal.valid_mask, batched_frame.proposal.valid_mask)
        assert torch.equal(serial.horizon.source_object_id, batched.horizon.source_object_id)
        assert torch.equal(serial.horizon.source_active, batched.horizon.source_active)
        assert torch.equal(serial.horizon.active_mask, batched.horizon.active_mask)
        torch.testing.assert_close(
            serial.horizon.positions,
            batched.horizon.positions,
            rtol=1.0e-5,
            atol=2.0e-6,
        )
        torch.testing.assert_close(
            serial.horizon.velocities,
            batched.horizon.velocities,
            rtol=1.0e-5,
            atol=2.0e-6,
        )


def test_paired_stream_is_ordered_bounded_and_consumes_each_row_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = PhysicalCell(object_count=6, contact=False, dynamic_membership=False)
    base = _manual_materialization(cell)
    materializations = tuple(
        replace(base, row=replace(base.row, ordinal=index, seed=123 + index)) for index in range(3)
    )
    base_trace = _manual_trace(base)
    yielded = 0
    batch_calls: list[tuple[int, int]] = []
    candidate = _PublicOnlySpyModel()
    reference = _PublicOnlySpyModel()

    def stream() -> Any:
        nonlocal yielded
        for materialization in materializations:
            yielded += 1
            yield materialization

    monkeypatch.setattr(
        DynamicSetMaterialization,
        "public_frames_with_boundary",
        _public_frames_with_boundary,
    )

    def public_batch(model: nn.Module, episodes: Any, **_kwargs: Any) -> Any:
        episodes = tuple(episodes)
        boundaries = tuple(_kwargs["public_boundary_evidence"])
        batch_calls.append((id(model), len(episodes)))
        trace = replace(base_trace, runtime_batch_size=len(episodes))
        if len(episodes) > 1:
            trace = replace(
                trace,
                perception_latencies_seconds=(999.0,) * evaluation.DYNAMIC_SET_FRAMES,
                six_horizon_latency_seconds=999.0,
                persistent_tensor_bytes=999_999,
                process_rss_bytes=999_999,
            )
        return tuple(replace(trace, public_boundary_evidence=boundary) for boundary in boundaries)

    monkeypatch.setattr(evaluation, "run_public_dynamic_set_batch", public_batch)
    result = evaluation.evaluate_paired_dynamic_set_materializations(
        candidate,
        reference,
        stream(),
        batch_size=2,
    ).validate()

    assert yielded == 3
    assert [size for _, size in batch_calls] == [1, 1, 2, 2]
    assert result.candidate.episode_count == 3
    assert result.reference.episode_count == 3
    assert [item.ordinal for item in result.candidate.per_example_score_evidence] == [0, 1, 2]
    assert [item.ordinal for item in result.reference.per_example_score_evidence] == [0, 1, 2]
    assert (
        evaluation.dynamic_set_truth_leakage_count(result.candidate.per_example_score_evidence) == 0
    )
    assert all(
        item.public_boundary_evidence.truth_bound
        for item in result.candidate.per_example_score_evidence
    )
    assert result.candidate.resources.persistent_tensor_bytes == 1024
    assert result.reference.resources.persistent_tensor_bytes == 1024
    assert result.candidate.resources.perception_latency_seconds == 0.001
    assert result.reference.resources.perception_latency_seconds == 0.001
    assert result.candidate.resources.six_horizon_rollout_seconds == 0.002
    assert result.reference.resources.six_horizon_rollout_seconds == 0.002
    # Process RSS remains conservative and includes batched inference.
    assert result.candidate.resources.process_rss_bytes == 999_999
    assert result.reference.resources.process_rss_bytes == 999_999


def test_static_trace_scores_pooled_state_identity_and_six_horizons() -> None:
    cell = PhysicalCell(object_count=1, contact=False, dynamic_membership=False)
    materialization = _manual_materialization(cell)
    evidence = score_dynamic_set_trace(_manual_trace(materialization), materialization)
    metrics = evidence.physical_metrics(cell)

    assert metrics.proposal_f1.value == 1.0
    assert metrics.exact_count_accuracy.value == 1.0
    assert metrics.current_position_rmse_m.value == 0.0
    assert metrics.mature_velocity_rmse_mps.support > 0
    assert metrics.post_event_velocity_rmse_mps is None
    assert not any(
        failure.startswith("post_event_velocity_rmse_mps")
        for failure in physical_cell_gate_failures(cell, metrics)
    )
    assert set(metrics.horizon_position_rmse_m) == set(evaluation.HORIZONS_SECONDS)
    assert all(
        item.value == 0.0 and item.support > 0 for item in metrics.horizon_position_rmse_m.values()
    )
    assert metrics.persistent_id_accuracy.value == 1.0
    assert metrics.identity_switch_rate.value == 0.0
    assert metrics.birth_precision is not None and metrics.birth_precision.value == 1.0
    assert metrics.birth_precision.support == 1
    assert metrics.removal_precision is not None and metrics.removal_precision.value == 1.0
    assert metrics.collision_f1 is not None and metrics.collision_f1.value == 1.0
    assert metrics.collision_f1.support == 1


def test_identity_excludes_only_each_ids_first_confirmation_frame() -> None:
    cell = PhysicalCell(object_count=1, contact=False, dynamic_membership=False)
    materialization = _manual_materialization(cell)
    trace = _manual_trace(materialization)
    frames = list(trace.frames)
    frames[0] = replace(
        frames[0],
        active=torch.zeros_like(frames[0].active),
        object_id=torch.full_like(frames[0].object_id, -1),
    )
    confirmed = score_dynamic_set_trace(
        replace(trace, frames=tuple(frames)),
        materialization,
    ).physical_metrics(cell)
    assert confirmed.persistent_id_accuracy.value == 1.0
    assert confirmed.persistent_id_accuracy.support == 55
    assert confirmed.current_position_rmse_m.value == 0.0
    assert confirmed.current_position_rmse_m.support == 55 * 3

    frames[1] = replace(
        frames[1],
        active=torch.zeros_like(frames[1].active),
        object_id=torch.full_like(frames[1].object_id, -1),
    )
    late = score_dynamic_set_trace(
        replace(trace, frames=tuple(frames)),
        materialization,
    ).physical_metrics(cell)
    assert late.persistent_id_accuracy.value == pytest.approx(54.0 / 55.0)
    assert late.current_position_rmse_m.value > 0.0
    assert late.current_position_rmse_m.support == 55 * 3


def test_truth_velocity_segments_begin_at_persistent_birth_confirmation() -> None:
    static = _manual_materialization(
        PhysicalCell(object_count=1, contact=False, dynamic_membership=False)
    )
    static_counts, static_post_event = evaluation._truth_segment_metadata(static.episode)
    assert static_counts[:17, 0].tolist() == list(range(17))
    assert not bool(static_post_event[:, 0].any())

    dynamic = _manual_materialization(
        PhysicalCell(object_count=1, contact=False, dynamic_membership=True),
        birth_frame=10,
    )
    counts, post_event = evaluation._truth_segment_metadata(dynamic.episode)
    assert counts[10:14, 0].tolist() == [0, 1, 2, 3]
    assert post_event[10:14, 0].tolist() == [True, True, True, True]


def test_state_and_horizon_error_support_is_truth_indexed_when_track_is_missing() -> None:
    cell = PhysicalCell(object_count=1, contact=False, dynamic_membership=False)
    materialization = _manual_materialization(cell)
    complete_trace = _manual_trace(materialization)
    complete = score_dynamic_set_trace(complete_trace, materialization)
    frames = list(complete_trace.frames)
    missing_frame = 20
    frames[missing_frame] = replace(
        frames[missing_frame],
        active=torch.zeros_like(frames[missing_frame].active),
        object_id=torch.full_like(frames[missing_frame].object_id, -1),
    )
    missing = score_dynamic_set_trace(
        replace(complete_trace, frames=tuple(frames)),
        materialization,
    )

    assert missing.current_position.coordinate_count == complete.current_position.coordinate_count
    assert missing.mature_velocity.coordinate_count == complete.mature_velocity.coordinate_count
    assert missing.uncertainty_90.denominator == complete.uncertainty_90.denominator
    assert missing.current_position.squared_error > complete.current_position.squared_error
    assert missing.mature_velocity.squared_error > complete.mature_velocity.squared_error
    assert missing.uncertainty_90.numerator < complete.uncertainty_90.numerator

    anchor_frames = list(complete_trace.frames)
    anchor = complete_trace.horizon.anchor_frame
    anchor_frames[anchor] = replace(
        anchor_frames[anchor],
        active=torch.zeros_like(anchor_frames[anchor].active),
        object_id=torch.full_like(anchor_frames[anchor].object_id, -1),
    )
    missing_anchor = score_dynamic_set_trace(
        replace(complete_trace, frames=tuple(anchor_frames)),
        materialization,
    )
    for horizon in evaluation.HORIZONS_SECONDS:
        assert (
            missing_anchor.horizon_position[horizon].coordinate_count
            == complete.horizon_position[horizon].coordinate_count
        )
        assert (
            missing_anchor.horizon_velocity[horizon].coordinate_count
            == complete.horizon_velocity[horizon].coordinate_count
        )
        assert missing_anchor.horizon_position[horizon].squared_error > 0.0
        assert missing_anchor.horizon_velocity[horizon].squared_error > 0.0


def test_static_no_contact_metrics_expose_false_positive_failures() -> None:
    cell = PhysicalCell(object_count=2, contact=False, dynamic_membership=False)
    evidence = DynamicSetCellAccumulator(episode_count=1)
    evidence.collision.add(false_positive=1)
    evidence.birth.add(false_positive=1)
    evidence.removal.add(false_positive=1)

    metrics = evidence.physical_metrics(cell)
    assert metrics.collision_f1 is not None and metrics.collision_f1.value == 0.0
    assert metrics.birth_precision is not None and metrics.birth_precision.value == 0.0
    assert metrics.removal_precision is not None and metrics.removal_precision.value == 0.0


def test_remove_then_birth_metrics_use_confirmation_and_visible_miss_latency() -> None:
    cell = PhysicalCell(object_count=1, contact=False, dynamic_membership=True)
    materialization = _manual_materialization(
        cell,
        schedule="remove_then_birth",
        removal_frame=20,
        birth_frame=25,
    )
    trace = _manual_trace(
        materialization,
        retirement_delay=1,
        birth_confirmation_delay=1,
    )
    metrics = score_dynamic_set_trace(trace, materialization).physical_metrics(cell)

    assert metrics.birth_precision is not None and metrics.birth_precision.value == 1.0
    assert metrics.birth_recall is not None and metrics.birth_recall.value == 1.0
    assert metrics.birth_latency_frames is not None
    assert metrics.birth_latency_frames.value == 0.0
    assert metrics.removal_precision is not None and metrics.removal_precision.value == 1.0
    assert metrics.removal_recall is not None and metrics.removal_recall.value == 1.0
    assert metrics.removal_latency_frames is not None
    assert metrics.removal_latency_frames.value == 1.0
    assert metrics.post_event_velocity_rmse_mps is not None
    assert metrics.post_event_velocity_rmse_mps.support > 0


def test_lifecycle_latency_support_penalizes_missed_truth_events() -> None:
    cell = PhysicalCell(object_count=1, contact=False, dynamic_membership=True)
    materialization = _manual_materialization(
        cell,
        schedule="remove_then_birth",
        removal_frame=20,
        birth_frame=25,
    )
    trace = _manual_trace(materialization)
    frames = tuple(
        replace(
            frame,
            active=torch.zeros_like(frame.active),
            object_id=torch.full_like(frame.object_id, -1),
        )
        for frame in trace.frames
    )
    metrics = score_dynamic_set_trace(
        replace(trace, frames=frames),
        materialization,
    ).physical_metrics(cell)

    assert metrics.birth_recall is not None and metrics.birth_recall.value == 0.0
    assert metrics.birth_latency_frames is not None
    assert metrics.birth_latency_frames.support == 1
    assert metrics.birth_latency_frames.value == 2.0
    assert metrics.removal_recall is not None and metrics.removal_recall.value == 0.0
    assert metrics.removal_latency_frames is not None
    assert metrics.removal_latency_frames.support == 1
    assert metrics.removal_latency_frames.value == 3.0


def test_collision_confusion_and_timing_are_event_supported() -> None:
    cell = PhysicalCell(object_count=2, contact=True, dynamic_membership=False)
    materialization = _manual_materialization(cell, collision_frame=20)
    trace = _manual_trace(materialization, predicted_collision_frame=20)
    assert not bool(trace.frames[20].motion_mode.eq(int(MotionMode.COLLISION)).any())
    assert trace.frames[20].interval_collision[:2].tolist() == [True, True]
    metrics = score_dynamic_set_trace(trace, materialization).physical_metrics(cell)

    assert metrics.collision_f1 is not None
    assert metrics.collision_f1.value == 1.0
    assert metrics.collision_f1.support == 2
    assert metrics.collision_timing_error_frames is not None
    assert metrics.collision_timing_error_frames.value == 0.0
    assert metrics.collision_timing_error_frames.support == 1
    assert metrics.post_event_velocity_rmse_mps is not None
    assert metrics.post_event_velocity_rmse_mps.support > 0


def test_collision_gate_scores_learned_pair_confidence_not_analytic_mask() -> None:
    cell = PhysicalCell(object_count=2, contact=True, dynamic_membership=False)
    materialization = _manual_materialization(cell, collision_frame=20)
    trace = _manual_trace(materialization)
    frames = list(trace.frames)
    hard_collision = frames[20].interval_collision.clone()
    hard_collision[:2] = True
    frames[20] = replace(frames[20], interval_collision=hard_collision)

    metrics = score_dynamic_set_trace(
        replace(trace, frames=tuple(frames)),
        materialization,
    ).physical_metrics(cell)

    assert metrics.collision_f1 is not None
    assert metrics.collision_f1.value == 0.0
    assert metrics.collision_timing_error_frames is not None
    assert metrics.collision_timing_error_frames.support == 1
    assert metrics.collision_timing_error_frames.value == evaluation.DYNAMIC_SET_FRAMES


def test_collision_timing_support_cannot_drop_missed_truth_events() -> None:
    total, support = evaluation._nearest_event_timing((10, 20, 30), (10, 31))

    assert support == 3
    assert total == 1.0 + evaluation.DYNAMIC_SET_FRAMES


def test_real_n6_contact_uses_interval_event_not_cleared_endpoint_mode() -> None:
    row = physical_manifest("development")[20]
    assert row.object_count == 6 and row.contact and not row.dynamic_membership
    materialization = materialize_dynamic_set_episode(row)
    model = OnlineWorldModel.from_config(load_config(_PROFILE), device="cpu")

    trace = run_public_dynamic_set_episode(model, materialization.public_frames())
    collision_frames = [frame for frame in trace.frames if bool(frame.interval_collision.any())]
    metrics = score_dynamic_set_trace(trace, materialization).physical_metrics(
        PHYSICAL_CELLS[row.cell_index]
    )

    assert collision_frames
    assert all(
        not bool(frame.motion_mode.eq(int(MotionMode.COLLISION)).any())
        for frame in collision_frames
    )
    assert metrics.collision_f1 is not None
    assert metrics.collision_f1.value == 1.0


def test_additive_merge_pools_sse_instead_of_averaging_episode_rmse() -> None:
    first = SquaredErrorSum()
    first.add_tensor(torch.tensor([1.0]))
    second = SquaredErrorSum()
    second.add_tensor(torch.zeros(9))
    first.merge(second)

    assert first.coordinate_count == 10
    assert first.rmse() == pytest.approx(10.0**-0.5)


def test_unsupported_values_are_omitted_from_flat_metrics() -> None:
    cell = PhysicalCell(object_count=1, contact=False, dynamic_membership=False)
    evidence = DynamicSetCellAccumulator()
    result = DynamicSetEvaluationResult(
        by_cell={cell: evidence.physical_metrics(cell)},
        evidence_by_cell={cell: evidence},
        score=SelectionScoreEvidence(components={}),
        resources=ResourceMetrics(0.0, 0.0, 0, 0, 0),
        episode_count=0,
    )
    flat = result.flat_metrics()
    prefix = "cell/n1/contact0/dynamic0/current_position_rmse_m"

    assert flat[f"{prefix}/support"] == 0.0
    assert prefix not in flat


def test_horizon_velocity_is_an_independent_fixed_score_component() -> None:
    total = DynamicSetCellAccumulator()
    total.current_position.add_tensor(torch.zeros(3))
    total.mature_velocity.add_tensor(torch.zeros(3))
    total.proposal.add(true_positive=1)
    total.exact_count.add(1, 1)
    for horizon in evaluation.HORIZONS_SECONDS:
        total.horizon_position[horizon].add_tensor(torch.zeros(3))
        total.horizon_velocity[horizon].add_tensor(torch.full((3,), 0.02))
    score = evaluation._score_components(total)

    assert score.components["horizon_position"] == 0.0
    assert score.components["horizon_velocity"] == pytest.approx(1.0)


def test_per_example_evidence_round_trips_and_pairs_by_exact_manifest_row() -> None:
    cell = PhysicalCell(object_count=2, contact=True, dynamic_membership=False)
    materialization = _manual_materialization(cell, collision_frame=20)
    evidence = score_dynamic_set_trace(
        _manual_trace(materialization, predicted_collision_frame=20),
        materialization,
    )
    boundary = _public_frames_with_boundary(materialization)[1]
    candidate = DynamicSetExampleScoreEvidence.create(
        materialization.row,
        evidence,
        public_boundary_evidence=boundary,
    ).validate()
    reference = DynamicSetExampleScoreEvidence.create(
        materialization.row,
        evidence,
        public_boundary_evidence=boundary,
    ).validate()

    pairs = evaluation.pair_dynamic_set_example_score_evidence((candidate,), (reference,))
    pooled = evaluation.merge_dynamic_set_example_score_evidence((pairs[0][0], pairs[0][0]))

    assert pooled.episode_count == 2
    assert (
        pooled.current_position.coordinate_count == 2 * evidence.current_position.coordinate_count
    )
    assert candidate.additive.to_accumulator().collision.f1() == 1.0
    changed = DynamicSetExampleScoreEvidence.create(
        replace(materialization.row, ordinal=1),
        evidence,
    )
    with pytest.raises(ValueError, match="row bindings"):
        evaluation.pair_dynamic_set_example_score_evidence((candidate,), (changed,))


def test_per_example_truth_leakage_is_derived_from_boundary_evidence() -> None:
    cell = PhysicalCell(object_count=2, contact=True, dynamic_membership=False)
    materialization = _manual_materialization(cell, collision_frame=20)
    additive = score_dynamic_set_trace(
        _manual_trace(materialization, predicted_collision_frame=20),
        materialization,
    )
    frames = _public_frames()
    objects = dict(materialization.episode["objects"])
    objects["injected_public_alias"] = frames[0].rgb
    aliased_materialization = replace(
        materialization,
        episode={**materialization.episode, "objects": objects},
    )
    boundary = inspect_dynamic_set_public_boundary(aliased_materialization, frames)
    record = DynamicSetExampleScoreEvidence.create(
        materialization.row,
        additive,
        public_boundary_evidence=boundary,
    ).validate()

    assert boundary.truth_storage_alias_count == 1
    assert evaluation.dynamic_set_truth_leakage_count((record,)) == 1


def test_accumulator_rejects_dirty_boundary_before_private_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = PhysicalCell(object_count=6, contact=False, dynamic_membership=False)
    materialization = _manual_materialization(cell)
    frames = _public_frames()
    objects = dict(materialization.episode["objects"])
    objects["injected_public_alias"] = frames[0].rgb
    aliased_materialization = replace(
        materialization,
        episode={**materialization.episode, "objects": objects},
    )
    dirty = inspect_dynamic_set_public_boundary(aliased_materialization, frames)
    trace = replace(_manual_trace(materialization), public_boundary_evidence=dirty)
    accumulator = evaluation._DynamicSetEvaluationAccumulator(
        model=_PublicOnlySpyModel(),
        config=evaluation.DEFAULT_EVALUATION_CONFIG,
    )

    monkeypatch.setattr(
        evaluation,
        "score_dynamic_set_trace",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("private scorer opened before boundary rejection")
        ),
    )
    with pytest.raises(evaluation.DynamicSetEvaluationError, match="invalid public-boundary"):
        accumulator.add(materialization, trace)


def test_authorized_protected_rows_validate_before_one_shot_materialization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cell = PhysicalCell(object_count=6, contact=False, dynamic_membership=False)
    protected = _manual_materialization(cell, split="selector")
    row = protected.row
    manifest_sha256 = canonical_sha256([asdict(row)])
    protocol_sha256 = "a" * 64
    ledger, permit = _protected_ledger(tmp_path, protocol_sha256)
    trace = _manual_trace(protected)
    evidence = score_dynamic_set_trace(trace, protected)
    calls = 0
    materialization_calls = 0

    monkeypatch.setattr(
        DynamicSetMaterialization,
        "public_frames_with_boundary",
        _public_frames_with_boundary,
    )

    def materialize(
        requested: PhysicalManifestRow,
        *,
        capability: Any,
    ) -> DynamicSetMaterialization:
        nonlocal materialization_calls
        capability.claim(requested)
        materialization_calls += 1
        assert requested == row
        return protected

    def public_pass(*args: Any, **kwargs: Any) -> DynamicSetEpisodeTrace:
        nonlocal calls
        calls += 1
        return replace(trace, public_boundary_evidence=kwargs["public_boundary_evidence"])

    monkeypatch.setattr(evaluation, "run_public_dynamic_set_episode", public_pass)
    monkeypatch.setattr(
        evaluation,
        "_materialize_protected_dynamic_set_episode",
        materialize,
    )
    monkeypatch.setattr(
        evaluation,
        "score_dynamic_set_trace",
        lambda *args, **kwargs: evidence,
    )
    monkeypatch.setitem(
        evaluation.FROZEN_PHYSICAL_MANIFEST_SHA256,
        "selector",
        manifest_sha256,
    )
    monkeypatch.setitem(evaluation.PHYSICAL_SPLIT_SIZES, "selector", 1)
    result = evaluation.evaluate_authorized_dynamic_set_rows(
        _PublicOnlySpyModel(),
        ledger=ledger,
        permit=permit,
        expected_protocol_sha256=protocol_sha256,
        expected_manifest_sha256=manifest_sha256,
        expected_rows=(row,),
        require_all_cells=False,
    )

    assert calls == 1
    assert materialization_calls == 1
    assert result.population_binding is not None
    assert result.population_binding.manifest_sha256 == manifest_sha256
    assert result.population_binding.row_count == 1
    assert len(result.per_example_score_evidence) == 1
    with pytest.raises(ValueError, match="digest"):
        evaluation.evaluate_authorized_dynamic_set_rows(
            _PublicOnlySpyModel(),
            ledger=ledger,
            permit=permit,
            expected_protocol_sha256=protocol_sha256,
            expected_manifest_sha256=manifest_sha256,
            expected_rows=(replace(row, ordinal=1),),
            require_all_cells=False,
        )
    assert calls == 1
    assert materialization_calls == 1

    opened = False

    def caller_stream():
        nonlocal opened
        opened = True
        yield protected

    with pytest.raises(PermissionError, match="caller-supplied"):
        evaluation.evaluate_authorized_dynamic_set_materializations(
            _PublicOnlySpyModel(),
            caller_stream(),
            permit=permit,
            expected_protocol_sha256=protocol_sha256,
            expected_manifest_sha256=manifest_sha256,
            expected_rows=(row,),
        )
    assert not opened


def test_authorized_paired_rows_materialize_once_and_bind_both_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cell = PhysicalCell(object_count=6, contact=False, dynamic_membership=False)
    protected = _manual_materialization(cell, split="selector")
    row = protected.row
    manifest_sha256 = canonical_sha256([asdict(row)])
    protocol_sha256 = "a" * 64
    ledger, permit = _protected_ledger(tmp_path, protocol_sha256)
    trace = _manual_trace(protected)
    evidence = score_dynamic_set_trace(trace, protected)
    calls = 0
    materialization_calls = 0

    def materialize(
        requested: PhysicalManifestRow,
        *,
        capability: Any,
    ) -> DynamicSetMaterialization:
        nonlocal materialization_calls
        capability.claim(requested)
        materialization_calls += 1
        assert requested == row
        return protected

    monkeypatch.setattr(
        DynamicSetMaterialization,
        "public_frames_with_boundary",
        _public_frames_with_boundary,
    )

    def public_pass(*args: Any, **kwargs: Any) -> DynamicSetEpisodeTrace:
        nonlocal calls
        calls += 1
        return replace(trace, public_boundary_evidence=kwargs["public_boundary_evidence"])

    monkeypatch.setattr(evaluation, "run_public_dynamic_set_episode", public_pass)
    monkeypatch.setattr(
        evaluation,
        "_materialize_protected_dynamic_set_episode",
        materialize,
    )
    monkeypatch.setattr(
        evaluation,
        "score_dynamic_set_trace",
        lambda *args, **kwargs: evidence,
    )
    monkeypatch.setitem(
        evaluation.FROZEN_PHYSICAL_MANIFEST_SHA256,
        "selector",
        manifest_sha256,
    )
    monkeypatch.setitem(evaluation.PHYSICAL_SPLIT_SIZES, "selector", 1)
    result = evaluation.evaluate_authorized_paired_dynamic_set_rows(
        _PublicOnlySpyModel(),
        _PublicOnlySpyModel(),
        ledger=ledger,
        permit=permit,
        expected_protocol_sha256=protocol_sha256,
        expected_manifest_sha256=manifest_sha256,
        expected_rows=(row,),
        require_all_cells=False,
        batch_size=1,
    ).validate()

    assert calls == 2
    assert materialization_calls == 1
    assert result.candidate.population_binding == result.reference.population_binding
    assert result.candidate.population_binding is not None
    assert result.candidate.population_binding.manifest_sha256 == manifest_sha256
    reopened = OrderedSplitLedger(
        ledger.artifacts,
        artifact_name=ledger.artifact_name,
        protocol_sha256=ledger.protocol_sha256,
        split_order=ledger.split_order,
    )
    with pytest.raises(RuntimeError, match="already claimed"):
        evaluation.evaluate_authorized_paired_dynamic_set_rows(
            _PublicOnlySpyModel(),
            _PublicOnlySpyModel(),
            ledger=reopened,
            permit=permit,
            expected_protocol_sha256=protocol_sha256,
            expected_manifest_sha256=manifest_sha256,
            expected_rows=(row,),
            require_all_cells=False,
            batch_size=1,
        )
    assert materialization_calls == 1
    with pytest.raises(PermissionError, match="index"):
        evaluation.evaluate_authorized_paired_dynamic_set_rows(
            _PublicOnlySpyModel(),
            _PublicOnlySpyModel(),
            ledger=ledger,
            permit=replace(permit, index=1),
            expected_protocol_sha256=protocol_sha256,
            expected_manifest_sha256=manifest_sha256,
            expected_rows=(row,),
            require_all_cells=False,
            batch_size=1,
        )
    assert calls == 2
    assert materialization_calls == 1


def test_rollout_resource_median_uses_only_b1_n6_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        DynamicSetMaterialization,
        "public_frames_with_boundary",
        _public_frames_with_boundary,
    )
    n1 = _manual_materialization(
        PhysicalCell(object_count=1, contact=False, dynamic_membership=False)
    )
    n6 = replace(
        _manual_materialization(
            PhysicalCell(object_count=6, contact=False, dynamic_membership=False)
        ),
        row=replace(
            _cell_row(PhysicalCell(object_count=6, contact=False, dynamic_membership=False)),
            ordinal=1,
        ),
    )
    traces = [
        replace(_manual_trace(n1), six_horizon_latency_seconds=0.001),
        replace(_manual_trace(n6), six_horizon_latency_seconds=2.2),
    ]
    evidences = [
        score_dynamic_set_trace(traces[0], n1),
        score_dynamic_set_trace(traces[1], n6),
    ]
    calls = 0

    def public_pass(*args: Any, **kwargs: Any) -> DynamicSetEpisodeTrace:
        nonlocal calls
        result = traces[calls]
        calls += 1
        return replace(result, public_boundary_evidence=kwargs["public_boundary_evidence"])

    monkeypatch.setattr(evaluation, "run_public_dynamic_set_episode", public_pass)
    monkeypatch.setattr(
        evaluation,
        "score_dynamic_set_trace",
        lambda _trace, materialization, **_kwargs: evidences[materialization.row.ordinal],
    )

    result = evaluation.evaluate_dynamic_set_materializations(
        _PublicOnlySpyModel(),
        iter((n1, n6)),
    )

    assert result.resources.six_horizon_rollout_seconds == 2.2
    assert calls == 2


def test_protected_split_rejected_before_inference_and_generator_not_drained() -> None:
    cell = PhysicalCell(object_count=1, contact=False, dynamic_membership=False)
    protected = _manual_materialization(cell, split="selector")
    advanced = False

    def rows() -> Any:
        nonlocal advanced
        yield protected
        advanced = True
        raise AssertionError("protected evaluation drained later episodes")

    with pytest.raises(PermissionError, match="protected"):
        evaluation.evaluate_dynamic_set_materializations(_PublicOnlySpyModel(), rows())
    assert not advanced


def test_selection_score_refuses_to_promote_without_required_planning_component() -> None:
    physical = {
        name: 0.0
        for name in (
            "current_position",
            "current_velocity",
            "horizon_position",
            "horizon_velocity",
            "proposal_error",
            "identity_lifecycle_error",
            "contact_error",
        )
    }
    evidence = SelectionScoreEvidence(components=physical)

    with pytest.raises(ValueError, match="planning_error"):
        evidence.with_planning_error(float("nan"))
    assert evidence.with_planning_error(0.2) == pytest.approx(0.03)
