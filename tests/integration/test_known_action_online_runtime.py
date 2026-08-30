"""Public RGB-D runtime checks for read-only known-action counterfactuals."""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from pathlib import Path

import pytest
import torch

from world_model.dynamics import WorldImpulseAction, free_motion_position_velocity
from world_model.observations import ObservationPacket
from world_model.planning import TerminalWorldPositionGoal, resolve_appearance_handle
from world_model.runtime import OnlineWorldModel
from world_model.runtime.prepared import tensor_identity_version_signature
from world_model.simulator import CameraFrame, SphereState, make_intrinsics, render_spheres
from world_model.utils.config import load_config

CONFIG_DIR = Path(__file__).parents[2] / "configs"
IMAGE_SIZE = (64, 64)


def _config():
    base = load_config(CONFIG_DIR / "rgbd_online_free_motion_cpu.yaml")
    return replace(
        base,
        model=replace(
            base.model,
            max_objects=2,
            state=replace(base.model.state, appearance_dim=3),
            rgbd=replace(base.model.rgbd, proposal_count=2),
            association=replace(base.model.association, appearance_weight=0.25),
        ),
        simulator=replace(base.simulator, min_objects=2, max_objects=2),
    )


def _camera() -> CameraFrame:
    identity = torch.eye(4, dtype=torch.float32)
    return CameraFrame(
        timestamp=0.0,
        world_from_camera=identity,
        camera_from_world=identity,
        intrinsics=make_intrinsics(IMAGE_SIZE, 50.0),
        position=torch.zeros(3),
        target=torch.tensor([0.0, 0.0, 1.0]),
    )


def _packet(
    position: torch.Tensor,
    velocity: torch.Tensor,
    timestamp: float,
    *,
    requires_grad: bool,
) -> tuple[ObservationPacket, torch.Tensor, torch.Tensor]:
    camera = _camera()
    state = SphereState(
        object_id=torch.arange(2, dtype=torch.int64),
        active=torch.ones(2, dtype=torch.bool),
        position=position,
        velocity=velocity,
        radius=torch.full((2, 1), 0.21),
        mass=torch.ones((2, 1)),
        restitution=torch.zeros((2, 1)),
        drag=torch.full((2, 1), 0.05),
        friction=torch.zeros((2, 1)),
        albedo=torch.tensor(
            [[0.90, 0.20, 0.18], [0.18, 0.82, 0.90]],
            dtype=torch.float32,
        ),
        orientation=torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(2, -1).clone(),
        angular_velocity=torch.zeros((2, 3)),
        sleeping=torch.zeros(2, dtype=torch.bool),
        sleep_counter=torch.zeros(2, dtype=torch.int64),
    )
    rendered = render_spheres(state, camera, IMAGE_SIZE)
    rgb = rendered.rgb.unsqueeze(0)
    depth = rendered.depth_buffer[None, None]
    if requires_grad:
        rgb.requires_grad_()
        depth.requires_grad_()
    return (
        ObservationPacket(
            modality="rgbd",
            sensor_id="camera0:rgbd",
            timestamp=timestamp,
            payload={"rgb": rgb, "depth": depth},
            calibration={
                "world_from_camera": camera.world_from_camera.unsqueeze(0),
                "intrinsics": camera.intrinsics.unsqueeze(0),
            },
            frame_id="camera:camera0:rgbd",
            metadata={"image_size": IMAGE_SIZE},
        ),
        rgb,
        depth,
    )


def _ingested_runtime() -> tuple[OnlineWorldModel, list[torch.Tensor], list[torch.Tensor]]:
    model = OnlineWorldModel.from_config(_config(), device="cpu")
    initial_position = torch.tensor([[-0.62, -0.10, 4.0], [0.66, 0.14, 4.25]])
    initial_velocity = torch.tensor([[0.035, 0.0, 0.0], [-0.035, 0.0, 0.0]])
    gravity = torch.zeros((1, 3))
    drag = torch.full((1, 2), 0.05)
    rgb_frames: list[torch.Tensor] = []
    depth_frames: list[torch.Tensor] = []
    for frame_index in range(16):
        timestamp = frame_index / 20.0
        position, velocity = free_motion_position_velocity(
            initial_position.unsqueeze(0),
            initial_velocity.unsqueeze(0),
            timestamp,
            gravity=gravity,
            drag=drag,
        )
        packet, rgb, depth = _packet(
            position[0],
            velocity[0],
            timestamp,
            requires_grad=True,
        )
        model.ingest(packet)
        rgb_frames.append(rgb)
        depth_frames.append(depth)
    return model, rgb_frames, depth_frames


def _assert_nested_equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
        return
    if is_dataclass(left) and not isinstance(left, type):
        assert type(right) is type(left)
        for field in fields(left):
            _assert_nested_equal(getattr(left, field.name), getattr(right, field.name))
        return
    if isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
        return
    assert left == right


def _action(model: OnlineWorldModel, impulse: torch.Tensor) -> WorldImpulseAction:
    assert model.belief is not None
    prototype = torch.tensor([[0.90, 0.20, 0.18]], dtype=model.belief.dtype)
    object_id = resolve_appearance_handle(
        model.belief,
        prototype,
        minimum_cosine_margin=0.1,
    )
    return WorldImpulseAction(
        timestamp=model.belief.timestamp + 0.3,
        object_id=object_id,
        impulse_world=impulse,
    )


def test_runtime_none_action_is_exact_and_known_action_is_read_only() -> None:
    model, _, _ = _ingested_runtime()
    assert model.belief is not None
    assert model.state_dict() == {}
    source_identity = tensor_identity_version_signature(model.belief)
    source = model.belief.clone()
    histories = tensor_identity_version_signature(model.state.temporal_histories)
    ingest_count = model.state.ingest_count
    diagnostics = model.diagnostics.latest
    measurements = model.last_measurements

    legacy = model.predict([0.1, 0.25, 0.5, 1.0, 2.0])
    explicit_none = model.predict([0.1, 0.25, 0.5, 1.0, 2.0], action=None)
    _assert_nested_equal(legacy, explicit_none)

    impulse = torch.tensor([[0.015, -0.006, 0.009]], requires_grad=True)
    action = _action(model, impulse)
    acted = model.predict([0.1, 0.25, 0.5, 1.0, 2.0], action=action)
    assert acted.auxiliary["known_action_applied"].sum().item() == 1
    assert acted.event_logits is not None
    assert model.belief is not None
    assert tensor_identity_version_signature(model.belief) == source_identity
    _assert_nested_equal(model.belief, source)
    assert model.state.ingest_count == ingest_count
    assert tensor_identity_version_signature(model.state.temporal_histories) == histories
    assert model.diagnostics.latest is diagnostics
    assert model.last_measurements is measurements


def test_runtime_action_and_plan_retain_rgbd_and_impulse_gradients() -> None:
    model, rgb_frames, depth_frames = _ingested_runtime()
    assert model.belief is not None
    impulse = torch.tensor([[0.012, -0.007, 0.009]], requires_grad=True)
    action = _action(model, impulse)
    query_times = [0.1, 0.25, 0.5, 1.0, 2.0]
    plus = model.predict(query_times, action=action)
    target_mask = model.belief.objects.object_id == action.object_id.unsqueeze(-1)
    target_slot = target_mask.to(torch.int64).argmax(dim=-1)
    batch = torch.arange(model.belief.batch_size)
    goal = TerminalWorldPositionGoal(
        object_id=action.object_id,
        position_world=plus.positions[batch, -1, target_slot].detach(),
    )
    opposite = replace(action, impulse_world=-action.impulse_world)
    plan = model.plan(query_times, (None, opposite, action), goal)
    assert plan.selected_index.tolist() == [2]

    target = (
        plus.positions[0, -1, target_slot[0]]
        .mul(plus.positions.new_tensor((0.5, -0.75, 1.25)))
        .sum()
    )
    sources = tuple(rgb_frames + depth_frames + [impulse])
    gradients = torch.autograd.grad(target, sources, retain_graph=True)
    for gradient in gradients:
        assert torch.isfinite(gradient).all()
        assert float(gradient.abs().sum()) > 0.0
    plan_gradients = torch.autograd.grad(plan.total_cost.sum(), impulse)
    assert torch.isfinite(plan_gradients[0]).all()
    assert float(plan_gradients[0].abs().sum()) > 0.0


def test_action_counterfactuals_fail_closed_with_hypothesis_controller() -> None:
    model, _, _ = _ingested_runtime()
    assert model.belief is not None
    action = _action(model, torch.tensor([[0.01, 0.0, 0.0]]))
    goal = TerminalWorldPositionGoal(
        object_id=action.object_id,
        position_world=model.belief.objects.position[:, 0],
    )
    model.hypothesis_controller = object()  # type: ignore[assignment]

    with pytest.raises(NotImplementedError, match="hypothesis controller"):
        model.predict([0.5], action=action)
    with pytest.raises(NotImplementedError, match="hypothesis controller"):
        model.plan([0.5], (action,), goal)
