"""Development-only perceptual capacity probes above the N=6 promotion line."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
from scipy.optimize import linear_sum_assignment

from world_model.evaluation.capability_factor_runner import _model_from_workbench_checkpoint
from world_model.observations import ObservationPacket
from world_model.simulator import CameraFrame, SphereState, make_intrinsics, render_spheres
from world_model.training.dynamic_set_config import OrpheusConfig, load_config


@dataclass(frozen=True, slots=True)
class PerceptualScaleProbeResult:
    object_count: int
    proposal_count: int
    birth_proposals: int
    observed_active_count: int
    exact_count: bool
    position_rmse_m: float | None
    finite: bool
    inference_latency_seconds: float
    learned_weight_bytes: int
    full_perceptual_qualification: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def scaled_set_config(
    config: OrpheusConfig,
    *,
    max_objects: int,
    birth_proposals: int = 2,
) -> OrpheusConfig:
    """Create a separate dynamic-set profile without altering the N=6 source."""

    if isinstance(max_objects, bool) or not isinstance(max_objects, int) or max_objects < 1:
        raise ValueError("max_objects must be a positive integer")
    if (
        isinstance(birth_proposals, bool)
        or not isinstance(birth_proposals, int)
        or birth_proposals < 1
    ):
        raise ValueError("birth_proposals must be a positive integer")
    scaled = replace(
        config,
        simulator=replace(
            config.simulator,
            min_objects=min(config.simulator.min_objects, max_objects),
            max_objects=max_objects,
        ),
        model=replace(
            config.model,
            max_objects=max_objects,
            rgbd=replace(
                config.model.rgbd,
                max_objects=max_objects,
                birth_proposals=birth_proposals,
                proposal_count=max_objects + birth_proposals,
            ),
        ),
    )
    scaled.validate()
    return scaled


def _camera(image_size: tuple[int, int]) -> CameraFrame:
    identity = torch.eye(4, dtype=torch.float32)
    return CameraFrame(
        timestamp=0.0,
        world_from_camera=identity,
        camera_from_world=identity,
        intrinsics=make_intrinsics(image_size, 48.0),
        position=torch.zeros(3),
        target=torch.tensor([0.0, 0.0, 1.0]),
    )


def _eight_sphere_state(timestamp: float) -> SphereState:
    positions = torch.tensor(
        [
            [-1.20, -0.52, 4.0],
            [-0.40, -0.52, 4.0],
            [0.40, -0.52, 4.0],
            [1.20, -0.52, 4.0],
            [-1.20, 0.52, 4.0],
            [-0.40, 0.52, 4.0],
            [0.40, 0.52, 4.0],
            [1.20, 0.52, 4.0],
        ],
        dtype=torch.float32,
    )
    velocity = torch.tensor(
        [[0.015 * (-1.0 if index % 2 else 1.0), 0.0, 0.0] for index in range(8)]
    )
    positions = positions + velocity * timestamp
    albedo = torch.tensor(
        [
            [0.90, 0.16, 0.12],
            [0.12, 0.82, 0.22],
            [0.12, 0.30, 0.92],
            [0.92, 0.76, 0.12],
            [0.72, 0.16, 0.86],
            [0.10, 0.82, 0.84],
            [0.94, 0.46, 0.12],
            [0.55, 0.55, 0.58],
        ],
        dtype=torch.float32,
    )
    return SphereState(
        object_id=torch.arange(100, 108, dtype=torch.int64),
        active=torch.ones(8, dtype=torch.bool),
        position=positions,
        velocity=velocity,
        radius=torch.full((8, 1), 0.21),
        mass=torch.ones(8, 1),
        restitution=torch.full((8, 1), 0.70),
        drag=torch.full((8, 1), 0.05),
        friction=torch.full((8, 1), 0.20),
        albedo=albedo,
        orientation=torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(8, -1).clone(),
        angular_velocity=torch.zeros(8, 3),
        sleeping=torch.zeros(8, dtype=torch.bool),
        sleep_counter=torch.zeros(8, dtype=torch.int64),
    )


def _packet(state: SphereState, timestamp: float, image_size: tuple[int, int]):
    camera = replace(_camera(image_size), timestamp=timestamp)
    rendered = render_spheres(state, camera, image_size)
    return ObservationPacket(
        modality="rgbd",
        sensor_id="camera0:rgbd",
        timestamp=timestamp,
        payload={
            "rgb": rendered.rgb.unsqueeze(0),
            "depth": rendered.depth_buffer[None, None],
        },
        calibration={
            "world_from_camera": camera.world_from_camera.unsqueeze(0),
            "intrinsics": camera.intrinsics.unsqueeze(0),
        },
        frame_id="camera:camera0:rgbd",
        metadata={"image_size": image_size},
    )


def run_n8_perception_probe(
    *,
    model_config_path: str | Path,
    checkpoint_path: str | Path,
) -> PerceptualScaleProbeResult:
    """Run one visible, separated N=8 development scene through public RGB-D."""

    base = load_config(model_config_path)
    config = scaled_set_config(base, max_objects=8, birth_proposals=2)
    model, _ = _model_from_workbench_checkpoint(config, checkpoint_path)
    image_size = tuple(int(value) for value in config.simulator.image_size)
    started = time.perf_counter()
    truth = None
    with torch.no_grad():
        for frame_index in range(3):
            timestamp = frame_index / float(config.simulator.frame_rate)
            truth = _eight_sphere_state(timestamp)
            model.ingest(_packet(truth, timestamp, image_size))
    latency = time.perf_counter() - started
    assert truth is not None and model.belief is not None
    active = model.belief.objects.active[0]
    observed_count = int(active.sum())
    finite = bool(
        torch.isfinite(model.belief.objects.position[0, active]).all()
        and torch.isfinite(model.belief.objects.velocity[0, active]).all()
    )
    rmse: float | None = None
    if observed_count == 8:
        estimated = model.belief.objects.position[0, active].detach().cpu()
        rows, columns = linear_sum_assignment(torch.cdist(estimated, truth.position).numpy())
        difference = estimated[rows] - truth.position[columns]
        rmse = float(difference.square().mean().sqrt())
    module = model.observation_modules["rgbd"]
    proposer = getattr(module, "set_proposer", None)
    if proposer is None:
        raise RuntimeError("scaled set profile did not construct a set proposer")
    learned_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    return PerceptualScaleProbeResult(
        object_count=8,
        proposal_count=int(proposer.proposal_count),
        birth_proposals=2,
        observed_active_count=observed_count,
        exact_count=observed_count == 8,
        position_rmse_m=rmse,
        finite=finite,
        inference_latency_seconds=latency,
        learned_weight_bytes=learned_bytes,
    )


__all__ = [
    "PerceptualScaleProbeResult",
    "run_n8_perception_probe",
    "scaled_set_config",
]
