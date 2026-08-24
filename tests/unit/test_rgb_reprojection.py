from __future__ import annotations

import torch

from world_model.belief import BeliefFactory
from world_model.observations.rgb.reprojection import soft_sphere_silhouette_reprojection
from world_model.simulator.camera import CameraTrajectory, CameraTrajectoryConfig
from world_model.simulator.physics import SphereState
from world_model.simulator.renderer import render_spheres


def _scene() -> tuple[SphereState, object]:
    dtype = torch.float32
    state = SphereState(
        object_id=torch.tensor([0], dtype=torch.int64),
        active=torch.tensor([True]),
        position=torch.tensor([[0.0, 0.95, 0.0]], dtype=dtype),
        velocity=torch.zeros(1, 3, dtype=dtype),
        radius=torch.tensor([[0.21]], dtype=dtype),
        mass=torch.ones(1, 1, dtype=dtype),
        restitution=torch.full((1, 1), 0.7, dtype=dtype),
        drag=torch.full((1, 1), 0.05, dtype=dtype),
        friction=torch.full((1, 1), 0.2, dtype=dtype),
        albedo=torch.tensor([[0.85, 0.18, 0.10]], dtype=dtype),
        orientation=torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=dtype),
        angular_velocity=torch.zeros(1, 3, dtype=dtype),
        sleeping=torch.tensor([False]),
        sleep_counter=torch.zeros(1, dtype=torch.int64),
    )
    camera = CameraTrajectory(
        CameraTrajectoryConfig(image_size=(32, 32), mode="fixed"),
        seed=3,
    ).at(0.0)
    return state, camera


def _belief(position: torch.Tensor, state: SphereState, camera: object):
    factory = BeliefFactory(max_objects=1, geometry_dim=1, appearance_dim=3)
    belief = factory.create(
        batch_size=1,
        intrinsics=camera.intrinsics.unsqueeze(0),
        world_from_camera=camera.world_from_camera.unsqueeze(0),
        active_modalities=("rgb",),
    )
    objects = belief.objects.replace(
        object_id=torch.tensor([[0]], dtype=torch.int64),
        active=torch.tensor([[True]]),
        position=position,
        geometry=state.radius.unsqueeze(0),
        appearance=state.albedo.unsqueeze(0),
    )
    return belief.replace(objects=objects, metadata={"initialised": True})


def test_soft_sphere_reprojection_is_rgb_only_geometric_and_differentiable() -> None:
    state, camera = _scene()
    rendered = render_spheres(state, camera, (32, 32), noise_std=0.0)
    calibration = {
        "world_from_camera": camera.world_from_camera.unsqueeze(0),
        "intrinsics": camera.intrinsics.unsqueeze(0),
    }
    exact = _belief(state.position.unsqueeze(0), state, camera)
    exact_loss, exact_metrics = soft_sphere_silhouette_reprojection(
        exact,
        rendered.rgb.unsqueeze(0),
        calibration,
        foreground_threshold=0.04,
    )

    shifted_position = (state.position + torch.tensor([[0.30, 0.0, 0.0]])).unsqueeze(0)
    shifted_position.requires_grad_(True)
    shifted = _belief(shifted_position, state, camera)
    shifted_loss, shifted_metrics = soft_sphere_silhouette_reprojection(
        shifted,
        rendered.rgb.unsqueeze(0),
        calibration,
        foreground_threshold=0.04,
    )

    assert exact_metrics["rgb_reprojection_supported_row_count"] == 1.0
    assert exact_metrics["rgb_reprojection_silhouette_iou_count"] == 1.0
    assert shifted_metrics["rgb_reprojection_projectable_object_count"] == 1.0
    assert exact_metrics["rgb_reprojection_foreground_pixel_count"] > 0
    assert exact_loss < shifted_loss
    shifted_loss.backward()
    assert shifted_position.grad is not None
    assert torch.isfinite(shifted_position.grad).all()
    assert bool((shifted_position.grad != 0).any())


def test_soft_sphere_reprojection_omits_unsupported_rows_without_nonfinite_loss() -> None:
    state, camera = _scene()
    inactive_position = state.position.unsqueeze(0).clone().requires_grad_(True)
    belief = _belief(inactive_position, state, camera)
    belief = belief.replace(objects=belief.objects.replace(active=torch.tensor([[False]])))
    background = render_spheres(
        state.__class__(
            **{**state.__dict__, "active": torch.tensor([False]), "object_id": torch.tensor([-1])}
        ),
        camera,
        (32, 32),
        noise_std=0.0,
    ).rgb.unsqueeze(0)
    loss, metrics = soft_sphere_silhouette_reprojection(
        belief,
        background,
        {
            "world_from_camera": camera.world_from_camera.unsqueeze(0),
            "intrinsics": camera.intrinsics.unsqueeze(0),
        },
        foreground_threshold=0.04,
    )
    assert metrics["rgb_reprojection_supported_row_count"] == 0.0
    assert torch.isfinite(loss)
    assert loss == 0
