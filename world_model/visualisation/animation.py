"""Held-out RGB-only prior/posterior demonstration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment

from world_model.belief import MotionMode
from world_model.observations import ObservationPacket
from world_model.runtime import OnlineWorldModel
from world_model.simulator import generate_episode
from world_model.training.checkpointing import load_checkpoint
from world_model.utils.config import OrpheusConfig
from world_model.utils.device import DeviceInfo
from world_model.visualisation.frames import normalized_to_pixels, overlay_points
from world_model.visualisation.plots import save_parameter_plot
from world_model.visualisation.trajectories import plot_xy_trajectory
from world_model.visualisation.uncertainty import add_uncertainty_ellipse


def _packet(
    episode: dict[str, Any],
    index: int,
    *,
    device: torch.device,
) -> ObservationPacket:
    return ObservationPacket(
        modality="rgb",
        sensor_id="camera0",
        timestamp=float(episode["timestamps"][index]),
        payload=episode["rgb"][index].to(device),
        calibration={
            "intrinsics": episode["camera"]["intrinsics"][index].to(device),
            "world_from_camera": episode["camera"]["world_from_camera"][index].to(device),
        },
        frame_id="camera:camera0",
        metadata={"image_size": tuple(episode["rgb"].shape[-2:])},
    )


def _matched_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    prediction_mask: torch.Tensor,
    target_mask: torch.Tensor,
) -> float:
    predicted = prediction[prediction_mask].detach().cpu()
    truth = target[target_mask].detach().cpu()
    if predicted.numel() == 0 or truth.numel() == 0:
        return float("nan")
    cost = torch.cdist(predicted, truth)
    rows, columns = linear_sum_assignment(cost.numpy())
    return float(cost[rows, columns].mean())


def _project_world(
    position: torch.Tensor,
    active: torch.Tensor,
    world_from_camera: torch.Tensor,
    intrinsics: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    device = position.device
    transform = world_from_camera.to(device=device, dtype=position.dtype)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    camera_position = (position - translation) @ rotation
    depth = camera_position[:, 2]
    projected = camera_position @ intrinsics.to(device=device, dtype=position.dtype).T
    pixels = projected[:, :2] / projected[:, 2:].clamp_min(1.0e-5)
    valid = active & (depth > 1.0e-4) & torch.isfinite(pixels).all(dim=-1)
    return pixels.detach().cpu().numpy(), valid.detach().cpu().numpy()


def create_demo(
    *,
    config: OrpheusConfig,
    checkpoint_path: str,
    output_dir: str | None,
    device_info: DeviceInfo,
) -> dict[str, Any]:
    """Create PNG frames, GIF, parameter plot, and truthful summary JSON."""

    if not config.evaluation.rgb_only or config.runtime.enable_debug_oracle:
        raise ValueError("The primary demo requires RGB-only configuration")
    output = Path(output_dir or f"demo_outputs/seed_{config.demo.seed}")
    frame_dir = output / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    model = OnlineWorldModel.from_config(config, device=device_info.device)
    checkpoint = load_checkpoint(
        checkpoint_path,
        model=model,
        map_location=device_info.device,
        expected_config=config,
    )
    model.eval()
    model.reset()
    episode = generate_episode(config, config.demo.seed)
    total_frames = int(episode["rgb"].shape[0])
    count = min(config.demo.max_frames, total_frames)
    horizon = min(
        config.demo.future_horizon_seconds,
        float(episode["timestamps"][-1] - episode["timestamps"][0]),
    )
    png_paths: list[Path] = []
    prior_errors: list[float] = []
    posterior_errors: list[float] = []
    prior_future_errors: list[float] = []
    posterior_future_errors: list[float] = []
    predicted_collision_probabilities: list[float] = []
    observation_mode_counts: dict[str, int] = {}
    parameter_timestamps: list[float] = []
    estimated_drag: list[float] = []
    estimated_restitution: list[float] = []
    target_drag: list[float] = []
    target_restitution: list[float] = []

    with torch.no_grad():
        for index in range(count):
            packet = _packet(episode, index, device=device_info.device)
            future_index = min(
                total_frames - 1,
                index + round(horizon * config.simulator.frame_rate),
            )
            future_seconds = float(
                episode["timestamps"][future_index] - episode["timestamps"][index]
            )
            previous = model.belief
            prior = None
            prior_future = None
            if previous is not None:
                dt = (
                    previous.timestamp.new_full(
                        previous.timestamp.shape,
                        packet.timestamp,
                    )
                    - previous.timestamp
                )
                prior = model.dynamics.predict(previous, dt)
                prior_future = model.dynamics.rollout(prior, [future_seconds])

            posterior = model.ingest(packet)
            measured = model.last_measurements
            if measured is None:
                raise RuntimeError("runtime did not retain its scheduled RGB measurements")
            posterior_future = model.predict([future_seconds])
            observation_mode = (
                model.diagnostics.latest.observation_mode
                if model.diagnostics.latest is not None
                else "UNKNOWN"
            )
            observation_mode_counts[observation_mode] = (
                observation_mode_counts.get(observation_mode, 0) + 1
            )

            target_position = episode["objects"]["position"][index]
            target_active = episode["objects"]["active"][index]
            future_target_position = episode["objects"]["position"][future_index]
            future_target_active = episode["objects"]["active"][future_index]
            current_prior_error = float("nan")
            future_prior_error = float("nan")
            future_posterior_error = float("nan")
            if prior is not None:
                current_prior_error = _matched_error(
                    prior.objects.position[0],
                    target_position,
                    prior.objects.active[0],
                    target_active,
                )
                prior_errors.append(current_prior_error)
                if future_seconds > 0.0 and prior_future is not None:
                    future_prior_error = _matched_error(
                        prior_future.positions[0, -1],
                        future_target_position,
                        prior_future.active_mask[0, -1],
                        future_target_active,
                    )
                    future_posterior_error = _matched_error(
                        posterior_future.positions[0, -1],
                        future_target_position,
                        posterior_future.active_mask[0, -1],
                        future_target_active,
                    )
                    prior_future_errors.append(future_prior_error)
                    posterior_future_errors.append(future_posterior_error)
            current_posterior_error = _matched_error(
                posterior.objects.position[0],
                target_position,
                posterior.objects.active[0],
                target_active,
            )
            posterior_errors.append(current_posterior_error)
            if posterior_future.event_logits is None:
                predicted_collision_probability = float("nan")
            else:
                event_probability = posterior_future.event_logits[0, -1].softmax(dim=-1)
                future_active = posterior_future.active_mask[0, -1]
                if future_active.any():
                    predicted_collision_probability = float(
                        event_probability[future_active, MotionMode.COLLISION].max().cpu()
                    )
                else:
                    predicted_collision_probability = float("nan")
            predicted_collision_probabilities.append(predicted_collision_probability)

            active = posterior.objects.active[0]
            if active.any():
                estimated_drag.append(float(posterior.objects.drag[0, active].mean().cpu()))
                estimated_restitution.append(
                    float(posterior.objects.restitution[0, active].mean().cpu())
                )
            else:
                estimated_drag.append(float("nan"))
                estimated_restitution.append(float("nan"))
            visible_target = target_active & episode["labels"]["visible"][index].bool()
            if not visible_target.any():
                visible_target = target_active
            target_drag.append(float(episode["objects"]["drag"][index, visible_target].mean()))
            target_restitution.append(
                float(episode["objects"]["restitution"][index, visible_target].mean())
            )
            parameter_timestamps.append(packet.timestamp)

            figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
            image = episode["rgb"][index].permute(1, 2, 0).numpy()
            axes[0].imshow(image)
            measurement_probability = measured.existence_logits[0].sigmoid()
            measurement_pixels = normalized_to_pixels(
                measured.values[0, :, :2].detach().cpu().numpy(),
                tuple(image.shape[:2]),
            )
            overlay_points(
                axes[0],
                measurement_pixels,
                color="yellow",
                marker="x",
                label=f"{observation_mode} measurement",
                valid=(measured.measurement_mask[0] & (measurement_probability > 0.35))
                .detach()
                .cpu()
                .numpy(),
            )
            calibration_world = episode["camera"]["world_from_camera"][index]
            calibration_intrinsics = episode["camera"]["intrinsics"][index]
            if prior is not None:
                prior_pixels, prior_valid = _project_world(
                    prior.objects.position[0],
                    prior.objects.active[0],
                    calibration_world,
                    calibration_intrinsics,
                )
                overlay_points(
                    axes[0],
                    prior_pixels,
                    color="orange",
                    marker="o",
                    label="prior",
                    valid=prior_valid,
                )
            posterior_pixels, posterior_valid = _project_world(
                posterior.objects.position[0],
                posterior.objects.active[0],
                calibration_world,
                calibration_intrinsics,
            )
            overlay_points(
                axes[0],
                posterior_pixels,
                color="lime",
                marker="o",
                label="posterior",
                valid=posterior_valid,
            )
            gt_pixels = episode["labels"]["projected_center_pixels"][index].numpy()
            overlay_points(
                axes[0],
                gt_pixels,
                color="white",
                marker="+",
                label="ground truth",
                valid=target_active.numpy(),
            )
            for slot, valid in enumerate(posterior_valid):
                if not valid:
                    continue
                variance = posterior.objects.fast_log_variance[0, slot, :2].exp()
                sigma = float(variance.mean().sqrt().cpu()) * 20.0
                add_uncertainty_ellipse(
                    axes[0],
                    x=float(posterior_pixels[slot, 0]),
                    y=float(posterior_pixels[slot, 1]),
                    sigma_x_pixels=sigma,
                    sigma_y_pixels=sigma,
                )
            if np.isfinite(current_prior_error):
                current_error_text = (
                    f"prior {current_prior_error:.3f} → posterior {current_posterior_error:.3f} m"
                )
            else:
                current_error_text = f"posterior error {current_posterior_error:.3f} m"
            axes[0].set_title(
                f"RGB-only online step {index} · t={packet.timestamp:.2f}s\n{current_error_text}"
            )
            axes[0].axis("off")
            axes[0].legend(loc="lower right", fontsize=7)

            truth_track = episode["objects"]["position"][index : future_index + 1].numpy()
            truth_active_track = episode["objects"]["active"][index : future_index + 1].numpy()
            plot_xy_trajectory(
                axes[1],
                truth_track,
                truth_active_track,
                color="royalblue",
                label="GT future",
            )
            if prior_future is not None:
                plot_xy_trajectory(
                    axes[1],
                    np.concatenate(
                        (
                            prior.objects.position[:, None].cpu().numpy(),
                            prior_future.positions.cpu().numpy(),
                        ),
                        axis=1,
                    )[0],
                    np.concatenate(
                        (
                            prior.objects.active[:, None].cpu().numpy(),
                            prior_future.active_mask.cpu().numpy(),
                        ),
                        axis=1,
                    )[0],
                    color="orangered",
                    label="prior rollout",
                    linestyle="--",
                )
            plot_xy_trajectory(
                axes[1],
                np.concatenate(
                    (
                        posterior.objects.position[:, None].cpu().numpy(),
                        posterior_future.positions.cpu().numpy(),
                    ),
                    axis=1,
                )[0],
                np.concatenate(
                    (
                        posterior.objects.active[:, None].cpu().numpy(),
                        posterior_future.active_mask.cpu().numpy(),
                    ),
                    axis=1,
                )[0],
                color="green",
                label="posterior rollout",
            )
            axes[1].set_xlabel("world x (m)")
            axes[1].set_ylabel("world y (m)")
            future_error_text = (
                f"future Δ={future_prior_error - future_posterior_error:+.3f} m"
                if np.isfinite(future_prior_error) and np.isfinite(future_posterior_error)
                else "future Δ=n/a"
            )
            axes[1].set_title(
                "future revision "
                f"({future_seconds:.2f}s) · predicted collision "
                f"p={predicted_collision_probability:.2f}\n"
                f"{future_error_text} · "
                "GT collision now="
                f"{bool(episode['events']['collision'][index].any())}"
            )
            axes[1].legend(fontsize=8)
            axes[1].grid(alpha=0.2)
            figure.tight_layout()
            path = frame_dir / f"frame_{index:04d}.png"
            figure.savefig(path, dpi=120)
            plt.close(figure)
            png_paths.append(path)

    images = [Image.open(path).convert("RGB") for path in png_paths]
    gif_path = output / "online_correction.gif"
    if images:
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=max(20, round(1000 / config.demo.fps)),
            loop=0,
        )
    for image in images:
        image.close()
    parameter_plot = save_parameter_plot(
        np.asarray(parameter_timestamps),
        np.asarray(estimated_drag),
        np.asarray(estimated_restitution),
        np.asarray(target_drag),
        np.asarray(target_restitution),
        output / "parameter_estimates.png",
    )
    finite_prior = np.asarray(prior_errors, dtype=float)
    finite_posterior = np.asarray(posterior_errors[1:], dtype=float)
    valid = np.isfinite(finite_prior) & np.isfinite(finite_posterior)
    mean_improvement = (
        float(np.mean(finite_prior[valid] - finite_posterior[valid]))
        if valid.any()
        else float("nan")
    )
    finite_prior_future = np.asarray(prior_future_errors, dtype=float)
    finite_posterior_future = np.asarray(posterior_future_errors, dtype=float)
    future_valid = np.isfinite(finite_prior_future) & np.isfinite(finite_posterior_future)
    mean_future_improvement = (
        float(np.mean(finite_prior_future[future_valid] - finite_posterior_future[future_valid]))
        if future_valid.any()
        else float("nan")
    )
    finite_collision = np.asarray(predicted_collision_probabilities, dtype=float)
    summary = {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_step": int(checkpoint["step"]),
        "seed": config.demo.seed,
        "rgb_only": True,
        "oracle_input": False,
        "device": str(device_info.device),
        "frames": count,
        "mean_current_prior_to_posterior_improvement_m": mean_improvement,
        "mean_future_prior_to_posterior_improvement_m": mean_future_improvement,
        "future_comparisons": int(future_valid.sum()),
        "maximum_predicted_collision_probability": (
            float(np.nanmax(finite_collision))
            if np.isfinite(finite_collision).any()
            else float("nan")
        ),
        "observation_mode_counts": observation_mode_counts,
        "gif": str(gif_path.resolve()),
        "parameter_plot": str(parameter_plot.resolve()),
        "frame_directory": str(frame_dir.resolve()),
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary["summary"] = str(summary_path.resolve())
    return summary
