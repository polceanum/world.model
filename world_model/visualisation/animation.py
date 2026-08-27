"""Held-out no-oracle visual prior/posterior demonstration."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from PIL import Image
from scipy.optimize import linear_sum_assignment

from world_model.belief import MotionMode
from world_model.observations import ObservationPacket
from world_model.runtime import OnlineWorldModel
from world_model.simulator import generate_episode
from world_model.training.checkpointing import load_checkpoint
from world_model.utils.artifacts import timestamped_artifact_path
from world_model.utils.config import OrpheusConfig
from world_model.utils.device import DeviceInfo
from world_model.visualisation.frames import normalized_to_pixels, overlay_points
from world_model.visualisation.plots import save_parameter_plot
from world_model.visualisation.trajectories import plot_xy_trajectory
from world_model.visualisation.uncertainty import add_uncertainty_ellipse

_GROUND_TRUTH_COLORS = (
    "royalblue",
    "darkorange",
    "mediumpurple",
    "deeppink",
    "saddlebrown",
    "teal",
    "olive",
    "slateblue",
)


@dataclass(frozen=True)
class _PositionMatches:
    """Evaluation-only endpoint matches used by the demo overlay."""

    prediction_indices: np.ndarray
    target_indices: np.ndarray
    prediction_points: np.ndarray
    target_points: np.ndarray
    distances: np.ndarray

    @property
    def mean_error(self) -> float:
        return float(self.distances.mean()) if self.distances.size else float("nan")


@dataclass(frozen=True)
class _ForecastTrace:
    """One posterior forecast retained in absolute world coordinates."""

    anchor_index: int
    anchor_timestamp: float
    positions: np.ndarray
    active: np.ndarray


def _demo_generation_config(config: OrpheusConfig) -> tuple[OrpheusConfig, int]:
    """Reserve label-only lookahead so every displayed frame has a full forecast."""

    display_count = min(config.demo.max_frames, config.simulator.sequence_frames)
    lookahead_frames = math.ceil(config.demo.future_horizon_seconds * config.simulator.frame_rate)
    generated_frames = max(
        config.simulator.sequence_frames,
        display_count + lookahead_frames,
    )
    return (
        replace(
            config,
            simulator=replace(
                config.simulator,
                sequence_frames=generated_frames,
            ),
        ),
        display_count,
    )


def _packet(
    episode: dict[str, Any],
    index: int,
    *,
    modality: str,
    device: torch.device,
) -> ObservationPacket:
    if modality not in {"rgb", "rgbd"}:
        raise ValueError("demo packets support only rgb or rgbd")
    if modality == "rgbd":
        payload: object = {
            "rgb": episode["rgb"][index].unsqueeze(0).to(device),
            "depth": episode["depth"][index].unsqueeze(0).to(device),
        }
        sensor_id = "camera0:rgbd"
        frame_id = "camera:camera0:rgbd"
        intrinsics = episode["camera"]["intrinsics"][index].unsqueeze(0).to(device)
        world_from_camera = episode["camera"]["world_from_camera"][index].unsqueeze(0).to(device)
    else:
        payload = episode["rgb"][index].to(device)
        sensor_id = "camera0"
        frame_id = "camera:camera0"
        intrinsics = episode["camera"]["intrinsics"][index].to(device)
        world_from_camera = episode["camera"]["world_from_camera"][index].to(device)
    return ObservationPacket(
        modality=modality,
        sensor_id=sensor_id,
        timestamp=float(episode["timestamps"][index]),
        payload=payload,
        calibration={
            "intrinsics": intrinsics,
            "world_from_camera": world_from_camera,
        },
        frame_id=frame_id,
        metadata={
            "image_size": tuple(episode["rgb"].shape[-2:]),
            "depth_semantics": (
                "observable_camera_z_surface_depth_zero_means_no_return"
                if modality == "rgbd"
                else "not_present"
            ),
        },
    )


def _match_positions(
    prediction: torch.Tensor,
    target: torch.Tensor,
    prediction_mask: torch.Tensor,
    target_mask: torch.Tensor,
) -> _PositionMatches:
    """Hungarian-match endpoints without exposing labels to the runtime."""

    if prediction.ndim != 2 or target.ndim != 2 or prediction.shape[-1] != target.shape[-1]:
        raise ValueError("prediction and target must have compatible [N,D] shapes")
    if prediction_mask.shape != prediction.shape[:1]:
        raise ValueError("prediction_mask must match the prediction object axis")
    if target_mask.shape != target.shape[:1]:
        raise ValueError("target_mask must match the target object axis")
    prediction_indices = torch.nonzero(prediction_mask, as_tuple=False).flatten().detach().cpu()
    target_indices = torch.nonzero(target_mask, as_tuple=False).flatten().detach().cpu()
    predicted = prediction.index_select(0, prediction_indices.to(prediction.device)).detach().cpu()
    truth = target.index_select(0, target_indices.to(target.device)).detach().cpu()
    if predicted.numel() == 0 or truth.numel() == 0:
        dimension = int(prediction.shape[-1])
        return _PositionMatches(
            prediction_indices=np.empty(0, dtype=np.int64),
            target_indices=np.empty(0, dtype=np.int64),
            prediction_points=np.empty((0, dimension), dtype=np.float32),
            target_points=np.empty((0, dimension), dtype=np.float32),
            distances=np.empty(0, dtype=np.float32),
        )
    cost = torch.cdist(predicted, truth)
    rows, columns = linear_sum_assignment(cost.numpy())
    row_tensor = torch.as_tensor(rows, dtype=torch.int64)
    column_tensor = torch.as_tensor(columns, dtype=torch.int64)
    return _PositionMatches(
        prediction_indices=prediction_indices[row_tensor].numpy(),
        target_indices=target_indices[column_tensor].numpy(),
        prediction_points=predicted[row_tensor].numpy(),
        target_points=truth[column_tensor].numpy(),
        distances=cost[row_tensor, column_tensor].numpy(),
    )


def _matched_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    prediction_mask: torch.Tensor,
    target_mask: torch.Tensor,
) -> float:
    return _match_positions(prediction, target, prediction_mask, target_mask).mean_error


def _future_query_seconds(
    timestamps: torch.Tensor,
    anchor_index: int,
    future_index: int,
) -> list[float]:
    """Return observation-aligned positive offsets for a displayed rollout."""

    if timestamps.ndim != 1:
        raise ValueError("demo timestamps must have shape [T]")
    if not 0 <= anchor_index <= future_index < timestamps.shape[0]:
        raise IndexError("demo forecast indices are outside the timestamp axis")
    anchor = float(timestamps[anchor_index])
    return [
        float(timestamps[index]) - anchor for index in range(anchor_index + 1, future_index + 1)
    ]


def _history_alpha(index: int, count: int) -> float:
    """Fade older forecasts while keeping every retained trace visible."""

    if count <= 0 or not 0 <= index < count:
        raise ValueError("history alpha index must lie inside a nonempty history")
    recency = (index + 1) / count
    return 0.04 + 0.24 * recency


def _plot_historical_forecasts(
    axis: Any,
    forecasts: list[_ForecastTrace],
) -> list[Line2D]:
    """Plot forecasts at their original absolute anchors with recency fading."""

    lines: list[Line2D] = []
    for index, forecast in enumerate(forecasts):
        lines.extend(
            plot_xy_trajectory(
                axis,
                forecast.positions,
                forecast.active,
                color="seagreen",
                label=None,
                alpha=_history_alpha(index, len(forecasts)),
                linewidth=0.8,
                zorder=2,
            )
        )
    return lines


def _plot_ground_truth_window(
    axis: Any,
    positions: np.ndarray,
    active: np.ndarray,
    object_ids: np.ndarray,
    *,
    current_index: int,
    future_index: int,
) -> list[Line2D]:
    """Draw past and current-horizon truth once, with identity and time direction."""

    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("ground-truth positions must have shape [T,N,3]")
    if active.shape != positions.shape[:2]:
        raise ValueError("ground-truth active mask must have shape [T,N]")
    if object_ids.shape != positions.shape[1:2]:
        raise ValueError("ground-truth object IDs must have shape [N]")
    if not 0 <= current_index <= future_index < positions.shape[0]:
        raise IndexError("ground-truth plot indices are outside the time axis")

    lines: list[Line2D] = []
    for slot in range(positions.shape[1]):
        color = _GROUND_TRUTH_COLORS[slot % len(_GROUND_TRUTH_COLORS)]
        past_valid = active[: current_index + 1, slot] & np.isfinite(
            positions[: current_index + 1, slot]
        ).all(axis=-1)
        if past_valid.sum() >= 2:
            lines.extend(
                axis.plot(
                    positions[: current_index + 1, slot, 0][past_valid],
                    positions[: current_index + 1, slot, 1][past_valid],
                    color=color,
                    linestyle=":",
                    linewidth=0.9,
                    alpha=0.28,
                    zorder=1,
                )
            )

        future_positions = positions[current_index : future_index + 1, slot]
        future_valid = active[current_index : future_index + 1, slot] & np.isfinite(
            future_positions
        ).all(axis=-1)
        future_points = future_positions[future_valid]
        if future_points.size == 0:
            continue
        lines.extend(
            axis.plot(
                future_points[:, 0],
                future_points[:, 1],
                color=color,
                linestyle="-",
                marker=".",
                markersize=3.0,
                linewidth=1.6,
                alpha=0.95,
                zorder=3,
            )
        )
        axis.scatter(
            future_points[0, 0],
            future_points[0, 1],
            marker="s",
            s=24,
            facecolors="white",
            edgecolors=color,
            linewidths=1.0,
            zorder=7,
        )
        axis.scatter(
            future_points[-1, 0],
            future_points[-1, 1],
            marker="x",
            s=28,
            color=color,
            linewidths=1.2,
            zorder=7,
        )
        for point_index in range(future_points.shape[0] - 1, 0, -1):
            previous = future_points[point_index - 1]
            current = future_points[point_index]
            if np.linalg.norm(current[:2] - previous[:2]) <= 1.0e-8:
                continue
            axis.annotate(
                "",
                xy=current[:2],
                xytext=previous[:2],
                arrowprops={
                    "arrowstyle": "->",
                    "color": color,
                    "linewidth": 1.1,
                    "alpha": 0.95,
                    "shrinkA": 0.0,
                    "shrinkB": 0.0,
                },
                zorder=8,
            )
            break
        object_id = int(object_ids[slot])
        if object_id >= 0:
            axis.annotate(
                f"GT {object_id}",
                xy=future_points[0, :2],
                xytext=(4, 5),
                textcoords="offset points",
                color=color,
                fontsize=6.5,
                weight="bold",
                zorder=8,
            )
    return lines


def _configure_world_axis(
    axis: Any,
    world_bounds: tuple[tuple[float, float], ...],
) -> None:
    """Fix the world panel geometry so GIF frames do not jump or rescale."""

    if len(world_bounds) < 2:
        raise ValueError("world bounds must provide x and y limits")
    x_min, x_max = world_bounds[0]
    y_min, y_max = world_bounds[1]
    x_padding = 0.04 * (x_max - x_min)
    y_padding = 0.04 * (y_max - y_min)
    axis.set_xlim(x_min - x_padding, x_max + x_padding)
    axis.set_ylim(y_min - y_padding, y_max + y_padding)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("world x (m)")
    axis.set_ylabel("world y (m)")
    axis.grid(alpha=0.2)


def _add_image_legend(axis: Any, *, modality: str = "rgb") -> None:
    if modality not in {"rgb", "rgbd"}:
        raise ValueError("image legend supports only rgb or rgbd")
    measurement_label = (
        "scheduled RGB-D measurement" if modality == "rgbd" else "scheduled RGB measurement"
    )
    handles = [
        Line2D(
            [],
            [],
            color="yellow",
            marker="x",
            linestyle="None",
            label=measurement_label,
        ),
        Line2D(
            [],
            [],
            color="orange",
            marker="o",
            markerfacecolor="none",
            linestyle="None",
            label="prior",
        ),
        Line2D(
            [],
            [],
            color="lime",
            marker="o",
            markerfacecolor="none",
            linestyle="None",
            label="posterior",
        ),
        Ellipse(
            (0.0, 0.0),
            width=1.8,
            height=1.0,
            fill=False,
            edgecolor="lime",
            linewidth=0.8,
            alpha=0.7,
            label="posterior 90% position uncertainty",
        ),
        Line2D(
            [],
            [],
            color="white",
            marker="+",
            linestyle="None",
            label="ground truth overlay",
        ),
    ]
    axis.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.85)


def _add_world_legend(axis: Any) -> None:
    handles = [
        Line2D(
            [],
            [],
            color="royalblue",
            linestyle=":",
            linewidth=1.0,
            alpha=0.45,
            label="GT past (through now)",
        ),
        Line2D(
            [],
            [],
            color="royalblue",
            marker=".",
            linewidth=1.6,
            label="GT current horizon (object colours)",
        ),
        Line2D(
            [],
            [],
            color="seagreen",
            linewidth=1.0,
            alpha=0.35,
            label="historical posterior forecasts",
        ),
        Line2D(
            [],
            [],
            color="orangered",
            linestyle="--",
            linewidth=1.2,
            label="latest prior forecast",
        ),
        Line2D([], [], color="green", linewidth=1.6, label="latest posterior forecast"),
        Line2D(
            [],
            [],
            color="black",
            marker="o",
            markerfacecolor="none",
            linestyle=":",
            linewidth=0.8,
            label="posterior endpoint ↔ matched GT",
        ),
    ]
    axis.legend(handles=handles, loc="upper right", fontsize=6.5, framealpha=0.9)


def _draw_endpoint_matches(axis: Any, matches: _PositionMatches) -> None:
    for predicted, target in zip(
        matches.prediction_points,
        matches.target_points,
        strict=True,
    ):
        axis.plot(
            [predicted[0], target[0]],
            [predicted[1], target[1]],
            color="black",
            linestyle=":",
            linewidth=0.8,
            alpha=0.8,
            zorder=6,
        )
    if matches.prediction_points.size:
        axis.scatter(
            matches.prediction_points[:, 0],
            matches.prediction_points[:, 1],
            marker="o",
            facecolors="none",
            edgecolors="green",
            linewidths=1.2,
            zorder=7,
        )
        axis.scatter(
            matches.target_points[:, 0],
            matches.target_points[:, 1],
            marker="x",
            color="royalblue",
            linewidths=1.2,
            zorder=7,
        )


def _optional_number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


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


def _measurement_image_pixels(
    values: torch.Tensor,
    measurement_mask: torch.Tensor,
    *,
    modality: str,
    image_size: tuple[int, int],
    world_from_camera: torch.Tensor,
    intrinsics: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Map modality-owned measurement coordinates into image pixels."""

    if values.ndim != 2 or measurement_mask.shape != values.shape[:1]:
        raise ValueError("measurement values/mask must have shapes [M,D] and [M]")
    if modality == "rgb":
        if values.shape[-1] < 2:
            raise ValueError("RGB measurements must provide normalized image coordinates")
        pixels = normalized_to_pixels(values[:, :2].detach().cpu().numpy(), image_size)
        valid = measurement_mask & torch.isfinite(values[:, :2]).all(dim=-1)
        return pixels, valid.detach().cpu().numpy()
    if modality == "rgbd":
        if values.shape[-1] != 3:
            raise ValueError("RGB-D measurements must provide metric world positions")
        return _project_world(
            values,
            measurement_mask,
            world_from_camera,
            intrinsics,
        )
    raise ValueError("measurement projection supports only rgb or rgbd")


def _project_world_uncertainty(
    position: torch.Tensor,
    position_log_variance: torch.Tensor,
    active: torch.Tensor,
    world_from_camera: torch.Tensor,
    intrinsics: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project diagonal world-position covariance into image pixels."""

    if position.shape != position_log_variance.shape or position.shape[-1] != 3:
        raise ValueError("position and log variance must have matching [N,3] shapes")
    transform = world_from_camera.to(device=position.device, dtype=position.dtype)
    calibration = intrinsics.to(device=position.device, dtype=position.dtype)
    rotation = transform[:3, :3]
    camera_position = (position - transform[:3, 3]) @ rotation
    homogeneous = camera_position @ calibration.T
    denominator = homogeneous[:, 2].clamp_min(1.0e-5)
    denominator_squared = denominator.square()
    jacobian_camera = torch.stack(
        (
            (
                calibration[0].unsqueeze(0) * denominator.unsqueeze(-1)
                - homogeneous[:, 0:1] * calibration[2].unsqueeze(0)
            )
            / denominator_squared.unsqueeze(-1),
            (
                calibration[1].unsqueeze(0) * denominator.unsqueeze(-1)
                - homogeneous[:, 1:2] * calibration[2].unsqueeze(0)
            )
            / denominator_squared.unsqueeze(-1),
        ),
        dim=1,
    )
    jacobian_world = jacobian_camera @ rotation.T
    world_covariance = torch.diag_embed(position_log_variance.exp())
    pixel_covariance = jacobian_world @ world_covariance @ jacobian_world.transpose(-1, -2)
    eigenvalues, eigenvectors = torch.linalg.eigh(pixel_covariance)
    eigenvalues = eigenvalues.clamp_min(0.0)
    minor_sigma = eigenvalues[:, 0].sqrt()
    major_sigma = eigenvalues[:, 1].sqrt()
    major_vector = eigenvectors[:, :, 1]
    angle_degrees = torch.rad2deg(torch.atan2(major_vector[:, 1], major_vector[:, 0]))
    sigma = torch.stack((major_sigma, minor_sigma), dim=-1)
    valid = (
        active
        & (camera_position[:, 2] > 1.0e-4)
        & torch.isfinite(sigma).all(dim=-1)
        & torch.isfinite(angle_degrees)
    )
    return (
        sigma.detach().cpu().numpy(),
        angle_degrees.detach().cpu().numpy(),
        valid.detach().cpu().numpy(),
    )


def create_demo(
    *,
    config: OrpheusConfig,
    checkpoint_path: str,
    output_dir: str | None,
    device_info: DeviceInfo,
) -> dict[str, Any]:
    """Create PNG frames, GIF, parameter plot, and truthful summary JSON."""

    expected_rgb_only = config.runtime.modality == "rgb"
    if (
        config.evaluation.rgb_only is not expected_rgb_only
        or config.runtime.enable_debug_oracle
        or config.runtime.modality not in {"rgb", "rgbd"}
    ):
        raise ValueError("The primary demo requires no-oracle RGB or RGB-D configuration")
    output = timestamped_artifact_path(output_dir or f"demo_outputs/seed_{config.demo.seed}")
    frame_dir = output / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    model = OnlineWorldModel.from_config(config, device=device_info.device)
    checkpoint = load_checkpoint(
        checkpoint_path,
        model=model,
        # Demo checkpoints may contain a full optimizer. Deserialize on CPU
        # and copy only model weights to the configured runtime devices.
        map_location="cpu",
        expected_config=config,
    )
    model.eval()
    model.reset()
    generation_config, count = _demo_generation_config(config)
    episode = generate_episode(generation_config, config.demo.seed)
    total_frames = int(episode["rgb"].shape[0])
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
    forecast_history: list[_ForecastTrace] = []
    per_frame_metrics: list[dict[str, Any]] = []

    with torch.no_grad():
        for index in range(count):
            packet = _packet(
                episode,
                index,
                modality=config.runtime.modality,
                device=device_info.device,
            )
            future_index = min(
                total_frames - 1,
                index + round(horizon * config.simulator.frame_rate),
            )
            query_seconds = _future_query_seconds(
                episode["timestamps"],
                index,
                future_index,
            )
            future_seconds = query_seconds[-1] if query_seconds else 0.0
            rollout_queries = query_seconds if query_seconds else [0.0]
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
                prior_future = model.dynamics.rollout(prior, rollout_queries)

            posterior = model.ingest(packet)
            measured = model.last_measurements
            if measured is None:
                raise RuntimeError("runtime did not retain its scheduled visual measurements")
            posterior_future = model.predict(rollout_queries)
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
            future_matches: _PositionMatches | None = None
            if future_seconds > 0.0:
                future_matches = _match_positions(
                    posterior_future.positions[0, -1],
                    future_target_position,
                    posterior_future.active_mask[0, -1],
                    future_target_active,
                )
                future_posterior_error = future_matches.mean_error
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
                event_probability = posterior_future.event_logits[0].softmax(dim=-1)
                future_active = posterior_future.active_mask[0]
                if future_active.any():
                    predicted_collision_probability = float(
                        event_probability[..., MotionMode.COLLISION]
                        .masked_select(future_active)
                        .max()
                        .cpu()
                    )
                else:
                    predicted_collision_probability = float("nan")
            predicted_collision_probabilities.append(predicted_collision_probability)
            if future_seconds > 0.0:
                forecast_history.append(
                    _ForecastTrace(
                        anchor_index=index,
                        anchor_timestamp=packet.timestamp,
                        positions=np.concatenate(
                            (
                                posterior.objects.position[:, None].cpu().numpy(),
                                posterior_future.positions.cpu().numpy(),
                            ),
                            axis=1,
                        )[0],
                        active=np.concatenate(
                            (
                                posterior.objects.active[:, None].cpu().numpy(),
                                posterior_future.active_mask.cpu().numpy(),
                            ),
                            axis=1,
                        )[0],
                    )
                )

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
            per_frame_metrics.append(
                {
                    "frame_index": index,
                    "timestamp_seconds": packet.timestamp,
                    "forecast_horizon_seconds": future_seconds,
                    "observation_mode": observation_mode,
                    "current_prior_error_m": _optional_number(current_prior_error),
                    "current_posterior_error_m": _optional_number(current_posterior_error),
                    "future_prior_error_m": _optional_number(future_prior_error),
                    "future_posterior_error_m": _optional_number(future_posterior_error),
                    "future_matched_pairs": (
                        int(future_matches.distances.size) if future_matches is not None else 0
                    ),
                    "future_target_objects": int(future_target_active.sum()),
                }
            )

            figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
            image = episode["rgb"][index].permute(1, 2, 0).numpy()
            axes[0].imshow(image)
            calibration_world = episode["camera"]["world_from_camera"][index]
            calibration_intrinsics = episode["camera"]["intrinsics"][index]
            measurement_probability = measured.existence_logits[0].sigmoid()
            measurement_pixels, measurement_projection_valid = _measurement_image_pixels(
                measured.values[0],
                measured.measurement_mask[0],
                modality=config.runtime.modality,
                image_size=tuple(image.shape[:2]),
                world_from_camera=calibration_world,
                intrinsics=calibration_intrinsics,
            )
            overlay_points(
                axes[0],
                measurement_pixels,
                color="yellow",
                marker="x",
                label=(
                    "scheduled RGB-D measurement"
                    if config.runtime.modality == "rgbd"
                    else "scheduled RGB measurement"
                ),
                valid=(
                    measurement_projection_valid
                    & (measurement_probability > 0.35).detach().cpu().numpy()
                ),
            )
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
            posterior_sigma, posterior_angle, uncertainty_valid = _project_world_uncertainty(
                posterior.objects.position[0],
                posterior.objects.fast_log_variance[0, :, :3],
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
            for slot, valid in enumerate(posterior_valid & uncertainty_valid):
                if not valid:
                    continue
                add_uncertainty_ellipse(
                    axes[0],
                    x=float(posterior_pixels[slot, 0]),
                    y=float(posterior_pixels[slot, 1]),
                    sigma_x_pixels=float(posterior_sigma[slot, 0]),
                    sigma_y_pixels=float(posterior_sigma[slot, 1]),
                    angle_degrees=float(posterior_angle[slot]),
                )
            if np.isfinite(current_prior_error):
                current_error_text = (
                    f"prior {current_prior_error:.3f} → posterior {current_posterior_error:.3f} m"
                )
            else:
                current_error_text = f"posterior error {current_posterior_error:.3f} m"
            axes[0].set_title(
                f"{'RGB-D' if config.runtime.modality == 'rgbd' else 'RGB'} "
                f"no-oracle online step {index} · "
                f"t={packet.timestamp:.2f}s · {observation_mode}\n"
                f"{current_error_text}"
            )
            axes[0].axis("off")
            _add_image_legend(axes[0], modality=config.runtime.modality)

            # Simulator labels below are evaluation/overlay data only. They are
            # read after RGB ingest and are never passed into the runtime.
            _plot_ground_truth_window(
                axes[1],
                episode["objects"]["position"].numpy(),
                episode["objects"]["active"].numpy(),
                episode["objects"]["id"][index].numpy(),
                current_index=index,
                future_index=future_index,
            )
            historical_forecasts = forecast_history
            if forecast_history and forecast_history[-1].anchor_index == index:
                historical_forecasts = forecast_history[:-1]
            _plot_historical_forecasts(axes[1], historical_forecasts)
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
                    label=None,
                    linestyle="--",
                    linewidth=1.2,
                    zorder=4,
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
                label=None,
                linewidth=1.6,
                zorder=5,
            )
            if future_matches is not None:
                _draw_endpoint_matches(axes[1], future_matches)
            if future_matches is not None and np.isfinite(future_posterior_error):
                future_error_lines = [
                    f"matched endpoints {future_matches.distances.size}/"
                    f"{int(future_target_active.sum())}",
                    f"posterior {future_posterior_error:.3f} m",
                ]
                if np.isfinite(future_prior_error):
                    future_error_lines[-1] += f" · prior {future_prior_error:.3f} m"
                    future_error_lines.append(
                        f"correction gain {future_prior_error - future_posterior_error:+.3f} m"
                    )
                else:
                    future_error_lines.append("prior/correction gain n/a at initialisation")
                future_error_text = "\n".join(future_error_lines)
            else:
                future_error_text = "matched future endpoint error: n/a"
            axes[1].text(
                0.02,
                0.98,
                future_error_text,
                transform=axes[1].transAxes,
                ha="left",
                va="top",
                fontsize=7.5,
                family="monospace",
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": "white",
                    "edgecolor": "0.7",
                    "alpha": 0.88,
                },
            )
            collision_probability_text = (
                f"{predicted_collision_probability:.2f}"
                if np.isfinite(predicted_collision_probability)
                else "n/a"
            )
            gt_collision_in_horizon = bool(
                episode["events"]["collision"][index + 1 : future_index + 1].any()
            )
            axes[1].set_title(
                f"recursive forecast · horizon {future_seconds:.2f}s · "
                f"collision p={collision_probability_text}\n"
                f"GT collision in horizon={gt_collision_in_horizon} · "
                f"retained forecasts={len(forecast_history)}"
            )
            _configure_world_axis(axes[1], config.simulator.world_bounds)
            _add_world_legend(axes[1])
            # Fixed margins avoid frame-to-frame panel movement from changing
            # title, legend, or numeric annotation widths.
            figure.subplots_adjust(
                left=0.035,
                right=0.98,
                bottom=0.11,
                top=0.82,
                wspace=0.22,
            )
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
    mean_current_prior_error = float(np.mean(finite_prior[valid])) if valid.any() else float("nan")
    mean_current_posterior_error = (
        float(np.mean(finite_posterior[valid])) if valid.any() else float("nan")
    )
    mean_improvement = (
        float(np.mean(finite_prior[valid] - finite_posterior[valid]))
        if valid.any()
        else float("nan")
    )
    finite_prior_future = np.asarray(prior_future_errors, dtype=float)
    finite_posterior_future = np.asarray(posterior_future_errors, dtype=float)
    future_valid = np.isfinite(finite_prior_future) & np.isfinite(finite_posterior_future)
    mean_future_prior_error = (
        float(np.mean(finite_prior_future[future_valid])) if future_valid.any() else float("nan")
    )
    mean_future_posterior_error = (
        float(np.mean(finite_posterior_future[future_valid]))
        if future_valid.any()
        else float("nan")
    )
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
        "rgb_only": config.runtime.modality == "rgb",
        "observation_modality": config.runtime.modality,
        "oracle_input": False,
        "device": str(device_info.device),
        "frames": count,
        "requested_forecast_horizon_seconds": config.demo.future_horizon_seconds,
        "undisplayed_lookahead_frames": total_frames - count,
        "current_comparisons": int(valid.sum()),
        "mean_current_prior_error_m": mean_current_prior_error,
        "mean_current_posterior_error_m": mean_current_posterior_error,
        "mean_current_prior_to_posterior_improvement_m": mean_improvement,
        "mean_future_prior_error_m": mean_future_prior_error,
        "mean_future_posterior_error_m": mean_future_posterior_error,
        "mean_future_prior_to_posterior_improvement_m": mean_future_improvement,
        "future_comparisons": int(future_valid.sum()),
        "maximum_predicted_collision_probability": (
            float(np.nanmax(finite_collision))
            if np.isfinite(finite_collision).any()
            else float("nan")
        ),
        "observation_mode_counts": observation_mode_counts,
        "retained_posterior_forecasts": len(forecast_history),
        "ground_truth_usage": "demo scoring and overlay only; never runtime input",
        "per_frame_metrics": per_frame_metrics,
        "gif": str(gif_path.resolve()),
        "parameter_plot": str(parameter_plot.resolve()),
        "frame_directory": str(frame_dir.resolve()),
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary["summary"] = str(summary_path.resolve())
    return summary
