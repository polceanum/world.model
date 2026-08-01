"""Fast projected-ROI measurement updater using ``grid_sample``."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def make_roi_grid(
    rois: Tensor,
    output_size: int,
) -> Tensor:
    """Create ``[B,N,S,S,2]`` grids for normalized ``[x1,y1,x2,y2]`` ROIs."""

    if rois.ndim != 3 or rois.shape[-1] != 4:
        raise ValueError("rois must have shape [B, N, 4]")
    if output_size <= 0:
        raise ValueError("output_size must be positive")
    axis = torch.linspace(
        0.0,
        1.0,
        output_size,
        device=rois.device,
        dtype=rois.dtype,
    )
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    unit = torch.stack((xx, yy), dim=-1)
    minimum = rois[..., :2].unsqueeze(-2).unsqueeze(-2)
    maximum = rois[..., 2:].unsqueeze(-2).unsqueeze(-2)
    return minimum + unit * (maximum - minimum)


def _sample_rois_native_bilinear(
    feature_map: Tensor,
    grid: Tensor,
) -> Tensor:
    """Apply align-corners bilinear sampling without ``grid_sample``.

    This gather-based implementation exists for MPS training, where the
    validated PyTorch build does not provide ``grid_sampler_2d_backward``. It
    preserves zero padding and gradients to both the feature map and sampling
    coordinates while keeping every tensor on the accelerator.
    """

    batch, channels, height, width = feature_map.shape
    if grid.ndim != 5 or grid.shape[0] != batch or grid.shape[-1] != 2:
        raise ValueError("grid must have shape [B, N, S, S, 2]")
    objects, output_height, output_width = grid.shape[1:4]
    x = (grid[..., 0] + 1.0) * ((width - 1) / 2.0)
    y = (grid[..., 1] + 1.0) * ((height - 1) / 2.0)
    x0 = torch.floor(x)
    y0 = torch.floor(y)
    x1 = x0 + 1.0
    y1 = y0 + 1.0
    x_fraction = x - x0
    y_fraction = y - y0
    flattened_features = feature_map.reshape(batch, channels, height * width)

    def gather(x_index: Tensor, y_index: Tensor) -> Tensor:
        valid = (x_index >= 0) & (x_index < width) & (y_index >= 0) & (y_index < height)
        linear_index = (
            y_index.clamp(0, height - 1).to(torch.int64) * width
            + x_index.clamp(0, width - 1).to(torch.int64)
        ).reshape(batch, -1)
        gathered = torch.gather(
            flattened_features,
            dim=2,
            index=linear_index[:, None].expand(-1, channels, -1),
        )
        return gathered * valid.reshape(batch, 1, -1).to(feature_map.dtype)

    top_left = gather(x0, y0)
    top_right = gather(x1, y0)
    bottom_left = gather(x0, y1)
    bottom_right = gather(x1, y1)
    top_left_weight = ((1.0 - x_fraction) * (1.0 - y_fraction)).reshape(batch, 1, -1)
    top_right_weight = (x_fraction * (1.0 - y_fraction)).reshape(batch, 1, -1)
    bottom_left_weight = ((1.0 - x_fraction) * y_fraction).reshape(batch, 1, -1)
    bottom_right_weight = (x_fraction * y_fraction).reshape(batch, 1, -1)
    sampled = (
        top_left * top_left_weight
        + top_right * top_right_weight
        + bottom_left * bottom_left_weight
        + bottom_right * bottom_right_weight
    )
    return (
        sampled.reshape(batch, channels, objects, output_height, output_width)
        .permute(0, 2, 1, 3, 4)
        .contiguous()
    )


def _uses_native_mps_gradient_sampler(
    *,
    training: bool,
    gradient_enabled: bool,
    device_type: str,
) -> bool:
    """Return whether ROI sampling needs the MPS backward-compatible path."""

    return training and gradient_enabled and device_type == "mps"


def sample_rois(
    feature_map: Tensor,
    rois: Tensor,
    output_size: int,
    *,
    training: bool = False,
) -> Tensor:
    """Sample feature crops as ``[B,N,C,S,S]`` without torchvision ops.

    PyTorch's MPS backend supports ``grid_sample`` forward but, as of the
    project's validated PyTorch 2.10 build, not
    ``aten::grid_sampler_2d_backward``. During gradient-enabled MPS training
    only, a gather-based bilinear equivalent keeps the operation and its
    gradients on MPS. CPU, CUDA, and all inference/no-grad calls use
    ``grid_sample`` unchanged.
    """

    if feature_map.ndim != 4:
        raise ValueError("feature_map must have shape [B, C, H, W]")
    batch, channels, _, _ = feature_map.shape
    if rois.shape[0] != batch:
        raise ValueError("ROI batch must match feature map")
    objects = rois.shape[1]
    grid = make_roi_grid(rois, output_size)
    if _uses_native_mps_gradient_sampler(
        training=training,
        gradient_enabled=torch.is_grad_enabled(),
        device_type=feature_map.device.type,
    ):
        return _sample_rois_native_bilinear(feature_map, grid)
    expanded_features = (
        feature_map[:, None]
        .expand(batch, objects, channels, *feature_map.shape[-2:])
        .reshape(batch * objects, channels, *feature_map.shape[-2:])
    )
    flattened_grid = grid.reshape(batch * objects, output_size, output_size, 2)
    sampled = F.grid_sample(
        expanded_features,
        flattened_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled.reshape(batch, objects, channels, output_size, output_size)


@dataclass
class ROIUpdateOutput:
    values: Tensor
    log_variance: Tensor
    existence_logits: Tensor
    visibility_logits: Tensor
    appearance: Tensor
    appearance_gate: Tensor
    object_features: Tensor
    support: Tensor
    event_features: Tensor


class FastROIUpdater(nn.Module):
    """Prior-conditioned per-object residual measurement network."""

    def __init__(
        self,
        *,
        feature_dim: int,
        measurement_dim: int = 7,
        appearance_dim: int = 32,
        roi_size: int = 20,
        hidden_dim: int = 96,
    ) -> None:
        super().__init__()
        self.measurement_dim = measurement_dim
        self.appearance_dim = appearance_dim
        self.roi_size = roi_size
        input_channels = feature_dim + 3
        self.roi_network = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, 3, padding=1),
            nn.GroupNorm(8 if hidden_dim % 8 == 0 else 1, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1),
            nn.GroupNorm(8 if hidden_dim % 8 == 0 else 1, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1),
            nn.GroupNorm(8 if hidden_dim % 8 == 0 else 1, hidden_dim),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.previous_projection = nn.Linear(feature_dim, hidden_dim)
        self.update_network = nn.Sequential(
            nn.Linear(hidden_dim + measurement_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.delta_head = nn.Linear(hidden_dim, measurement_dim)
        self.variance_head = nn.Linear(hidden_dim, measurement_dim)
        self.existence_head = nn.Linear(hidden_dim, 1)
        self.visibility_head = nn.Linear(hidden_dim, 1)
        self.appearance_head = nn.Linear(hidden_dim, appearance_dim)
        self.appearance_gate_head = nn.Linear(hidden_dim, 1)
        self.event_head = nn.Linear(hidden_dim, 8)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.constant_(self.variance_head.bias, -2.5)

    @staticmethod
    def _support_and_coordinates(
        batch: int,
        objects: int,
        size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        axis = torch.linspace(-1.0, 1.0, size, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        radius_squared = xx.square() + yy.square()
        support = torch.sigmoid((1.0 - radius_squared) * 8.0)
        coordinates = torch.stack((xx, yy), dim=0)
        return (
            support.reshape(1, 1, 1, size, size).expand(batch, objects, -1, -1, -1),
            coordinates.reshape(1, 1, 2, size, size).expand(batch, objects, -1, -1, -1),
        )

    def forward(
        self,
        feature_map: Tensor,
        rois: Tensor,
        predicted_values: Tensor,
        *,
        previous_object_features: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> ROIUpdateOutput:
        crops = sample_rois(
            feature_map,
            rois,
            self.roi_size,
            training=self.training,
        )
        batch, objects, _, size, _ = crops.shape
        support, coordinates = self._support_and_coordinates(
            batch,
            objects,
            size,
            device=crops.device,
            dtype=crops.dtype,
        )
        augmented = torch.cat((crops, support, coordinates), dim=2)
        roi_features = self.roi_network(
            augmented.reshape(batch * objects, augmented.shape[2], size, size)
        ).flatten(1)
        roi_features = roi_features.reshape(batch, objects, -1)
        if previous_object_features is None:
            previous = torch.zeros(
                batch,
                objects,
                self.previous_projection.in_features,
                device=crops.device,
                dtype=crops.dtype,
            )
        else:
            previous = previous_object_features
            if previous.shape[:2] != (batch, objects):
                previous = torch.zeros(
                    batch,
                    objects,
                    self.previous_projection.in_features,
                    device=crops.device,
                    dtype=crops.dtype,
                )
            if previous.shape[-1] != self.previous_projection.in_features:
                previous = torch.zeros_like(previous[..., :1]).expand(
                    batch, objects, self.previous_projection.in_features
                )
        hidden = self.update_network(
            torch.cat(
                (
                    roi_features,
                    predicted_values,
                    self.previous_projection(previous),
                ),
                dim=-1,
            )
        )
        delta = torch.tanh(self.delta_head(hidden))
        # Geometric changes are bounded tightly; colour can react faster.
        scale = predicted_values.new_tensor([0.25, 0.25, 0.35, 0.5, 0.25, 0.25, 0.25])
        values = predicted_values + delta * scale
        values = torch.cat(
            (
                # A projected ROI may remain valid while its sphere centre is
                # outside the image; the visible rim still carries residual
                # evidence. The delta is already bounded, so clipping the
                # absolute centre here silently moved such priors toward the
                # frame and created impossible off-crop supervision.
                values[..., :2],
                values[..., 2:3].clamp(-8.0, 1.0),
                values[..., 3:4].clamp(1.0e-3, 20.0),
                values[..., 4:7].clamp(0.0, 1.0),
            ),
            dim=-1,
        )
        appearance_delta = F.normalize(self.appearance_head(hidden), dim=-1)
        appearance_gate = torch.sigmoid(self.appearance_gate_head(hidden))
        if valid_mask is not None:
            mask = valid_mask.unsqueeze(-1)
            values = torch.where(mask, values, predicted_values)
            hidden = hidden * mask
        return ROIUpdateOutput(
            values=values,
            log_variance=self.variance_head(hidden).clamp(-8.0, 3.0),
            existence_logits=self.existence_head(hidden).squeeze(-1),
            visibility_logits=self.visibility_head(hidden).squeeze(-1),
            appearance=appearance_delta,
            appearance_gate=appearance_gate,
            object_features=crops.mean(dim=(-1, -2)),
            support=support.squeeze(2),
            event_features=self.event_head(hidden),
        )
