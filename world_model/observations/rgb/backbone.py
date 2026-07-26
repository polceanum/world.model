"""Small pure-PyTorch RGB feature extractor."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ConvStage(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int, stride: int) -> None:
        super().__init__(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
            ),
            nn.GroupNorm(_group_count(output_channels), output_channels),
            nn.SiLU(),
            nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(_group_count(output_channels), output_channels),
            nn.SiLU(),
        )


class RGBBackbone(nn.Module):
    """Four-stage feature pyramid with a cheaper two-stage fast path."""

    def __init__(
        self,
        channels: tuple[int, int, int, int] = (32, 64, 96, 128),
        pyramid_channels: int = 64,
    ) -> None:
        super().__init__()
        if len(channels) != 4 or min(channels) <= 0:
            raise ValueError("RGB backbone requires four positive channel sizes")
        c1, c2, c3, c4 = channels
        self.stages = nn.ModuleList(
            (
                ConvStage(3, c1, stride=1),
                ConvStage(c1, c2, stride=2),
                ConvStage(c2, c3, stride=2),
                ConvStage(c3, c4, stride=2),
            )
        )
        self.projections = nn.ModuleList(
            nn.Conv2d(channels_in, pyramid_channels, kernel_size=1) for channels_in in channels
        )
        self.fast_projection = nn.Conv2d(c2, pyramid_channels, kernel_size=1)
        self.output_channels = pyramid_channels

    @staticmethod
    def _validate(image: Tensor) -> None:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("RGB input must have shape [B, 3, H, W]")
        if not image.is_floating_point():
            raise TypeError("RGB input must be floating point")
        if not torch.isfinite(image).all():
            raise ValueError("RGB input contains NaN or Inf")

    def forward(self, image: Tensor) -> dict[str, Tensor]:
        self._validate(image)
        features: list[Tensor] = []
        value = image
        for stage in self.stages:
            value = stage(value)
            features.append(value)
        projected = [
            projection(feature)
            for projection, feature in zip(self.projections, features, strict=True)
        ]
        fused = projected[-1]
        for level in reversed(projected[:-1]):
            fused = F.interpolate(
                fused,
                size=level.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            fused = fused + level
        fused = fused / float(len(projected))
        return {
            "full": fused,
            "coarse": projected[-1],
            "stage2": projected[1],
        }

    def forward_fast(self, image: Tensor) -> dict[str, Tensor]:
        """Compute only features needed by projected ROI updates."""

        self._validate(image)
        stage1 = self.stages[0](image)
        stage2 = self.stages[1](stage1)
        return {"stage2": self.fast_projection(stage2)}
