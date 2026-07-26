"""Sensor-local RGB feature cache."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from world_model.observations.base import ModalityCache


@dataclass
class RGBModalityCache(ModalityCache):
    feature_map: Tensor
    object_features: Tensor
    rois: Tensor
    support: Tensor
    previous_image: Tensor
    timestamp: float
    object_ids: Tensor

    def detach(self) -> RGBModalityCache:
        return RGBModalityCache(
            feature_map=self.feature_map.detach(),
            object_features=self.object_features.detach(),
            rois=self.rois.detach(),
            support=self.support.detach(),
            previous_image=self.previous_image.detach(),
            timestamp=self.timestamp,
            object_ids=self.object_ids.detach(),
        )
