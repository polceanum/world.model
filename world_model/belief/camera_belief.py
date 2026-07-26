"""Camera pose and calibration belief."""

from __future__ import annotations

from dataclasses import dataclass, replace

from torch import Tensor

from world_model.belief._base import TensorDataclassMixin


@dataclass
class CameraBelief(TensorDataclassMixin):
    """Batched camera state using ``T_world_from_camera`` transforms."""

    world_from_camera: Tensor
    linear_velocity: Tensor
    angular_velocity: Tensor
    intrinsics: Tensor
    log_variance: Tensor
    calibrated: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.world_from_camera.shape[0])

    def replace(self, **updates: Tensor) -> CameraBelief:
        return replace(self, **updates)

    def validate(
        self,
        *,
        log_variance_bounds: tuple[float, float] = (-30.0, 20.0),
    ) -> CameraBelief:
        from world_model.belief.validation import validate_camera_belief

        validate_camera_belief(self, log_variance_bounds=log_variance_bounds)
        return self
