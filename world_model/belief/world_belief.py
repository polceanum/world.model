"""World-level persistent belief and trajectory contracts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import torch
from torch import Tensor

from world_model.belief._base import TensorDataclassMixin
from world_model.belief.camera_belief import CameraBelief
from world_model.belief.object_belief import (
    NUM_MOTION_MODES,
    MotionMode,
    ObjectBeliefTensor,
)


@dataclass
class WorldBelief(TensorDataclassMixin):
    """Persistent modality-independent source of truth for online inference."""

    timestamp: Tensor
    objects: ObjectBeliefTensor
    camera: CameraBelief
    gravity: Tensor
    global_code: Tensor
    global_log_variance: Tensor
    next_object_id: Tensor
    active_modalities: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        return int(self.timestamp.shape[0])

    @property
    def device(self) -> torch.device:
        return self.timestamp.device

    @property
    def dtype(self) -> torch.dtype:
        return self.timestamp.dtype

    def replace(self, **updates: Any) -> WorldBelief:
        return replace(self, **updates)

    def with_timestamp(self, timestamp: float | Tensor) -> WorldBelief:
        """Return a copy at ``timestamp`` and reject time reversal."""

        value = torch.as_tensor(
            timestamp,
            device=self.timestamp.device,
            dtype=self.timestamp.dtype,
        )
        if value.ndim == 0:
            value = value.expand_as(self.timestamp).clone()
        if value.shape != self.timestamp.shape:
            raise ValueError(
                f"timestamp must be scalar or {tuple(self.timestamp.shape)}, "
                f"got {tuple(value.shape)}"
            )
        if not torch.isfinite(value).all():
            raise ValueError("timestamp must be finite")
        if torch.any(value < self.timestamp):
            raise ValueError("WorldBelief timestamp cannot move backward")
        return replace(self, timestamp=value)

    def validate(
        self,
        *,
        log_variance_bounds: tuple[float, float] = (-30.0, 20.0),
        quaternion_tolerance: float = 1e-4,
    ) -> WorldBelief:
        from world_model.belief.validation import validate_world_belief

        validate_world_belief(
            self,
            log_variance_bounds=log_variance_bounds,
            quaternion_tolerance=quaternion_tolerance,
        )
        return self


@dataclass
class BeliefTrajectory(TensorDataclassMixin):
    """Sampled future belief means and uncertainties at arbitrary times."""

    timestamps: Tensor
    positions: Tensor
    velocities: Tensor
    orientations: Tensor
    motion_mode_logits: Tensor
    fast_log_variance: Tensor
    active_mask: Tensor
    event_logits: Tensor | None = None
    auxiliary: dict[str, Tensor] = field(default_factory=dict)

    def validate(self) -> BeliefTrajectory:
        if self.timestamps.ndim != 2:
            raise ValueError("trajectory timestamps must have shape [B,T]")
        batch, steps = self.timestamps.shape
        if self.positions.ndim != 4 or self.positions.shape[:2] != (batch, steps):
            raise ValueError("trajectory positions must have shape [B,T,N,3]")
        if self.positions.shape[-1] != 3:
            raise ValueError("trajectory positions must have final dimension 3")
        object_shape = self.positions.shape[:3]
        if self.velocities.shape != (*object_shape, 3):
            raise ValueError("trajectory velocities must have shape [B,T,N,3]")
        if self.orientations.shape != (*object_shape, 4):
            raise ValueError("trajectory orientations must have shape [B,T,N,4]")
        if self.motion_mode_logits.shape[:3] != object_shape:
            raise ValueError("motion mode logits must begin with [B,T,N]")
        if self.fast_log_variance.shape[:3] != object_shape:
            raise ValueError("fast log variance must begin with [B,T,N]")
        if self.active_mask.shape != object_shape:
            raise ValueError("active mask must have shape [B,T,N]")
        if self.active_mask.dtype is not torch.bool:
            raise TypeError("active mask must be torch.bool")
        if self.event_logits is not None and self.event_logits.shape[:3] != object_shape:
            raise ValueError("trajectory event logits must begin with [B,T,N]")
        if steps > 1 and torch.any(self.timestamps[:, 1:] < self.timestamps[:, :-1]):
            raise ValueError("trajectory timestamps must be sorted")
        for name, value in (
            ("timestamps", self.timestamps),
            ("positions", self.positions),
            ("velocities", self.velocities),
            ("orientations", self.orientations),
            ("motion_mode_logits", self.motion_mode_logits),
            ("fast_log_variance", self.fast_log_variance),
        ):
            if not torch.isfinite(value).all():
                raise ValueError(f"trajectory {name} contains NaN or Inf")
        if self.event_logits is not None and not torch.isfinite(self.event_logits).all():
            raise ValueError("trajectory event_logits contains NaN or Inf")
        for name, value in self.auxiliary.items():
            if not isinstance(value, Tensor):
                raise TypeError(f"trajectory auxiliary {name} must be a tensor")
            if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(
                value
            ).all():
                raise ValueError(f"trajectory auxiliary {name} contains NaN or Inf")
        return self


@dataclass(frozen=True)
class BeliefFactory:
    """Create empty, correctly typed beliefs without simulator-state inputs."""

    max_objects: int
    geometry_dim: int = 1
    appearance_dim: int = 16
    residual_dynamics_dim: int = 8
    modal_count: int = 2
    modal_dim: int = 2
    parameter_memory_dim: int = 32
    global_code_dim: int = 8
    camera_variance_dim: int = 12
    num_motion_modes: int = NUM_MOTION_MODES
    initial_radius: float = 0.1
    initial_mass: float = 1.0
    initial_restitution: float = 0.7
    initial_drag: float = 0.05
    initial_friction: float = 0.2
    initial_log_variance: float = 0.0

    @classmethod
    def from_config(cls, config: Any) -> BeliefFactory:
        """Build from ``OrpheusConfig`` or its ``ModelConfig`` subsection.

        This uses structural attributes rather than importing the root config
        module, keeping the low-level belief contract independently reusable.
        """

        model = getattr(config, "model", config)
        state = getattr(model, "state", model)
        if not hasattr(model, "max_objects"):
            raise TypeError("config must expose model.max_objects")
        return cls(
            max_objects=int(model.max_objects),
            geometry_dim=int(state.geometry_dim),
            appearance_dim=int(state.appearance_dim),
            residual_dynamics_dim=int(state.residual_dynamics_dim),
            modal_count=int(state.modal_count),
            modal_dim=int(state.modal_dim),
            parameter_memory_dim=int(state.parameter_memory_dim),
            global_code_dim=int(state.global_dim),
            initial_log_variance=float(state.fast_log_variance_max),
        )

    def __post_init__(self) -> None:
        integer_dims = (
            self.max_objects,
            self.geometry_dim,
            self.appearance_dim,
            self.residual_dynamics_dim,
            self.modal_count,
            self.modal_dim,
            self.parameter_memory_dim,
            self.global_code_dim,
            self.camera_variance_dim,
            self.num_motion_modes,
        )
        if self.max_objects <= 0 or any(value < 0 for value in integer_dims[1:]):
            raise ValueError("belief dimensions must be nonnegative and max_objects > 0")
        if self.geometry_dim < 1:
            raise ValueError("geometry_dim must include at least the sphere radius")
        if self.num_motion_modes < NUM_MOTION_MODES:
            raise ValueError(f"num_motion_modes must be at least {NUM_MOTION_MODES}")
        if self.initial_radius <= 0 or self.initial_mass <= 0:
            raise ValueError("initial radius and mass must be positive")
        if self.initial_drag <= 0:
            raise ValueError("initial drag must be positive")
        for name, value in (
            ("initial_restitution", self.initial_restitution),
            ("initial_friction", self.initial_friction),
        ):
            if not 0 < value < 1:
                raise ValueError(f"{name} must lie strictly between zero and one")

    @property
    def fast_state_dim(self) -> int:
        return 13 + self.modal_count * 2 * self.modal_dim

    @property
    def slow_state_dim(self) -> int:
        return 4 + self.geometry_dim + self.appearance_dim + self.residual_dynamics_dim

    def create(
        self,
        batch_size: int = 1,
        *,
        timestamp: float | Tensor = 0.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        gravity: Sequence[float] = (0.0, -9.81, 0.0),
        intrinsics: Tensor | None = None,
        world_from_camera: Tensor | None = None,
        active_modalities: tuple[str, ...] = (),
    ) -> WorldBelief:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        shape = (batch_size, self.max_objects)
        zeros = lambda *tail: torch.zeros(  # noqa: E731
            *shape, *tail, device=device, dtype=dtype
        )
        object_id = torch.full(shape, -1, device=device, dtype=torch.int64)
        active = torch.zeros(shape, device=device, dtype=torch.bool)
        orientation = zeros(4)
        orientation[..., 3] = 1.0
        geometry = zeros(self.geometry_dim)
        geometry[..., 0] = self.initial_radius
        motion_logits = zeros(self.num_motion_modes)
        motion_logits[..., MotionMode.FREE] = 1.0
        restitution_logit = torch.full(
            (*shape, 1),
            torch.logit(torch.tensor(self.initial_restitution)).item(),
            device=device,
            dtype=dtype,
        )
        friction_logit = torch.full(
            (*shape, 1),
            torch.logit(torch.tensor(self.initial_friction)).item(),
            device=device,
            dtype=dtype,
        )
        objects = ObjectBeliefTensor(
            object_id=object_id,
            active=active,
            existence_logit=zeros(),
            position=zeros(3),
            velocity=zeros(3),
            orientation=orientation,
            angular_velocity=zeros(3),
            geometry=geometry,
            appearance=zeros(self.appearance_dim),
            residual_dynamics=zeros(self.residual_dynamics_dim),
            modal_state=zeros(self.modal_count, 2, self.modal_dim),
            modal_frequency=zeros(self.modal_count, self.modal_dim),
            modal_decay_raw=zeros(self.modal_count, self.modal_dim),
            log_mass=torch.full(
                (*shape, 1),
                float(torch.log(torch.tensor(self.initial_mass))),
                device=device,
                dtype=dtype,
            ),
            restitution_logit=restitution_logit,
            log_drag=torch.full(
                (*shape, 1),
                float(torch.log(torch.tensor(self.initial_drag))),
                device=device,
                dtype=dtype,
            ),
            friction_logit=friction_logit,
            motion_mode_logits=motion_logits,
            visibility_logit=zeros(),
            age_steps=torch.zeros(shape, device=device, dtype=torch.int64),
            missed_steps=torch.zeros(shape, device=device, dtype=torch.int64),
            fast_log_variance=torch.full(
                (*shape, self.fast_state_dim),
                self.initial_log_variance,
                device=device,
                dtype=dtype,
            ),
            slow_log_variance=torch.full(
                (*shape, self.slow_state_dim),
                self.initial_log_variance,
                device=device,
                dtype=dtype,
            ),
            parameter_memory=zeros(self.parameter_memory_dim),
        )
        identity = torch.eye(4, device=device, dtype=dtype).expand(batch_size, -1, -1).clone()
        if world_from_camera is not None:
            identity = world_from_camera.to(device=device, dtype=dtype)
        camera_intrinsics = (
            torch.eye(3, device=device, dtype=dtype).expand(batch_size, -1, -1).clone()
        )
        if intrinsics is not None:
            camera_intrinsics = intrinsics.to(device=device, dtype=dtype)
        camera = CameraBelief(
            world_from_camera=identity,
            linear_velocity=torch.zeros(batch_size, 3, device=device, dtype=dtype),
            angular_velocity=torch.zeros(batch_size, 3, device=device, dtype=dtype),
            intrinsics=camera_intrinsics,
            log_variance=torch.full(
                (batch_size, self.camera_variance_dim),
                self.initial_log_variance,
                device=device,
                dtype=dtype,
            ),
            calibrated=torch.ones(batch_size, device=device, dtype=torch.bool),
        )
        belief_timestamp = torch.as_tensor(timestamp, device=device, dtype=dtype)
        if belief_timestamp.ndim == 0:
            belief_timestamp = belief_timestamp.expand(batch_size).clone()
        world = WorldBelief(
            timestamp=belief_timestamp,
            objects=objects,
            camera=camera,
            gravity=torch.as_tensor(gravity, device=device, dtype=dtype)
            .expand(batch_size, -1)
            .clone(),
            global_code=torch.zeros(batch_size, self.global_code_dim, device=device, dtype=dtype),
            global_log_variance=torch.full(
                (batch_size, self.global_code_dim),
                self.initial_log_variance,
                device=device,
                dtype=dtype,
            ),
            next_object_id=torch.zeros(batch_size, device=device, dtype=torch.int64),
            active_modalities=active_modalities,
            metadata={"initialised": False},
        )
        return world.validate()

    empty = create
