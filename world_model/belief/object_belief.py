"""Canonical object-centric belief tensor."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum

from torch import Tensor

from world_model.belief._base import TensorDataclassMixin


class MotionMode(IntEnum):
    """Canonical physical/observation modes stored in ``motion_mode_logits``."""

    FREE = 0
    GROUND_CONTACT = 1
    PAIR_CONTACT = 2
    COLLISION = 3
    ROLLING = 4
    SLIDING = 5
    SLEEPING = 6
    OCCLUDED = 7
    EXTERNALLY_ACTUATED = 8
    CREATED = 9
    REMOVED = 10


NUM_MOTION_MODES = len(MotionMode)


@dataclass
class ObjectBeliefTensor(TensorDataclassMixin):
    """Batched, padded persistent object beliefs.

    Tensor axes are ``[B, N, ...]``.  Inactive padding slots have
    ``object_id == -1``.  Quaternions are scalar-last ``[x, y, z, w]``.
    The first geometry component is the sphere radius in the toy world; later
    components remain a modality-independent geometry code.
    """

    object_id: Tensor
    active: Tensor
    existence_logit: Tensor

    position: Tensor
    velocity: Tensor
    orientation: Tensor
    angular_velocity: Tensor

    geometry: Tensor
    appearance: Tensor
    residual_dynamics: Tensor

    modal_state: Tensor
    modal_frequency: Tensor
    modal_decay_raw: Tensor

    log_mass: Tensor
    restitution_logit: Tensor
    log_drag: Tensor
    friction_logit: Tensor

    motion_mode_logits: Tensor
    visibility_logit: Tensor
    age_steps: Tensor
    missed_steps: Tensor

    fast_log_variance: Tensor
    slow_log_variance: Tensor

    parameter_memory: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.object_id.shape[0])

    @property
    def max_objects(self) -> int:
        return int(self.object_id.shape[1])

    @property
    def geometry_dim(self) -> int:
        return int(self.geometry.shape[-1])

    @property
    def appearance_dim(self) -> int:
        return int(self.appearance.shape[-1])

    @property
    def residual_dynamics_dim(self) -> int:
        return int(self.residual_dynamics.shape[-1])

    @property
    def modal_count(self) -> int:
        return int(self.modal_state.shape[-3])

    @property
    def modal_dim(self) -> int:
        return int(self.modal_state.shape[-1])

    @property
    def fast_state_dim(self) -> int:
        # position, velocity, scalar-last quaternion, angular velocity, modes
        return 13 + self.modal_count * 2 * self.modal_dim

    @property
    def slow_state_dim(self) -> int:
        # Four physical scalars plus geometry/appearance/dynamics codes.
        return 4 + self.geometry_dim + self.appearance_dim + self.residual_dynamics_dim

    @property
    def radius(self) -> Tensor:
        """Positive toy-sphere radius ``[B,N,1]`` from the geometry code."""

        if self.geometry_dim < 1:
            raise ValueError("geometry must contain radius in component zero")
        return self.geometry[..., :1].clamp_min(1e-6)

    @property
    def mass(self) -> Tensor:
        return self.log_mass.clamp(-12.0, 12.0).exp()

    @property
    def restitution(self) -> Tensor:
        return self.restitution_logit.sigmoid()

    @property
    def drag(self) -> Tensor:
        return self.log_drag.clamp(-16.0, 8.0).exp()

    @property
    def friction(self) -> Tensor:
        return self.friction_logit.sigmoid()

    @property
    def mode(self) -> Tensor:
        return self.motion_mode_logits.argmax(dim=-1)

    def replace(self, **updates: Tensor) -> ObjectBeliefTensor:
        """Return a shallow structural copy with selected tensor fields replaced."""

        return replace(self, **updates)

    def validate(
        self,
        *,
        log_variance_bounds: tuple[float, float] = (-30.0, 20.0),
        quaternion_tolerance: float = 1e-4,
    ) -> ObjectBeliefTensor:
        from world_model.belief.validation import validate_object_belief

        validate_object_belief(
            self,
            log_variance_bounds=log_variance_bounds,
            quaternion_tolerance=quaternion_tolerance,
        )
        return self
