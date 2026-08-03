"""Arbitrary-time functional rollout engine."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from world_model.belief import BeliefTrajectory, WorldBelief


@dataclass
class RolloutStep:
    """One prediction segment.

    ``belief.objects.motion_mode_logits`` describes the state at the segment
    endpoint. The collision channel in ``event_logits`` records collision
    occurrence anywhere between the previous and current query timestamps.
    """

    belief: WorldBelief
    event_logits: Tensor
    auxiliary: dict[str, Tensor]


class RolloutEngine:
    """Sample a receding-horizon predictor without mutating its source belief."""

    def _normalise_query_times(
        self,
        belief: WorldBelief,
        query_times: Tensor | Sequence[float],
    ) -> Tensor:
        times = torch.as_tensor(
            query_times,
            device=belief.device,
            dtype=belief.dtype,
        )
        if times.ndim == 1:
            times = times.unsqueeze(0).expand(belief.batch_size, -1).clone()
        if times.ndim != 2 or times.shape[0] != belief.batch_size:
            raise ValueError("query_times must have shape [T] or [B,T]")
        if not torch.isfinite(times).all() or torch.any(times < 0):
            raise ValueError("query times must be finite nonnegative offsets")
        if times.shape[1] > 1 and torch.any(times[:, 1:] < times[:, :-1]):
            raise ValueError("query times must be sorted for every batch element")
        return times

    def rollout(
        self,
        predictor: Callable[[WorldBelief, Tensor], RolloutStep],
        belief: WorldBelief,
        query_times: Tensor | Sequence[float],
        *,
        return_events: bool = True,
        return_auxiliary: bool = True,
    ) -> BeliefTrajectory:
        """Roll forward while optionally retaining event and auxiliary traces."""

        offsets = self._normalise_query_times(belief, query_times)
        count = offsets.shape[1]
        if count == 0:
            objects = belief.objects
            trajectory = BeliefTrajectory(
                timestamps=belief.timestamp.new_empty(belief.batch_size, 0),
                positions=objects.position.new_empty(belief.batch_size, 0, objects.max_objects, 3),
                velocities=objects.velocity.new_empty(belief.batch_size, 0, objects.max_objects, 3),
                orientations=objects.orientation.new_empty(
                    belief.batch_size, 0, objects.max_objects, 4
                ),
                motion_mode_logits=objects.motion_mode_logits.new_empty(
                    belief.batch_size,
                    0,
                    objects.max_objects,
                    objects.motion_mode_logits.shape[-1],
                ),
                fast_log_variance=objects.fast_log_variance.new_empty(
                    belief.batch_size,
                    0,
                    objects.max_objects,
                    objects.fast_state_dim,
                ),
                active_mask=objects.active.new_empty(belief.batch_size, 0, objects.max_objects),
                event_logits=None,
            )
            return trajectory.validate()

        current = belief.clone()
        previous_offset = torch.zeros(
            belief.batch_size,
            device=belief.device,
            dtype=belief.dtype,
        )
        beliefs: list[WorldBelief] = []
        event_values: list[Tensor] = []
        auxiliary_values: dict[str, list[Tensor]] = {}
        for index in range(count):
            delta = offsets[:, index] - previous_offset
            step = predictor(current, delta)
            current = step.belief
            beliefs.append(current)
            if return_events:
                event_values.append(step.event_logits)
            if return_auxiliary:
                for name, value in step.auxiliary.items():
                    auxiliary_values.setdefault(name, []).append(value)
            previous_offset = offsets[:, index]

        trajectory = BeliefTrajectory(
            timestamps=torch.stack([item.timestamp for item in beliefs], dim=1),
            positions=torch.stack([item.objects.position for item in beliefs], dim=1),
            velocities=torch.stack([item.objects.velocity for item in beliefs], dim=1),
            orientations=torch.stack([item.objects.orientation for item in beliefs], dim=1),
            motion_mode_logits=torch.stack(
                [item.objects.motion_mode_logits for item in beliefs], dim=1
            ),
            fast_log_variance=torch.stack(
                [item.objects.fast_log_variance for item in beliefs], dim=1
            ),
            active_mask=torch.stack([item.objects.active for item in beliefs], dim=1),
            event_logits=(torch.stack(event_values, dim=1) if return_events else None),
            auxiliary=(
                {name: torch.stack(values, dim=1) for name, values in auxiliary_values.items()}
                if return_auxiliary
                else {}
            ),
        )
        return trajectory.validate()
