"""Bounded persistent-ID temporal history for causal RGB motion evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from world_model.observations.base import ModalityHistory


@dataclass
class RGBTemporalPositionHistory(ModalityHistory):
    """Recent corrected posterior positions aligned by persistent object ID."""

    object_ids: Tensor
    timestamps: Tensor
    positions: Tensor
    position_log_variance: Tensor
    valid_mask: Tensor
    history_size: int

    @classmethod
    def empty(
        cls,
        *,
        object_ids: Tensor,
        active_mask: Tensor,
        history_size: int,
        dtype: torch.dtype,
    ) -> RGBTemporalPositionHistory:
        if history_size < 3:
            raise ValueError("RGB temporal history_size must be at least three")
        if object_ids.shape != active_mask.shape or active_mask.dtype != torch.bool:
            raise ValueError("object_ids and active_mask must be compatible [B,N] tensors")
        batch, objects = object_ids.shape
        ids = torch.where(active_mask, object_ids, torch.full_like(object_ids, -1))
        return cls(
            object_ids=ids,
            timestamps=torch.zeros(
                batch,
                objects,
                history_size,
                device=object_ids.device,
                dtype=dtype,
            ),
            positions=torch.zeros(
                batch,
                objects,
                history_size,
                3,
                device=object_ids.device,
                dtype=dtype,
            ),
            position_log_variance=torch.zeros(
                batch,
                objects,
                history_size,
                3,
                device=object_ids.device,
                dtype=dtype,
            ),
            valid_mask=torch.zeros(
                batch,
                objects,
                history_size,
                device=object_ids.device,
                dtype=torch.bool,
            ),
            history_size=history_size,
        )

    def _compatible(
        self,
        *,
        object_ids: Tensor,
        positions: Tensor,
        history_size: int,
    ) -> bool:
        return (
            self.history_size == history_size
            and self.object_ids.ndim == 2
            and self.object_ids.shape == object_ids.shape
            and self.timestamps.shape == (*object_ids.shape, history_size)
            and self.positions.shape == (*object_ids.shape, history_size, 3)
            and self.position_log_variance.shape == self.positions.shape
            and self.valid_mask.shape == self.timestamps.shape
            and self.object_ids.device == object_ids.device
            and self.positions.device == positions.device
            and self.positions.dtype == positions.dtype
        )

    def append(
        self,
        *,
        object_ids: Tensor,
        active_mask: Tensor,
        observed_mask: Tensor,
        timestamp: Tensor,
        positions: Tensor,
        position_log_variance: Tensor,
        minimum_dt: float,
    ) -> RGBTemporalPositionHistory:
        """Align by ID, append only fresh observed positions, and drop dead IDs."""

        if positions.shape != (*object_ids.shape, 3):
            raise ValueError("positions must have shape [B,N,3]")
        if position_log_variance.shape != positions.shape:
            raise ValueError("position_log_variance must match positions")
        if active_mask.shape != object_ids.shape or active_mask.dtype != torch.bool:
            raise ValueError("active_mask must be boolean [B,N]")
        if observed_mask.shape != object_ids.shape or observed_mask.dtype != torch.bool:
            raise ValueError("observed_mask must be boolean [B,N]")
        if timestamp.shape != object_ids.shape[:1]:
            raise ValueError("timestamp must have shape [B]")
        if not math.isfinite(minimum_dt) or minimum_dt <= 0:
            raise ValueError("minimum_dt must be finite and positive")

        source = self
        if not self._compatible(
            object_ids=object_ids,
            positions=positions,
            history_size=self.history_size,
        ):
            source = self.empty(
                object_ids=object_ids,
                active_mask=active_mask,
                history_size=self.history_size,
                dtype=positions.dtype,
            )
        current_ids = torch.where(
            active_mask & (object_ids >= 0),
            object_ids,
            torch.full_like(object_ids, -1),
        )
        aligned = self.empty(
            object_ids=current_ids,
            active_mask=current_ids >= 0,
            history_size=self.history_size,
            dtype=positions.dtype,
        )
        batch, objects = current_ids.shape
        for batch_index in range(batch):
            for slot in range(objects):
                object_id = current_ids[batch_index, slot]
                if object_id < 0:
                    continue
                previous_slots = torch.nonzero(
                    source.object_ids[batch_index] == object_id,
                    as_tuple=False,
                ).flatten()
                if previous_slots.numel() == 0:
                    continue
                previous_slot = int(previous_slots[0])
                aligned.timestamps[batch_index, slot] = source.timestamps[
                    batch_index, previous_slot
                ]
                aligned.positions[batch_index, slot] = source.positions[batch_index, previous_slot]
                aligned.position_log_variance[batch_index, slot] = source.position_log_variance[
                    batch_index, previous_slot
                ]
                aligned.valid_mask[batch_index, slot] = source.valid_mask[
                    batch_index, previous_slot
                ]

        append_mask = observed_mask & active_mask & (current_ids >= 0)
        finite = torch.isfinite(positions).all(dim=-1) & torch.isfinite(position_log_variance).all(
            dim=-1
        )
        append_mask = append_mask & finite & torch.isfinite(timestamp).unsqueeze(-1)
        for batch_index, slot in torch.nonzero(append_mask, as_tuple=False).tolist():
            count = int(aligned.valid_mask[batch_index, slot].sum())
            current_timestamp = timestamp[batch_index]
            if count:
                previous_timestamp = aligned.timestamps[batch_index, slot, count - 1]
                if current_timestamp <= previous_timestamp + minimum_dt:
                    continue
            if count >= self.history_size:
                aligned.timestamps[batch_index, slot, :-1] = aligned.timestamps[
                    batch_index, slot, 1:
                ].clone()
                aligned.positions[batch_index, slot, :-1] = aligned.positions[
                    batch_index, slot, 1:
                ].clone()
                aligned.position_log_variance[batch_index, slot, :-1] = (
                    aligned.position_log_variance[batch_index, slot, 1:].clone()
                )
                aligned.valid_mask[batch_index, slot, :-1] = aligned.valid_mask[
                    batch_index, slot, 1:
                ].clone()
                count = self.history_size - 1
            aligned.timestamps[batch_index, slot, count] = current_timestamp
            aligned.positions[batch_index, slot, count] = positions[batch_index, slot]
            aligned.position_log_variance[batch_index, slot, count] = position_log_variance[
                batch_index, slot
            ]
            aligned.valid_mask[batch_index, slot, count] = True
        return aligned

    def least_squares_velocity(
        self,
        *,
        minimum_dt: float,
        variance_scale: float,
        variance_floor: float,
        variance_ceiling: float | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return causal LS slope and propagated diagonal uncertainty."""

        if variance_scale < 1 or not math.isfinite(variance_scale):
            raise ValueError("variance_scale must be finite and at least one")
        if variance_floor <= 0 or not math.isfinite(variance_floor):
            raise ValueError("variance_floor must be finite and positive")
        if variance_ceiling is not None and (
            not math.isfinite(variance_ceiling) or variance_ceiling < variance_floor
        ):
            raise ValueError("variance_ceiling must be finite and no smaller than variance_floor")
        mask = self.valid_mask
        count = mask.sum(dim=-1)
        mask_float = mask.to(self.timestamps.dtype)
        mean_timestamp = (self.timestamps * mask_float).sum(dim=-1) / count.clamp_min(1)
        centred = (self.timestamps - mean_timestamp.unsqueeze(-1)) * mask_float
        denominator = centred.square().sum(dim=-1)
        weights = centred / denominator.clamp_min(minimum_dt * minimum_dt).unsqueeze(-1)
        velocity = (weights.unsqueeze(-1) * self.positions).sum(dim=-2)
        position_variance = self.position_log_variance.clamp(-30.0, 30.0).exp()
        velocity_variance = (weights.square().unsqueeze(-1) * position_variance).sum(
            dim=-2
        ) * variance_scale
        velocity_variance = velocity_variance.clamp_min(variance_floor)
        if variance_ceiling is not None:
            velocity_variance = velocity_variance.clamp_max(variance_ceiling)
        adjacent = mask[..., 1:] & mask[..., :-1]
        monotonic = (
            (~adjacent) | ((self.timestamps[..., 1:] - self.timestamps[..., :-1]) > minimum_dt)
        ).all(dim=-1)
        valid = (
            (count >= 3)
            & monotonic
            & (denominator > minimum_dt * minimum_dt)
            & torch.isfinite(velocity).all(dim=-1)
            & torch.isfinite(velocity_variance).all(dim=-1)
        )
        velocity = torch.where(valid.unsqueeze(-1), velocity, torch.zeros_like(velocity))
        default_log_variance = torch.full_like(velocity, math.log(variance_floor))
        log_variance = torch.where(
            valid.unsqueeze(-1),
            velocity_variance.log(),
            default_log_variance,
        )
        return velocity, log_variance, valid

    def detach(self) -> RGBTemporalPositionHistory:
        return RGBTemporalPositionHistory(
            object_ids=self.object_ids.detach(),
            timestamps=self.timestamps.detach(),
            positions=self.positions.detach(),
            position_log_variance=self.position_log_variance.detach(),
            valid_mask=self.valid_mask.detach(),
            history_size=self.history_size,
        )
