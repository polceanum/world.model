"""Bounded raw metric-position history for the one-slot RGB-D bridge."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from world_model.dynamics import FreeMotionFitResult, fit_free_motion
from world_model.observations.base import ModalityHistory


@dataclass
class RGBDTemporalPositionHistory(ModalityHistory):
    """Fixed-size raw RGB-D positions aligned by persistent object ID.

    ``sample_mask`` records observation times even when metric depth is
    unusable.  ``valid_mask`` is deliberately separate: an invalid depth row
    occupies its causal place in the window and makes the complete uniform
    fit fail closed instead of silently selecting a shorter history.
    """

    object_ids: Tensor
    timestamps: Tensor
    positions: Tensor
    sample_mask: Tensor
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
    ) -> RGBDTemporalPositionHistory:
        if history_size < 2:
            raise ValueError("RGB-D temporal history_size must be at least two")
        if object_ids.ndim != 2 or object_ids.dtype != torch.int64:
            raise TypeError("object_ids must be int64 [B,N]")
        if active_mask.shape != object_ids.shape or active_mask.dtype != torch.bool:
            raise ValueError("active_mask must be boolean and match object_ids")
        if dtype not in {torch.float32, torch.float64}:
            raise TypeError("RGB-D temporal history supports only float32 and float64")
        batch, objects = object_ids.shape
        return cls(
            object_ids=torch.where(active_mask, object_ids, torch.full_like(object_ids, -1)),
            timestamps=torch.zeros(
                batch,
                objects,
                history_size,
                dtype=dtype,
                device=object_ids.device,
            ),
            positions=torch.zeros(
                batch,
                objects,
                history_size,
                3,
                dtype=dtype,
                device=object_ids.device,
            ),
            sample_mask=torch.zeros(
                batch,
                objects,
                history_size,
                dtype=torch.bool,
                device=object_ids.device,
            ),
            valid_mask=torch.zeros(
                batch,
                objects,
                history_size,
                dtype=torch.bool,
                device=object_ids.device,
            ),
            history_size=history_size,
        )

    def _validate_storage(self) -> None:
        if self.object_ids.ndim != 2 or self.object_ids.dtype != torch.int64:
            raise TypeError("history object_ids must be int64 [B,N]")
        shape = (*self.object_ids.shape, self.history_size)
        if self.timestamps.shape != shape:
            raise ValueError("history timestamps have an incompatible shape")
        if self.positions.shape != (*shape, 3):
            raise ValueError("history positions have an incompatible shape")
        for name, value in (
            ("sample_mask", self.sample_mask),
            ("valid_mask", self.valid_mask),
        ):
            if value.shape != shape or value.dtype != torch.bool:
                raise ValueError(f"history {name} must be boolean [B,N,H]")
        if self.positions.dtype not in {torch.float32, torch.float64}:
            raise TypeError("history positions must use float32 or float64")
        if self.timestamps.dtype != self.positions.dtype:
            raise TypeError("history timestamps and positions must share dtype")
        if any(
            value.device != self.positions.device
            for value in (
                self.object_ids,
                self.timestamps,
                self.sample_mask,
                self.valid_mask,
            )
        ):
            raise ValueError("history tensors must share one device")
        if not torch.isfinite(self.timestamps).all() or not torch.isfinite(self.positions).all():
            raise ValueError("history tensors must be finite")
        if torch.any(self.valid_mask & ~self.sample_mask):
            raise ValueError("valid history rows must also be sampled")

    def _aligned(
        self,
        *,
        object_ids: Tensor,
        active_mask: Tensor,
        dtype: torch.dtype,
    ) -> RGBDTemporalPositionHistory:
        """Return history gathered into current persistent-slot order."""

        self._validate_storage()
        if object_ids.shape != self.object_ids.shape or object_ids.dtype != torch.int64:
            raise ValueError("current object_ids must match history batch/slot shape")
        if active_mask.shape != object_ids.shape or active_mask.dtype != torch.bool:
            raise ValueError("current active_mask must be boolean [B,N]")
        if object_ids.device != self.object_ids.device:
            raise ValueError("current object_ids and history must share device")
        if dtype != self.positions.dtype:
            raise TypeError("current belief and RGB-D history must share dtype")

        same_identity = (
            (object_ids[:, :, None] == self.object_ids[:, None, :])
            & active_mask[:, :, None]
            & (object_ids[:, :, None] >= 0)
        )
        match_count = same_identity.sum(dim=-1)
        if torch.any(match_count > 1):
            raise ValueError("RGB-D history contains duplicate persistent object IDs")
        retained = match_count == 1
        source_slot = same_identity.to(torch.int64).argmax(dim=-1)
        batch_index = torch.arange(
            object_ids.shape[0],
            dtype=torch.int64,
            device=object_ids.device,
        )[:, None]
        gathered_timestamps = self.timestamps[batch_index, source_slot]
        gathered_positions = self.positions[batch_index, source_slot]
        gathered_samples = self.sample_mask[batch_index, source_slot]
        gathered_valid = self.valid_mask[batch_index, source_slot]
        return RGBDTemporalPositionHistory(
            object_ids=torch.where(active_mask, object_ids, torch.full_like(object_ids, -1)),
            timestamps=torch.where(
                retained[..., None],
                gathered_timestamps,
                torch.zeros_like(gathered_timestamps),
            ),
            positions=torch.where(
                retained[..., None, None],
                gathered_positions,
                torch.zeros_like(gathered_positions),
            ),
            sample_mask=retained[..., None] & gathered_samples,
            valid_mask=retained[..., None] & gathered_valid,
            history_size=self.history_size,
        )

    def append(
        self,
        *,
        object_ids: Tensor,
        active_mask: Tensor,
        append_mask: Tensor,
        timestamp: Tensor,
        positions: Tensor,
        valid_mask: Tensor,
        minimum_dt: float,
    ) -> RGBDTemporalPositionHistory:
        """Append one causal row for selected current persistent slots."""

        aligned = self._aligned(
            object_ids=object_ids,
            active_mask=active_mask,
            dtype=positions.dtype,
        )
        expected = object_ids.shape
        if append_mask.shape != expected or append_mask.dtype != torch.bool:
            raise ValueError("append_mask must be boolean [B,N]")
        if valid_mask.shape != expected or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be boolean [B,N]")
        if torch.any(append_mask & ~active_mask):
            raise ValueError("only active slots may be appended")
        if torch.any(valid_mask & ~append_mask):
            raise ValueError("valid rows must be a subset of appended rows")
        if positions.shape != (*expected, 3):
            raise ValueError("positions must have shape [B,N,3]")
        if (
            positions.dtype != aligned.positions.dtype
            or positions.device != aligned.positions.device
        ):
            raise ValueError("positions must match history dtype and device")
        if timestamp.shape != (expected[0],):
            raise ValueError("timestamp must have shape [B]")
        if timestamp.dtype != positions.dtype or timestamp.device != positions.device:
            raise ValueError("timestamp must match position dtype and device")
        if not torch.isfinite(timestamp).all() or not torch.isfinite(positions).all():
            raise ValueError("appended timestamp and positions must be finite")
        if not torch.isfinite(torch.as_tensor(minimum_dt)) or minimum_dt <= 0.0:
            raise ValueError("minimum_dt must be finite and positive")

        has_previous = aligned.sample_mask.any(dim=-1)
        sample_index = torch.arange(
            self.history_size,
            dtype=torch.int64,
            device=aligned.positions.device,
        )
        latest_index = (
            torch.where(
                aligned.sample_mask,
                sample_index,
                torch.full_like(sample_index, -1),
            )
            .amax(dim=-1)
            .clamp_min(0)
        )
        latest_timestamp = torch.gather(
            aligned.timestamps,
            dim=-1,
            index=latest_index.unsqueeze(-1),
        ).squeeze(-1)
        stale = append_mask & has_previous & (timestamp[:, None] - latest_timestamp < minimum_dt)
        if torch.any(stale):
            raise ValueError("RGB-D temporal timestamps must increase by minimum_dt")

        shifted_timestamps = torch.cat(
            (
                aligned.timestamps[..., 1:],
                timestamp[:, None, None].expand(-1, expected[1], 1),
            ),
            dim=-1,
        )
        shifted_positions = torch.cat(
            (aligned.positions[..., 1:, :], positions.unsqueeze(-2)),
            dim=-2,
        )
        shifted_samples = torch.cat(
            (
                aligned.sample_mask[..., 1:],
                append_mask.unsqueeze(-1),
            ),
            dim=-1,
        )
        shifted_valid = torch.cat(
            (
                aligned.valid_mask[..., 1:],
                valid_mask.unsqueeze(-1),
            ),
            dim=-1,
        )
        return RGBDTemporalPositionHistory(
            object_ids=aligned.object_ids,
            timestamps=torch.where(
                append_mask[..., None],
                shifted_timestamps,
                aligned.timestamps,
            ),
            positions=torch.where(
                append_mask[..., None, None],
                shifted_positions,
                aligned.positions,
            ),
            sample_mask=torch.where(
                append_mask[..., None],
                shifted_samples,
                aligned.sample_mask,
            ),
            valid_mask=torch.where(
                append_mask[..., None],
                shifted_valid,
                aligned.valid_mask,
            ),
            history_size=aligned.history_size,
        )

    def validate_next_timestamp(
        self,
        *,
        object_ids: Tensor,
        active_mask: Tensor,
        timestamp: Tensor,
        minimum_dt: float,
    ) -> None:
        """Reject a stale append without changing history or runtime state."""

        aligned = self._aligned(
            object_ids=object_ids,
            active_mask=active_mask,
            dtype=timestamp.dtype,
        )
        if timestamp.shape != (object_ids.shape[0],):
            raise ValueError("timestamp must have shape [B]")
        if timestamp.device != aligned.timestamps.device:
            raise ValueError("timestamp and history must share device")
        if not torch.isfinite(timestamp).all():
            raise ValueError("timestamp must be finite")
        if not torch.isfinite(torch.as_tensor(minimum_dt)) or minimum_dt <= 0.0:
            raise ValueError("minimum_dt must be finite and positive")
        has_previous = aligned.sample_mask.any(dim=-1)
        sample_index = torch.arange(
            self.history_size,
            dtype=torch.int64,
            device=aligned.timestamps.device,
        )
        latest_index = (
            torch.where(
                aligned.sample_mask,
                sample_index,
                torch.full_like(sample_index, -1),
            )
            .amax(dim=-1)
            .clamp_min(0)
        )
        latest_timestamp = torch.gather(
            aligned.timestamps,
            dim=-1,
            index=latest_index.unsqueeze(-1),
        ).squeeze(-1)
        stale = active_mask & has_previous & (timestamp[:, None] - latest_timestamp < minimum_dt)
        if torch.any(stale):
            raise ValueError("RGB-D temporal timestamps must increase by minimum_dt")

    def fit(
        self,
        *,
        gravity: Tensor,
        drag: Tensor,
        minimum_support: int,
        minimum_dt: float,
        conditioning_limit: float,
        max_missing_rows: int = 0,
        require_latest_valid: bool = True,
    ) -> tuple[FreeMotionFitResult, Tensor]:
        """Fit the complete timestamp window and return a fail-closed mask.

        The accepted bridge keeps ``max_missing_rows=0`` and therefore takes
        the historical all-row uniform-fit path exactly.  The bounded recovery
        opt-in admits at most one invalid observation row, excludes that row
        from the otherwise uniform fit, and never emits a fit unless the newest
        row is a fresh valid observation.
        """

        self._validate_storage()
        if not 2 <= minimum_support <= self.history_size:
            raise ValueError("minimum_support must lie within the RGB-D history")
        if not torch.isfinite(torch.as_tensor(minimum_dt)) or minimum_dt <= 0.0:
            raise ValueError("minimum_dt must be finite and positive")
        if (
            isinstance(max_missing_rows, bool)
            or not isinstance(max_missing_rows, int)
            or max_missing_rows not in {0, 1}
        ):
            raise ValueError("RGB-D max_missing_rows must be integer zero or one")
        if require_latest_valid is not True:
            raise ValueError("RGB-D temporal fits require the latest row to be valid")

        fit_arguments = {
            "gravity": gravity,
            "drag": drag,
            "anchor_time": self.timestamps[..., -1],
            "conditioning_limit": conditioning_limit,
        }
        if max_missing_rows == 0:
            # Preserve the accepted all-16 implementation, including its
            # fail-closed diagnostics for malformed/invalid windows.
            fit = fit_free_motion(
                self.positions.permute(0, 2, 1, 3),
                self.timestamps.permute(0, 2, 1),
                minimum_support=self.history_size,
                **fit_arguments,
            )
        else:
            fit = fit_free_motion(
                self.positions.permute(0, 2, 1, 3),
                self.timestamps.permute(0, 2, 1),
                support=self.valid_mask.permute(0, 2, 1),
                minimum_support=minimum_support - max_missing_rows,
                **fit_arguments,
            )
        span = self.timestamps[..., -1] - self.timestamps[..., 0]
        complete_timestamp_window = self.sample_mask.sum(dim=-1).ge(minimum_support)
        complete_timestamp_window &= self.sample_mask.all(dim=-1)
        if max_missing_rows == 0:
            observation_rows_valid = self.valid_mask.all(dim=-1)
        else:
            missing_rows = (self.sample_mask & ~self.valid_mask).sum(dim=-1)
            observation_rows_valid = (
                self.valid_mask.sum(dim=-1).ge(minimum_support - max_missing_rows)
                & (missing_rows <= max_missing_rows)
                & self.valid_mask[..., -1]
            )
        sequence_valid = (
            complete_timestamp_window
            & observation_rows_valid
            & (span >= minimum_dt)
            & fit.valid
            & (self.object_ids >= 0)
        )
        return fit, sequence_valid

    def detach(self) -> RGBDTemporalPositionHistory:
        return RGBDTemporalPositionHistory(
            object_ids=self.object_ids.detach(),
            timestamps=self.timestamps.detach(),
            positions=self.positions.detach(),
            sample_mask=self.sample_mask.detach(),
            valid_mask=self.valid_mask.detach(),
            history_size=self.history_size,
        )


__all__ = ["RGBDTemporalPositionHistory"]
