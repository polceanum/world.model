"""Sequence-aware identity and uncertainty metrics through full occlusion."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from torch import Tensor


@dataclass
class _OcclusionTrack:
    """Evaluator-only state for one ground-truth target trajectory."""

    prediction_id: int
    pre_position_std_m: float
    in_occlusion: bool = False
    peak_occluded_position_std_m: float | None = None
    prediction_present_throughout: bool = True


@dataclass
class OcclusionTransitionAccumulator:
    """Measure visible -> fully occluded -> visible belief behaviour.

    A distance-gated association on a reliably visible frame establishes the
    target-to-prediction identity. During an occlusion transition, the
    accumulator follows that persistent prediction ID directly. It deliberately
    does not rematch by position or require the prediction to remain within a
    localization threshold while the target is fully hidden.
    """

    visible_fraction_threshold: float = 0.5
    fully_occluded_fraction_threshold: float = 0.05
    _tracks: dict[tuple[int, int], _OcclusionTrack] = field(default_factory=dict)
    _qualifying_sequences: int = 0
    _identity_survivals: int = 0
    _pre_position_std_m: list[float] = field(default_factory=list)
    _peak_occluded_position_std_m: list[float] = field(default_factory=list)
    _position_std_growth_m: list[float] = field(default_factory=list)
    _reobservation_position_std_m: list[float] = field(default_factory=list)
    _reobservation_contraction_m: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.fully_occluded_fraction_threshold < 0.0:
            raise ValueError("fully occluded visibility threshold must be nonnegative")
        if self.visible_fraction_threshold > 1.0:
            raise ValueError("visible fraction threshold must be at most one")
        if self.fully_occluded_fraction_threshold >= self.visible_fraction_threshold:
            raise ValueError("fully occluded threshold must be below the visible threshold")

    def update_frame(
        self,
        *,
        predicted_ids: Tensor,
        predicted_active: Tensor,
        position_std_m: Tensor,
        target_ids: Tensor,
        target_active: Tensor,
        target_visible_fraction: Tensor,
        matched_target_indices: Tensor,
        reliable_visible_matches: Tensor,
        episode_offset: int,
    ) -> None:
        """Consume one online frame.

        ``matched_target_indices`` maps each prediction slot to a target slot.
        ``reliable_visible_matches`` should be the evaluator's distance-gated
        association mask. It is used only to establish or confirm alignment on
        visible frames, never to follow a prediction during occlusion.
        """

        self._validate_shapes(
            predicted_ids=predicted_ids,
            predicted_active=predicted_active,
            position_std_m=position_std_m,
            target_ids=target_ids,
            target_active=target_active,
            target_visible_fraction=target_visible_fraction,
            matched_target_indices=matched_target_indices,
            reliable_visible_matches=reliable_visible_matches,
        )
        if episode_offset < 0:
            raise ValueError("episode_offset must be nonnegative")

        predicted_ids_cpu = predicted_ids.detach().cpu()
        predicted_active_cpu = predicted_active.detach().bool().cpu()
        position_std_cpu = position_std_m.detach().float().cpu()
        target_ids_cpu = target_ids.detach().cpu()
        target_active_cpu = target_active.detach().bool().cpu()
        visible_fraction_cpu = target_visible_fraction.detach().float().cpu()
        target_indices_cpu = matched_target_indices.detach().cpu()
        reliable_matches_cpu = reliable_visible_matches.detach().bool().cpu()

        for batch_index in range(predicted_ids.shape[0]):
            episode_id = episode_offset + batch_index
            prediction_slot_by_id = {
                int(predicted_ids_cpu[batch_index, slot]): slot
                for slot in range(predicted_ids.shape[1])
                if bool(predicted_active_cpu[batch_index, slot])
                and int(predicted_ids_cpu[batch_index, slot]) >= 0
            }
            reliable_slot_by_target = {
                int(target_indices_cpu[batch_index, slot]): slot
                for slot in range(predicted_ids.shape[1])
                if bool(reliable_matches_cpu[batch_index, slot])
                and bool(predicted_active_cpu[batch_index, slot])
                and int(target_indices_cpu[batch_index, slot]) >= 0
            }
            active_target_ids: set[int] = set()

            for target_slot in range(target_ids.shape[1]):
                if not bool(target_active_cpu[batch_index, target_slot]):
                    continue
                target_id = int(target_ids_cpu[batch_index, target_slot])
                if target_id < 0:
                    continue
                active_target_ids.add(target_id)
                key = (episode_id, target_id)
                visible_fraction = float(visible_fraction_cpu[batch_index, target_slot])
                reliably_visible = visible_fraction >= self.visible_fraction_threshold
                fully_occluded = visible_fraction <= self.fully_occluded_fraction_threshold
                reliable_slot = (
                    reliable_slot_by_target.get(target_slot) if reliably_visible else None
                )
                track = self._tracks.get(key)

                if track is not None and track.in_occlusion:
                    if reliably_visible:
                        self._finish_occlusion(
                            track=track,
                            reliable_slot=reliable_slot,
                            predicted_ids=predicted_ids_cpu[batch_index],
                            position_std_m=position_std_cpu[batch_index],
                        )
                        self._replace_visible_anchor(
                            key=key,
                            reliable_slot=reliable_slot,
                            predicted_ids=predicted_ids_cpu[batch_index],
                            position_std_m=position_std_cpu[batch_index],
                        )
                    else:
                        self._follow_occluded_prediction(
                            track=track,
                            prediction_slot_by_id=prediction_slot_by_id,
                            position_std_m=position_std_cpu[batch_index],
                            sample_uncertainty=fully_occluded,
                        )
                    continue

                if fully_occluded:
                    if track is not None:
                        track.in_occlusion = True
                        self._follow_occluded_prediction(
                            track=track,
                            prediction_slot_by_id=prediction_slot_by_id,
                            position_std_m=position_std_cpu[batch_index],
                            sample_uncertainty=True,
                        )
                    continue

                if reliably_visible:
                    self._replace_visible_anchor(
                        key=key,
                        reliable_slot=reliable_slot,
                        predicted_ids=predicted_ids_cpu[batch_index],
                        position_std_m=position_std_cpu[batch_index],
                    )

            for key in [
                key
                for key in self._tracks
                if key[0] == episode_id and key[1] not in active_target_ids
            ]:
                del self._tracks[key]

    def metrics(self) -> dict[str, float | None]:
        """Return paired transition metrics, using null for unavailable values."""

        sequence_count = self._qualifying_sequences
        growth_count = len(self._position_std_growth_m)
        contraction_count = len(self._reobservation_contraction_m)
        return {
            "occlusion_qualifying_sequence_count": float(sequence_count),
            "occlusion_identity_survival_count": float(self._identity_survivals),
            "occlusion_identity_survival_rate": (
                self._identity_survivals / sequence_count if sequence_count else None
            ),
            "occlusion_growth_evaluated_sequence_count": float(growth_count),
            "occlusion_pre_position_std_mean_m": self._mean_or_none(self._pre_position_std_m),
            "occlusion_peak_position_std_mean_m": self._mean_or_none(
                self._peak_occluded_position_std_m
            ),
            "occlusion_position_std_growth_mean_m": self._mean_or_none(self._position_std_growth_m),
            "occlusion_position_std_growth_positive_rate": (
                sum(value > 0.0 for value in self._position_std_growth_m) / growth_count
                if growth_count
                else None
            ),
            "occlusion_reobservation_contraction_evaluated_sequence_count": float(
                contraction_count
            ),
            "occlusion_reobservation_position_std_mean_m": self._mean_or_none(
                self._reobservation_position_std_m
            ),
            "occlusion_reobservation_std_contraction_mean_m": self._mean_or_none(
                self._reobservation_contraction_m
            ),
            "occlusion_reobservation_std_contraction_positive_rate": (
                sum(value > 0.0 for value in self._reobservation_contraction_m) / contraction_count
                if contraction_count
                else None
            ),
        }

    def _follow_occluded_prediction(
        self,
        *,
        track: _OcclusionTrack,
        prediction_slot_by_id: dict[int, int],
        position_std_m: Tensor,
        sample_uncertainty: bool,
    ) -> None:
        prediction_slot = prediction_slot_by_id.get(track.prediction_id)
        if prediction_slot is None:
            track.prediction_present_throughout = False
            return
        if not sample_uncertainty:
            return
        value = float(position_std_m[prediction_slot])
        if not math.isfinite(value):
            return
        if track.peak_occluded_position_std_m is None:
            track.peak_occluded_position_std_m = value
        else:
            track.peak_occluded_position_std_m = max(
                track.peak_occluded_position_std_m,
                value,
            )

    def _finish_occlusion(
        self,
        *,
        track: _OcclusionTrack,
        reliable_slot: int | None,
        predicted_ids: Tensor,
        position_std_m: Tensor,
    ) -> None:
        self._qualifying_sequences += 1
        same_identity = (
            reliable_slot is not None
            and int(predicted_ids[reliable_slot]) == track.prediction_id
            and track.prediction_present_throughout
        )
        if same_identity:
            self._identity_survivals += 1

        peak_std = track.peak_occluded_position_std_m
        if peak_std is None:
            return
        growth = peak_std - track.pre_position_std_m
        self._pre_position_std_m.append(track.pre_position_std_m)
        self._peak_occluded_position_std_m.append(peak_std)
        self._position_std_growth_m.append(growth)

        if not same_identity or reliable_slot is None:
            return
        reobservation_std = float(position_std_m[reliable_slot])
        if not math.isfinite(reobservation_std):
            return
        self._reobservation_position_std_m.append(reobservation_std)
        self._reobservation_contraction_m.append(peak_std - reobservation_std)

    def _replace_visible_anchor(
        self,
        *,
        key: tuple[int, int],
        reliable_slot: int | None,
        predicted_ids: Tensor,
        position_std_m: Tensor,
    ) -> None:
        if reliable_slot is None:
            self._tracks.pop(key, None)
            return
        prediction_id = int(predicted_ids[reliable_slot])
        std = float(position_std_m[reliable_slot])
        if prediction_id < 0 or not math.isfinite(std):
            self._tracks.pop(key, None)
            return
        self._tracks[key] = _OcclusionTrack(
            prediction_id=prediction_id,
            pre_position_std_m=std,
        )

    @staticmethod
    def _mean_or_none(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    @staticmethod
    def _validate_shapes(
        *,
        predicted_ids: Tensor,
        predicted_active: Tensor,
        position_std_m: Tensor,
        target_ids: Tensor,
        target_active: Tensor,
        target_visible_fraction: Tensor,
        matched_target_indices: Tensor,
        reliable_visible_matches: Tensor,
    ) -> None:
        if predicted_ids.ndim != 2 or target_ids.ndim != 2:
            raise ValueError("occlusion metrics expect batched [B,N] tensors")
        if predicted_active.shape != predicted_ids.shape:
            raise ValueError("predicted_active must match predicted_ids")
        if position_std_m.shape != predicted_ids.shape:
            raise ValueError("position_std_m must match predicted_ids")
        if matched_target_indices.shape != predicted_ids.shape:
            raise ValueError("matched_target_indices must match predicted_ids")
        if reliable_visible_matches.shape != predicted_ids.shape:
            raise ValueError("reliable_visible_matches must match predicted_ids")
        if target_active.shape != target_ids.shape:
            raise ValueError("target_active must match target_ids")
        if target_visible_fraction.shape != target_ids.shape:
            raise ValueError("target_visible_fraction must match target_ids")
        if predicted_ids.shape[0] != target_ids.shape[0]:
            raise ValueError("prediction and target batch sizes must match")


__all__ = ["OcclusionTransitionAccumulator"]
