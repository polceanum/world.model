"""Simple transparent identity/lifecycle metrics for padded toy episodes."""

from __future__ import annotations

from torch import Tensor


def tracking_metrics(
    predicted_ids: Tensor,
    target_ids: Tensor,
    matched_target_indices: Tensor,
    pair_mask: Tensor,
) -> dict[str, float]:
    """Measure persistent-ID switches along already associated trajectories.

    Inputs are ``[T,N]``. ``matched_target_indices`` maps each belief slot to a
    target slot or ``-1`` on each frame.
    """

    if predicted_ids.ndim != 2:
        raise ValueError("tracking metrics expect [T,N] tensors")
    associations = 0
    switches = 0
    last_predicted_for_target: dict[int, int] = {}
    for step in range(predicted_ids.shape[0]):
        for slot in range(predicted_ids.shape[1]):
            if not bool(pair_mask[step, slot]):
                continue
            target_slot = int(matched_target_indices[step, slot])
            if target_slot < 0:
                continue
            target_id = int(target_ids[step, target_slot])
            predicted_id = int(predicted_ids[step, slot])
            if target_id < 0 or predicted_id < 0:
                continue
            associations += 1
            previous = last_predicted_for_target.get(target_id)
            if previous is not None and previous != predicted_id:
                switches += 1
            last_predicted_for_target[target_id] = predicted_id
    rate = switches / associations if associations else float("nan")
    return {
        "identity_switches": float(switches),
        "object_frame_associations": float(associations),
        "identity_switch_rate": float(rate),
    }
