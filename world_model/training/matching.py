"""Transparent Hungarian matching for supervised RGB proposals."""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor


def match_measurements_to_targets(
    predicted_values: Tensor,
    target_values: Tensor,
    target_mask: Tensor,
    *,
    existence_logits: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Align targets to proposal slots.

    Returns ``aligned_targets [B,M,D]``, ``matched [B,M]``, and
    ``target_indices [B,M]`` with ``-1`` for unmatched proposals.
    """

    if predicted_values.ndim != 3 or target_values.ndim != 3:
        raise ValueError("predicted and target values must have shape [B,M,D]")
    if predicted_values.shape[0] != target_values.shape[0]:
        raise ValueError("batch dimensions must match")
    if predicted_values.shape[-1] != target_values.shape[-1]:
        raise ValueError("measurement dimensions must match")
    batch, proposals, dimensions = predicted_values.shape
    aligned = torch.zeros_like(predicted_values)
    matched = torch.zeros(batch, proposals, dtype=torch.bool, device=predicted_values.device)
    indices = torch.full((batch, proposals), -1, dtype=torch.int64, device=predicted_values.device)
    for batch_index in range(batch):
        valid_targets = torch.nonzero(target_mask[batch_index], as_tuple=False).flatten()
        if valid_targets.numel() == 0 or proposals == 0:
            continue
        prediction = predicted_values[batch_index, :, :4].detach().cpu()
        target = target_values[batch_index, valid_targets, :4].detach().cpu()
        # Centre dominates; log-radius/inverse-depth retain physically meaningful scale.
        scales = prediction.new_tensor((1.0, 1.0, 0.5, 0.25))
        cost = torch.cdist(prediction * scales, target * scales, p=1)
        if existence_logits is not None:
            confidence_cost = -existence_logits[batch_index].detach().cpu().sigmoid()
            cost = cost + 0.05 * confidence_cost[:, None]
        rows, columns = linear_sum_assignment(np.asarray(cost))
        row_tensor = torch.as_tensor(rows, device=predicted_values.device, dtype=torch.int64)
        target_tensor = valid_targets[
            torch.as_tensor(columns, device=valid_targets.device, dtype=torch.int64)
        ]
        aligned[batch_index, row_tensor] = target_values[batch_index, target_tensor]
        matched[batch_index, row_tensor] = True
        indices[batch_index, row_tensor] = target_tensor
    if aligned.shape != (batch, proposals, dimensions):
        raise AssertionError("matching shape invariant failed")
    return aligned, matched, indices
