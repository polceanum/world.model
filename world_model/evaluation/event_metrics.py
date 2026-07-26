"""Collision/contact metrics without heavyweight external packages."""

from __future__ import annotations

from torch import Tensor


def binary_event_metrics(
    scores: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    threshold: float = 0.5,
    scores_are_logits: bool = True,
) -> dict[str, float]:
    selected_scores = scores.masked_select(mask)
    selected_target = target.bool().masked_select(mask)
    if selected_scores.numel() == 0:
        return {
            "event_precision": float("nan"),
            "event_recall": float("nan"),
            "event_f1": float("nan"),
            "event_false_positive_rate": float("nan"),
        }
    probability = selected_scores.sigmoid() if scores_are_logits else selected_scores
    prediction = probability >= threshold
    true_positive = (prediction & selected_target).sum().float()
    false_positive = (prediction & ~selected_target).sum().float()
    false_negative = (~prediction & selected_target).sum().float()
    true_negative = (~prediction & ~selected_target).sum().float()
    precision = true_positive / (true_positive + false_positive).clamp_min(1)
    recall = true_positive / (true_positive + false_negative).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1.0e-8)
    false_positive_rate = false_positive / (false_positive + true_negative).clamp_min(1)
    return {
        "event_precision": float(precision),
        "event_recall": float(recall),
        "event_f1": float(f1),
        "event_false_positive_rate": float(false_positive_rate),
    }
