"""Device-aware component latency measurement."""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

_PROMOTION_LATENCY_PREFIXES = (
    "rgb_global_update",
    "rgb_fast_update",
    "future_rollout",
)
_DEFAULT_PAIRED_LATENCY_MAXIMUM_RATIO = 1.10


@dataclass(frozen=True)
class PairedLatencyGuardrail:
    """Result of a matched candidate/reference latency comparison."""

    supported: bool
    promotion_eligible: bool
    maximum_ratio: float
    ratios: dict[str, float] = field(default_factory=dict)
    failures: tuple[dict[str, float | str | None], ...] = ()

    def metrics(self) -> dict[str, Any]:
        return {
            "latency_guardrail_supported": self.supported,
            "latency_guardrail_passed": self.supported and not self.failures,
            # This helper owns only the paired cost contract. The caller must
            # combine it with physical/accuracy guardrails before emitting a
            # comprehensive promotion decision.
            "latency_guardrail_promotion_eligible": self.promotion_eligible,
            "latency_guardrail_maximum_ratio": self.maximum_ratio,
            "latency_guardrail_ratios": dict(self.ratios),
            "latency_guardrail_failures": [dict(failure) for failure in self.failures],
        }


def paired_latency_guardrail(
    candidate_metrics: Mapping[str, Any],
    reference_metrics: Mapping[str, Any],
    *,
    maximum_ratio: float = _DEFAULT_PAIRED_LATENCY_MAXIMUM_RATIO,
) -> PairedLatencyGuardrail:
    """Compare matched component timings and fail closed on missing support.

    The caller owns the pairing contract (same checkpoint protocol, device,
    precision, manifest, and execution policy). This pure helper consumes the
    resulting reports only. Timing stays outside deterministic physical metric
    hashes because wall-clock observations are not deterministic tensors.
    """

    if not math.isfinite(maximum_ratio) or maximum_ratio < 1.0:
        raise ValueError("paired latency maximum_ratio must be finite and at least one")
    ratios: dict[str, float] = {}
    failures: list[dict[str, float | str | None]] = []
    supported = True
    for prefix in _PROMOTION_LATENCY_PREFIXES:
        mean_key = f"{prefix}_latency_mean_ms"
        sum_key = f"{prefix}_latency_sum_ms"
        count_key = f"{prefix}_latency_sample_count"
        values: dict[str, tuple[float, float]] = {}
        for role, metrics in (("candidate", candidate_metrics), ("reference", reference_metrics)):
            try:
                mean = float(metrics[mean_key])
                total = float(metrics[sum_key])
                count = float(metrics[count_key])
            except (KeyError, TypeError, ValueError):
                supported = False
                failures.append(
                    {
                        "metric": prefix,
                        "direction": "paired_support_required",
                        "candidate": None,
                        "reference": None,
                        "limit": maximum_ratio,
                        "delta": None,
                        "detail": f"missing_or_invalid_{role}_latency_evidence",
                    }
                )
                break
            if (
                not math.isfinite(mean)
                or mean < 0.0
                or not math.isfinite(total)
                or total < 0.0
                or not math.isfinite(count)
                or count <= 0.0
                or not count.is_integer()
            ):
                supported = False
                failures.append(
                    {
                        "metric": prefix,
                        "direction": "paired_support_required",
                        "candidate": mean if role == "candidate" else None,
                        "reference": mean if role == "reference" else None,
                        "limit": maximum_ratio,
                        "delta": None,
                        "detail": f"unsupported_{role}_latency_evidence",
                    }
                )
                break
            if not math.isclose(total, mean * count, rel_tol=1.0e-9, abs_tol=1.0e-7):
                supported = False
                failures.append(
                    {
                        "metric": prefix,
                        "direction": "paired_support_required",
                        "candidate": mean if role == "candidate" else None,
                        "reference": mean if role == "reference" else None,
                        "limit": maximum_ratio,
                        "delta": None,
                        "detail": f"contradictory_{role}_latency_sum_count_evidence",
                    }
                )
                break
            values[role] = (mean, count)
        if len(values) != 2:
            continue
        candidate_mean, candidate_count = values["candidate"]
        reference_mean, reference_count = values["reference"]
        if candidate_count != reference_count:
            supported = False
            failures.append(
                {
                    "metric": prefix,
                    "direction": "paired_support_required",
                    "candidate": candidate_count,
                    "reference": reference_count,
                    "limit": reference_count,
                    "delta": candidate_count - reference_count,
                    "detail": "mismatched_latency_sample_count",
                }
            )
            continue
        if reference_mean <= 0.0:
            supported = False
            failures.append(
                {
                    "metric": prefix,
                    "direction": "positive_reference_required",
                    "candidate": candidate_mean,
                    "reference": reference_mean,
                    "limit": maximum_ratio,
                    "delta": candidate_mean - reference_mean,
                }
            )
            continue
        ratio = candidate_mean / reference_mean
        ratios[prefix] = ratio
        if ratio > maximum_ratio:
            failures.append(
                {
                    "metric": prefix,
                    "direction": "maximum_ratio",
                    "candidate": candidate_mean,
                    "reference": reference_mean,
                    "limit": reference_mean * maximum_ratio,
                    "delta": candidate_mean - reference_mean,
                }
            )
    return PairedLatencyGuardrail(
        supported=supported,
        promotion_eligible=supported and not failures,
        maximum_ratio=maximum_ratio,
        ratios=ratios,
        failures=tuple(failures),
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def benchmark_callable(
    function: Callable[[], Any],
    *,
    device: torch.device,
    warmup: int = 2,
    repeats: int = 10,
) -> dict[str, float]:
    for _ in range(warmup):
        function()
    synchronize(device)
    elapsed: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        synchronize(device)
        elapsed.append((time.perf_counter() - start) * 1000)
    return {
        "latency_mean_ms": statistics.mean(elapsed),
        "latency_median_ms": statistics.median(elapsed),
        "latency_min_ms": min(elapsed),
        "latency_max_ms": max(elapsed),
    }
