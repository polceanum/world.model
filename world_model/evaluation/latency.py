"""Device-aware component latency measurement."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from typing import Any

import torch


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
