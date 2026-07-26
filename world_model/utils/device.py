"""Explicit device selection without hidden global state."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceInfo:
    """Resolved execution device and the capabilities observed at selection."""

    device: torch.device
    requested: str
    torch_version: str
    mps_built: bool
    mps_available: bool
    cuda_available: bool
    precision: str = "float32"


def select_device(preference: str = "auto") -> DeviceInfo:
    """Resolve ``auto|cpu|mps|cuda`` and fail clearly for unavailable requests."""

    preference = preference.lower()
    if preference not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError(f"Unsupported device preference {preference!r}")

    mps_built = bool(torch.backends.mps.is_built())
    mps_available = bool(torch.backends.mps.is_available())
    cuda_available = bool(torch.cuda.is_available())

    if preference == "cuda":
        if not cuda_available:
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        device = torch.device("cuda")
    elif preference == "mps":
        if not mps_available:
            raise RuntimeError(
                "MPS was requested but is unavailable "
                f"(compiled={mps_built}, available={mps_available})"
            )
        device = torch.device("mps")
    elif preference == "cpu":
        device = torch.device("cpu")
    elif cuda_available:
        device = torch.device("cuda")
    elif mps_available:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return DeviceInfo(
        device=device,
        requested=preference,
        torch_version=torch.__version__,
        mps_built=mps_built,
        mps_available=mps_available,
        cuda_available=cuda_available,
    )
