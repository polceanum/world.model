"""Shared deterministic test configuration."""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True)
def deterministic_test_seed() -> None:
    torch.manual_seed(0)
