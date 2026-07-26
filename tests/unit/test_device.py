import pytest
import torch

from world_model.utils.device import select_device


def test_cpu_selection_is_explicit() -> None:
    info = select_device("cpu")
    assert info.device == torch.device("cpu")
    assert info.requested == "cpu"
    assert info.torch_version == torch.__version__


def test_auto_returns_available_backend() -> None:
    info = select_device("auto")
    if info.device.type == "cuda":
        assert info.cuda_available
    elif info.device.type == "mps":
        assert info.mps_available
    else:
        assert info.device.type == "cpu"


def test_unknown_device_fails() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        select_device("quantum")
