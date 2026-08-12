from __future__ import annotations

import torch

from scripts.measure_attention_node_activity import _gradient_enabled_active_residual


def test_emitted_trace_excludes_no_grad_attention_calls() -> None:
    output = torch.tensor(
        [[[0.0, 0.25, -0.5], [0.75, -1.0, 0.5]]],
        requires_grad=True,
    )
    active = torch.tensor([[True, False]])

    selected = _gradient_enabled_active_residual(output, active, maximum=0.5)
    assert selected is not None
    torch.testing.assert_close(
        selected,
        0.5 * torch.tanh(output.detach()[0, :1]).to(torch.float64),
    )

    with torch.no_grad():
        excluded = _gradient_enabled_active_residual(output, active, maximum=0.5)
    assert excluded is None


def test_emitted_trace_rejects_incompatible_active_mask() -> None:
    output = torch.zeros(1, 2, 3)
    active = torch.ones(1, 3, dtype=torch.bool)

    with torch.no_grad():
        # No-gradient calls are excluded before their irrelevant shape is read.
        assert _gradient_enabled_active_residual(output, active, maximum=0.5) is None

    try:
        _gradient_enabled_active_residual(output, active, maximum=0.5)
    except ValueError as error:
        assert "shapes are incompatible" in str(error)
    else:
        raise AssertionError("incompatible trace shapes should fail")
