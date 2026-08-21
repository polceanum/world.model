from __future__ import annotations

import json

import pytest

from world_model.evaluation.latency import paired_latency_guardrail


def _latency_metrics(
    *,
    global_ms: float = 10.0,
    fast_ms: float = 4.0,
    rollout_ms: float = 20.0,
) -> dict[str, float]:
    global_count = 8.0
    fast_count = 20.0
    rollout_count = 12.0
    return {
        "rgb_global_update_latency_mean_ms": global_ms,
        "rgb_global_update_latency_sum_ms": global_ms * global_count,
        "rgb_global_update_latency_sample_count": global_count,
        "rgb_fast_update_latency_mean_ms": fast_ms,
        "rgb_fast_update_latency_sum_ms": fast_ms * fast_count,
        "rgb_fast_update_latency_sample_count": fast_count,
        "future_rollout_latency_mean_ms": rollout_ms,
        "future_rollout_latency_sum_ms": rollout_ms * rollout_count,
        "future_rollout_latency_sample_count": rollout_count,
    }


def test_paired_latency_guardrail_accepts_complete_matched_evidence() -> None:
    reference = _latency_metrics()
    candidate = _latency_metrics(global_ms=11.0, fast_ms=4.2, rollout_ms=18.0)

    result = paired_latency_guardrail(candidate, reference)

    assert result.supported
    assert result.promotion_eligible
    assert not result.failures
    assert result.ratios == pytest.approx(
        {
            "rgb_global_update": 1.1,
            "rgb_fast_update": 1.05,
            "future_rollout": 0.9,
        }
    )
    assert result.metrics()["latency_guardrail_passed"] is True
    assert result.metrics()["latency_guardrail_promotion_eligible"] is True
    assert "comprehensive_promotion_eligible" not in result.metrics()


def test_paired_latency_guardrail_rejects_one_cost_regression() -> None:
    result = paired_latency_guardrail(
        _latency_metrics(rollout_ms=22.1),
        _latency_metrics(),
    )

    assert result.supported
    assert not result.promotion_eligible
    assert result.metrics()["latency_guardrail_passed"] is False
    assert {failure["metric"] for failure in result.failures} == {"future_rollout"}


@pytest.mark.parametrize(
    "unsupported_metrics",
    [
        {},
        {
            **_latency_metrics(),
            "rgb_fast_update_latency_sample_count": 0.0,
        },
        {
            **_latency_metrics(),
            "future_rollout_latency_sample_count": 1.5,
        },
    ],
)
def test_paired_latency_guardrail_fails_closed_without_complete_support(
    unsupported_metrics: dict[str, float],
) -> None:
    result = paired_latency_guardrail(unsupported_metrics, _latency_metrics())
    serialized = json.dumps(result.metrics(), allow_nan=False, sort_keys=True)

    assert not result.supported
    assert not result.promotion_eligible
    assert result.metrics()["latency_guardrail_supported"] is False
    assert result.metrics()["latency_guardrail_promotion_eligible"] is False
    assert "paired_support_required" in serialized


def test_paired_latency_guardrail_validates_declared_limit() -> None:
    with pytest.raises(ValueError, match="at least one"):
        paired_latency_guardrail({}, {}, maximum_ratio=0.99)


@pytest.mark.parametrize(
    "candidate",
    [
        {
            **_latency_metrics(),
            "rgb_fast_update_latency_sum_ms": 1.0,
        },
        {
            **_latency_metrics(),
            "rgb_fast_update_latency_sum_ms": 4.0 * 19.0,
            "rgb_fast_update_latency_sample_count": 19.0,
        },
    ],
)
def test_paired_latency_guardrail_rejects_unmatched_or_contradictory_additive_evidence(
    candidate: dict[str, float],
) -> None:
    result = paired_latency_guardrail(candidate, _latency_metrics())

    assert not result.supported
    assert not result.promotion_eligible
    assert result.failures[0]["direction"] == "paired_support_required"
