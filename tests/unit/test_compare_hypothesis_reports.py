from scripts.compare_hypothesis_reports import compare_reports


def _report(x: float, *, event_f1: float = 0.5) -> dict:
    return {
        "horizons_seconds": [1.0],
        "episode_results": [
            {
                "selected_rmse_m": {"1.0": [x, x, x]},
                "selected_lifecycle_mismatch": {"1.0": 2},
                "selected_identity_coverage": {"1.0": 3},
                "selected_event_metrics": {"1.0": {"collision_f1": event_f1}},
                "selected_mean_position_std_m": {"1.0": 0.4},
            }
        ],
    }


def test_report_comparator_accepts_non_regression() -> None:
    result = compare_reports(_report(1.0), _report(0.9, event_f1=0.6))
    assert result["passed"]
    assert result["regressions"] == []


def test_report_comparator_reports_axis_and_event_regressions() -> None:
    result = compare_reports(_report(1.0, event_f1=0.5), _report(1.1, event_f1=0.4))
    assert not result["passed"]
    assert "1.0.rmse_axis_0" in result["regressions"]
    assert "1.0.event_f1" in result["regressions"]
