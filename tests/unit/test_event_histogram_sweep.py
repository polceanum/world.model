from scripts.sweep_event_histograms import sweep_report


def test_sweep_aggregates_labels_and_counts() -> None:
    report = {
        "candidate_names": ["learned"],
        "episode_results": [
            {
                "event_probability_positive_histograms": {"0.5": [[1, 2, 0, 0, 0, 0, 0, 0, 0, 0]]},
                "event_probability_negative_histograms": {"0.5": [[3, 1, 0, 0, 0, 0, 0, 0, 0, 0]]},
            },
            {
                "event_probability_positive_histograms": {"0.5": [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]]},
                "event_probability_negative_histograms": {"0.5": [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]},
            },
        ],
    }
    metrics = sweep_report(report)["0.5"]["learned"]
    threshold_zero = metrics[0]
    assert threshold_zero["true_positive"] == 4
    assert threshold_zero["false_positive"] == 5
    threshold_point_one = metrics[1]
    assert threshold_point_one["true_positive"] == 3
    assert threshold_point_one["false_positive"] == 1
    assert threshold_point_one["recall"] == 0.75
