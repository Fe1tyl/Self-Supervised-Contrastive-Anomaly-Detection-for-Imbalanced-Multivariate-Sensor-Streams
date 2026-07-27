import numpy as np

from hai_repro.metrics import (
    average_precision,
    evaluate_records,
    roc_auc,
)


def test_perfect_and_constant_rank_metrics() -> None:
    labels = np.array([0, 1, 0, 1], dtype=np.uint8)
    perfect = np.array([0.1, 0.4, 0.2, 0.3])
    constant = np.ones(4)
    assert average_precision(labels, perfect) == 1.0
    assert roc_auc(labels, perfect) == 1.0
    assert average_precision(labels, constant) == 0.5
    assert roc_auc(labels, constant) == 0.5


def test_event_coverage_exposes_one_point_hit() -> None:
    labels = np.zeros(20, dtype=np.uint8)
    labels[5:15] = 1
    scores = np.zeros(20, dtype=np.float64)
    scores[5] = 1.0
    result = evaluate_records(
        [
            {
                "file_id": "test",
                "y_true": labels,
                "score": scores,
                "timestamps": np.arange(20),
            }
        ],
        threshold=0.5,
        coverage_thresholds=(0.1, 0.5),
    )
    pooled = result["pooled"]
    assert pooled["overlap_event_recall"] == 1.0
    assert pooled["duration_recall"] == 0.1
    assert pooled["events_covered_10pct"] == 1
    assert pooled["events_covered_50pct"] == 0
    assert pooled["events_missed_50pct"] == 1


def test_events_never_cross_file_boundaries() -> None:
    records = [
        {
            "file_id": "a",
            "y_true": np.array([0, 1], dtype=np.uint8),
            "score": np.array([0.0, 1.0]),
            "timestamps": np.array([0, 1]),
        },
        {
            "file_id": "b",
            "y_true": np.array([1, 0], dtype=np.uint8),
            "score": np.array([1.0, 0.0]),
            "timestamps": np.array([2, 3]),
        },
    ]
    result = evaluate_records(records, threshold=0.5)
    assert result["pooled"]["reference_event_count"] == 2
    assert result["pooled"]["predicted_event_count"] == 2

