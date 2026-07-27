from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import numpy as np


def _validate_binary_inputs(
    y_true: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true, dtype=np.uint8).reshape(-1)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.shape != values.shape:
        raise ValueError("Labels and scores have different lengths")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels are not binary")
    if not np.isfinite(values).all():
        raise ValueError("Scores contain NaN or infinity")
    return labels, values


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    labels, values = _validate_binary_inputs(y_true, scores)
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(values, kind="mergesort")[::-1]
    labels = labels[order]
    values = values[order]
    distinct_ends = np.r_[np.flatnonzero(np.diff(values)), len(values) - 1]
    true_positives = np.cumsum(labels, dtype=np.int64)[distinct_ends]
    predicted_positives = distinct_ends + 1
    precision = true_positives / predicted_positives
    recall = true_positives / positives
    recall_increment = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_increment * precision))


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    labels, values = _validate_binary_inputs(y_true, scores)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_labels = labels[order]
    positive_rank_sum = 0.0
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        average_rank = ((start + 1) + stop) / 2.0
        positive_rank_sum += average_rank * int(sorted_labels[start:stop].sum())
        start = stop
    statistic = positive_rank_sum - positives * (positives + 1) / 2.0
    return float(statistic / (positives * negatives))


def binary_counts(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, int]:
    labels = np.asarray(y_true, dtype=np.uint8).reshape(-1)
    predictions = np.asarray(y_pred, dtype=np.uint8).reshape(-1)
    if labels.shape != predictions.shape:
        raise ValueError("Labels and predictions have different lengths")
    return {
        "tp": int(np.sum((labels == 1) & (predictions == 1))),
        "fp": int(np.sum((labels == 0) & (predictions == 1))),
        "tn": int(np.sum((labels == 0) & (predictions == 0))),
        "fn": int(np.sum((labels == 1) & (predictions == 0))),
    }


def precision_recall_f1(counts: dict[str, int]) -> dict[str, float]:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


@dataclass(frozen=True)
class Interval:
    file_id: str
    start_index: int
    end_index: int
    start_time: int
    end_time: int

    @property
    def points(self) -> int:
        return self.end_index - self.start_index + 1


def extract_intervals(
    binary: np.ndarray,
    timestamps: np.ndarray,
    file_id: str,
) -> list[Interval]:
    values = np.asarray(binary, dtype=np.uint8).reshape(-1)
    times = np.asarray(timestamps, dtype=np.int64).reshape(-1)
    if values.shape != times.shape:
        raise ValueError("Binary vector and timestamps have different lengths")
    padded = np.r_[0, values.astype(np.int8), 0]
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1) - 1
    return [
        Interval(
            file_id=file_id,
            start_index=int(start),
            end_index=int(stop),
            start_time=int(times[start]),
            end_time=int(times[stop]),
        )
        for start, stop in zip(starts, stops, strict=True)
    ]


def overlap_points(first: Interval, second: Interval) -> int:
    if first.file_id != second.file_id:
        return 0
    return max(
        0,
        min(first.end_index, second.end_index)
        - max(first.start_index, second.start_index)
        + 1,
    )


def maximum_overlap_matching(
    references: Sequence[Interval],
    predictions: Sequence[Interval],
) -> list[tuple[int, int]]:
    adjacency: list[list[int]] = []
    for reference in references:
        candidates = [
            (index, overlap_points(reference, prediction))
            for index, prediction in enumerate(predictions)
        ]
        adjacency.append(
            [
                index
                for index, overlap in sorted(
                    candidates, key=lambda item: item[1], reverse=True
                )
                if overlap > 0
            ]
        )
    prediction_to_reference: dict[int, int] = {}

    def assign(reference_index: int, visited: set[int]) -> bool:
        for prediction_index in adjacency[reference_index]:
            if prediction_index in visited:
                continue
            visited.add(prediction_index)
            previous = prediction_to_reference.get(prediction_index)
            if previous is None or assign(previous, visited):
                prediction_to_reference[prediction_index] = reference_index
                return True
        return False

    for reference_index in range(len(references)):
        assign(reference_index, set())
    return sorted(
        (reference_index, prediction_index)
        for prediction_index, reference_index in prediction_to_reference.items()
    )


def interval_rows(
    intervals: Sequence[Interval], interval_type: str
) -> list[dict[str, Any]]:
    return [
        {
            "interval_type": interval_type,
            **asdict(interval),
            "points": interval.points,
        }
        for interval in intervals
    ]


def event_metrics_for_file(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: np.ndarray,
    file_id: str,
    sampling_interval_seconds: float,
    coverage_thresholds: Sequence[float],
) -> dict[str, Any]:
    labels = np.asarray(y_true, dtype=np.uint8)
    predictions = np.asarray(y_pred, dtype=np.uint8)
    references = extract_intervals(labels, timestamps, file_id)
    predicted_intervals = extract_intervals(predictions, timestamps, file_id)
    matches = maximum_overlap_matching(references, predicted_intervals)
    matched_reference = {reference for reference, _ in matches}
    matched_prediction = {prediction for _, prediction in matches}
    match_count = len(matches)
    event_precision = (
        match_count / len(predicted_intervals) if predicted_intervals else 0.0
    )
    event_recall = match_count / len(references) if references else 0.0
    event_f1 = (
        2.0 * event_precision * event_recall / (event_precision + event_recall)
        if event_precision + event_recall
        else 0.0
    )

    coverage_ratios: list[float] = []
    fragments: list[int] = []
    for reference in references:
        segment = predictions[reference.start_index : reference.end_index + 1]
        coverage_ratios.append(float(segment.mean()))
        fragments.append(
            sum(overlap_points(reference, prediction) > 0 for prediction in predicted_intervals)
        )

    delays = []
    match_rows = []
    for reference_index, prediction_index in matches:
        reference = references[reference_index]
        prediction = predicted_intervals[prediction_index]
        first_alarm_index = max(reference.start_index, prediction.start_index)
        delay = (first_alarm_index - reference.start_index) * sampling_interval_seconds
        delays.append(float(delay))
        match_rows.append(
            {
                "file_id": file_id,
                "reference_index": reference_index,
                "prediction_index": prediction_index,
                "overlap_points": overlap_points(reference, prediction),
                "delay_seconds": float(delay),
            }
        )

    point_counts = binary_counts(labels, predictions)
    duration = precision_recall_f1(point_counts)
    normal_hours = (
        point_counts["tn"] + point_counts["fp"]
    ) * sampling_interval_seconds / 3600.0
    result: dict[str, Any] = {
        "file_id": file_id,
        "reference_event_count": len(references),
        "predicted_event_count": len(predicted_intervals),
        "matched_reference_event_count": len(matched_reference),
        "matched_predicted_event_count": len(matched_prediction),
        "missed_reference_event_count": len(references) - len(matched_reference),
        "false_predicted_event_count": len(predicted_intervals)
        - len(matched_prediction),
        "overlap_event_precision": event_precision,
        "overlap_event_recall": event_recall,
        "overlap_event_f1": event_f1,
        "duration_precision": duration["precision"],
        "duration_recall": duration["recall"],
        "duration_f1": duration["f1"],
        "mean_reference_coverage": float(np.mean(coverage_ratios))
        if coverage_ratios
        else float("nan"),
        "median_reference_coverage": float(np.median(coverage_ratios))
        if coverage_ratios
        else float("nan"),
        "mean_fragments_per_reference": float(np.mean(fragments))
        if fragments
        else float("nan"),
        "false_alarm_events_per_normal_hour": (
            (len(predicted_intervals) - len(matched_prediction)) / normal_hours
            if normal_hours
            else float("nan")
        ),
        "median_detection_delay_seconds": float(np.median(delays))
        if delays
        else float("nan"),
        "detection_delay_iqr_seconds": (
            float(np.quantile(delays, 0.75) - np.quantile(delays, 0.25))
            if delays
            else float("nan")
        ),
        "_references": references,
        "_predictions": predicted_intervals,
        "_matches": match_rows,
        "_coverage_ratios": coverage_ratios,
        "_delays": delays,
        "_point_counts": point_counts,
    }
    for threshold in coverage_thresholds:
        key = f"{int(round(threshold * 100))}pct"
        detected = sum(value >= threshold for value in coverage_ratios)
        result[f"events_covered_{key}"] = detected
        result[f"events_missed_{key}"] = len(references) - detected
        result[f"event_recall_{key}"] = (
            detected / len(references) if references else float("nan")
        )
    return result


def evaluate_records(
    records: Sequence[dict[str, Any]],
    threshold: float,
    sampling_interval_seconds: float = 1.0,
    coverage_thresholds: Sequence[float] = (0.1, 0.5),
) -> dict[str, Any]:
    per_file: list[dict[str, Any]] = []
    internal: list[dict[str, Any]] = []
    for record in records:
        labels, scores = _validate_binary_inputs(record["y_true"], record["score"])
        predictions = (scores > threshold).astype(np.uint8)
        point_counts = binary_counts(labels, predictions)
        point_metrics = precision_recall_f1(point_counts)
        event = event_metrics_for_file(
            labels,
            predictions,
            record["timestamps"],
            str(record["file_id"]),
            sampling_interval_seconds,
            coverage_thresholds,
        )
        row = {
            "file_id": str(record["file_id"]),
            "scored_points": len(labels),
            "anomaly_points": int(labels.sum()),
            "threshold": float(threshold),
            "average_precision": average_precision(labels, scores),
            "auroc": roc_auc(labels, scores),
            **{f"point_{key}": value for key, value in point_metrics.items()},
            **point_counts,
            **{key: value for key, value in event.items() if not key.startswith("_")},
        }
        per_file.append(row)
        internal.append(event)

    labels = np.concatenate([np.asarray(record["y_true"]) for record in records])
    scores = np.concatenate([np.asarray(record["score"]) for record in records])
    predictions = (scores > threshold).astype(np.uint8)
    point_counts = binary_counts(labels, predictions)
    point_metrics = precision_recall_f1(point_counts)
    total_references = sum(item["reference_event_count"] for item in internal)
    total_predictions = sum(item["predicted_event_count"] for item in internal)
    total_matches = sum(item["matched_reference_event_count"] for item in internal)
    event_precision = total_matches / total_predictions if total_predictions else 0.0
    event_recall = total_matches / total_references if total_references else 0.0
    event_f1 = (
        2 * event_precision * event_recall / (event_precision + event_recall)
        if event_precision + event_recall
        else 0.0
    )
    coverage = [
        value for item in internal for value in item["_coverage_ratios"]
    ]
    delays = [value for item in internal for value in item["_delays"]]
    false_events = sum(item["false_predicted_event_count"] for item in internal)
    normal_hours = (
        point_counts["tn"] + point_counts["fp"]
    ) * sampling_interval_seconds / 3600.0
    pooled: dict[str, Any] = {
        "scored_points": len(labels),
        "anomaly_points": int(labels.sum()),
        "anomaly_prevalence": float(labels.mean()),
        "threshold": float(threshold),
        "average_precision": average_precision(labels, scores),
        "auroc": roc_auc(labels, scores),
        **{f"point_{key}": value for key, value in point_metrics.items()},
        **point_counts,
        "reference_event_count": total_references,
        "predicted_event_count": total_predictions,
        "matched_reference_event_count": total_matches,
        "missed_reference_event_count": total_references - total_matches,
        "false_predicted_event_count": false_events,
        "overlap_event_precision": event_precision,
        "overlap_event_recall": event_recall,
        "overlap_event_f1": event_f1,
        "duration_precision": point_metrics["precision"],
        "duration_recall": point_metrics["recall"],
        "duration_f1": point_metrics["f1"],
        "mean_reference_coverage": float(np.mean(coverage)),
        "median_reference_coverage": float(np.median(coverage)),
        "false_alarm_events_per_normal_hour": false_events / normal_hours,
        "median_detection_delay_seconds": float(np.median(delays))
        if delays
        else float("nan"),
        "detection_delay_iqr_seconds": (
            float(np.quantile(delays, 0.75) - np.quantile(delays, 0.25))
            if delays
            else float("nan")
        ),
        "macro_average_precision": float(
            np.mean([row["average_precision"] for row in per_file])
        ),
        "macro_auroc": float(np.mean([row["auroc"] for row in per_file])),
        "macro_point_f1": float(np.mean([row["point_f1"] for row in per_file])),
        "macro_overlap_event_f1": float(
            np.mean([row["overlap_event_f1"] for row in per_file])
        ),
    }
    for coverage_threshold in coverage_thresholds:
        key = f"{int(round(coverage_threshold * 100))}pct"
        detected = sum(value >= coverage_threshold for value in coverage)
        pooled[f"events_covered_{key}"] = detected
        pooled[f"events_missed_{key}"] = total_references - detected
        pooled[f"event_recall_{key}"] = detected / total_references

    intervals = [
        row
        for item in internal
        for row in (
            interval_rows(item["_references"], "reference")
            + interval_rows(item["_predictions"], "prediction")
        )
    ]
    matches = [row for item in internal for row in item["_matches"]]
    return {
        "per_file": per_file,
        "pooled": pooled,
        "intervals": intervals,
        "matches": matches,
    }

