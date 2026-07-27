from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .config import save_json
from .metrics import evaluate_records
from .score import load_score_manifest
from .train import run_directory
from .utils import write_sha256_manifest


def _combination_key(components: Sequence[str]) -> str:
    return "+".join(components)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _score_frame(
    arrays: dict[str, np.ndarray],
    components: Sequence[str],
    combination_key: str,
    threshold: float,
    file_id: str,
) -> pd.DataFrame:
    fused = arrays[f"fused_{combination_key}"]
    data: dict[str, Any] = {
        "file_id": file_id,
        "timestamp_unix_seconds": arrays["timestamp"],
        "source_row_index_zero_based": arrays["row"],
        "window_endpoint_index": arrays["endpoint"],
        "y_true": arrays["y_true"],
    }
    for component in components:
        data[f"raw_{component}"] = arrays[f"raw_{component}"]
        data[f"z_{component}"] = arrays[f"z_{component}"]
    data["fused_score"] = fused
    data["threshold"] = threshold
    data["y_pred"] = (fused > threshold).astype(np.uint8)
    return pd.DataFrame(data)


def evaluate_saved_scores(
    config: dict[str, Any],
    model_name: str,
    seed: int,
    *,
    components: Sequence[str] | None = None,
    quantile: float | None = None,
    stage: str = "formal",
    export_timestamp_scores: bool = True,
) -> Path:
    run_dir = run_directory(config, stage, model_name, seed)
    scoring = load_score_manifest(run_dir)
    selected = tuple(components or scoring["components"])
    unknown = set(selected) - set(scoring["components"])
    if unknown:
        raise ValueError(f"Unavailable score components: {sorted(unknown)}")
    key = _combination_key(selected)
    q = float(
        quantile
        if quantile is not None
        else config["scoring"]["threshold_quantile"]
    )
    try:
        threshold_details = scoring["thresholds"][key][str(q)]
    except KeyError as error:
        raise ValueError(f"No saved threshold for {key} at q={q}") from error
    threshold = float(threshold_details["threshold"])
    score_dir = run_dir / "scores"
    records: list[dict[str, Any]] = []
    loaded_files: list[tuple[str, dict[str, np.ndarray]]] = []
    for item in scoring["test_files"]:
        with np.load(item["path"]) as stored:
            arrays = {name: stored[name] for name in stored.files}
        records.append(
            {
                "file_id": item["file_id"],
                "y_true": arrays["y_true"],
                "score": arrays[f"fused_{key}"],
                "timestamps": arrays["timestamp"],
            }
        )
        loaded_files.append((item["file_id"], arrays))
    metrics = evaluate_records(
        records,
        threshold,
        float(config["evaluation"]["sampling_interval_seconds"]),
        [float(value) for value in config["evaluation"]["coverage_thresholds"]],
    )
    output = (
        run_dir
        / "evaluation"
        / _slug(key)
        / f"q_{str(q).replace('.', '_')}"
    )
    if output.exists():
        raise FileExistsError(f"Evaluation already exists: {output}")
    output.mkdir(parents=True)
    result = {
        "schema_version": 1,
        "model": model_name,
        "seed": seed,
        "components": list(selected),
        "quantile": q,
        "threshold_details": threshold_details,
        "pooled": metrics["pooled"],
        "metric_definitions": {
            "average_precision": "non-interpolated AP with tied-score grouping",
            "overlap_event": "maximum-cardinality one-to-one interval overlap matching",
            "duration_precision": "alarm seconds overlapping anomalies / all alarm seconds",
            "duration_recall": "anomalous seconds covered by alarms / all anomalous seconds",
            "event_coverage": "fraction of each reference interval covered by any alarm",
            "false_alarm_rate_denominator": "scored normal hours",
        },
    }
    save_json(output / "metrics.json", result)
    _write_csv(output / "per_file_metrics.csv", metrics["per_file"])
    _write_csv(output / "intervals.csv", metrics["intervals"])
    _write_csv(output / "event_matches.csv", metrics["matches"])

    if export_timestamp_scores:
        for file_id, arrays in loaded_files:
            frame = _score_frame(arrays, selected, key, threshold, file_id)
            frame.to_csv(
                output / f"timestamp_scores_{file_id}.csv.gz",
                index=False,
                compression="gzip",
            )
        with np.load(score_dir / "calibration_scores.npz") as stored:
            calibration = {name: stored[name] for name in stored.files}
        calibration_frame = _score_frame(
            calibration, selected, key, threshold, "train3_calibration"
        )
        calibration_frame.to_csv(
            output / "timestamp_scores_calibration.csv.gz",
            index=False,
            compression="gzip",
        )
    write_sha256_manifest(run_dir)
    return output

