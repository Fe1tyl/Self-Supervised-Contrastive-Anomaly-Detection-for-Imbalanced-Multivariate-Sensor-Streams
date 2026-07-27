from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import save_json
from .utils import write_sha256_manifest


SUMMARY_METRICS = [
    "macro_average_precision",
    "auroc",
    "point_f1",
    "overlap_event_f1",
    "duration_precision",
    "duration_recall",
    "duration_f1",
    "event_recall_10pct",
    "event_recall_50pct",
    "false_alarm_events_per_normal_hour",
    "median_detection_delay_seconds",
]


def aggregate_completed_runs(config: dict[str, Any]) -> Path:
    artifacts = Path(config["artifacts_dir"])
    formal = artifacts / "formal"
    output = artifacts / "summary"
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    file_rows: list[dict[str, Any]] = []
    for metrics_path in sorted(formal.glob("*/*/evaluation/*/q_*/metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        run_dir = metrics_path.parents[3]
        status_path = run_dir / "status.json"
        status = (
            json.loads(status_path.read_text(encoding="utf-8"))
            if status_path.exists()
            else {}
        )
        pooled = metrics["pooled"]
        row = {
            "model": metrics["model"],
            "seed": metrics["seed"],
            "components": "+".join(metrics["components"]),
            "quantile": metrics["quantile"],
            "best_epoch": status.get("best_epoch"),
            **pooled,
            "metrics_path": str(metrics_path),
        }
        rows.append(row)
        per_file_path = metrics_path.with_name("per_file_metrics.csv")
        if per_file_path.exists():
            frame = pd.read_csv(per_file_path)
            frame.insert(0, "quantile", metrics["quantile"])
            frame.insert(0, "components", "+".join(metrics["components"]))
            frame.insert(0, "seed", metrics["seed"])
            frame.insert(0, "model", metrics["model"])
            file_rows.extend(frame.to_dict(orient="records"))
    if not rows:
        raise FileNotFoundError("No completed evaluation metrics were found")
    seed_frame = pd.DataFrame(rows)
    seed_frame.to_csv(output / "seed_level_metrics.csv", index=False)
    if file_rows:
        pd.DataFrame(file_rows).to_csv(
            output / "seed_file_metrics.csv", index=False
        )

    summaries = []
    group_columns = ["model", "components", "quantile"]
    for keys, group in seed_frame.groupby(group_columns, dropna=False):
        summary: dict[str, Any] = dict(zip(group_columns, keys, strict=True))
        summary["completed_seeds"] = int(group["seed"].nunique())
        summary["seed_values"] = ",".join(str(value) for value in sorted(group["seed"]))
        for metric in SUMMARY_METRICS:
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy()
            summary[f"{metric}_mean"] = float(np.mean(values)) if len(values) else np.nan
            summary[f"{metric}_sample_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
            )
        summaries.append(summary)
    pd.DataFrame(summaries).to_csv(output / "model_summary.csv", index=False)

    failures = []
    failed_root = artifacts / "failed"
    if failed_root.exists():
        for path in sorted(failed_root.glob("*/*/status.json")):
            status = json.loads(path.read_text(encoding="utf-8"))
            failures.append({**status, "status_path": str(path)})
    save_json(
        output / "summary_manifest.json",
        {
            "completed_evaluations": len(rows),
            "completed_models": sorted(seed_frame["model"].unique().tolist()),
            "completed_seeds_by_model": {
                model: sorted(
                    int(value)
                    for value in seed_frame.loc[
                        seed_frame["model"] == model, "seed"
                    ].unique()
                )
                for model in seed_frame["model"].unique()
            },
            "failed_attempts": failures,
            "warning": (
                "Summary rows are descriptive until all preregistered seeds and "
                "configurations are present."
            ),
        },
    )
    write_sha256_manifest(output)
    return output

