from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import save_json
from .data import load_data_manifest, segments_for_split
from .train import run_directory
from .utils import seed_everything, software_device_manifest


def _endpoints(rows: int, length: int, stride: int) -> np.ndarray:
    return np.arange(length - 1, rows, stride, dtype=np.int64)


def _persistence_residuals(
    segment: dict[str, Any], length: int, stride: int
) -> tuple[np.ndarray, np.ndarray]:
    values = np.load(segment["x"], mmap_mode="r")
    endpoints = _endpoints(len(values), length, stride)
    residuals = np.abs(
        np.asarray(values[endpoints], dtype=np.float64)
        - np.asarray(values[endpoints - 1], dtype=np.float64)
    )
    return endpoints, residuals


def _top_five_score(
    residuals: np.ndarray,
    median: np.ndarray,
    denominator: np.ndarray,
) -> np.ndarray:
    standardized = np.maximum(0.0, (residuals - median) / denominator)
    top_five = np.partition(standardized, -5, axis=1)[:, -5:]
    return top_five.mean(axis=1, dtype=np.float64).astype(np.float32)


def _metadata_arrays(
    segment: dict[str, Any], endpoints: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        "y_true": np.asarray(np.load(segment["y"], mmap_mode="r")[endpoints]),
        "timestamp": np.asarray(
            np.load(segment["time"], mmap_mode="r")[endpoints]
        ),
        "row": np.asarray(np.load(segment["row"], mmap_mode="r")[endpoints]),
        "endpoint": endpoints,
    }


def _window_summary_features(
    values: np.ndarray,
    length: int,
    stride: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    endpoints = _endpoints(len(values), length, stride)
    windows = np.lib.stride_tricks.sliding_window_view(
        values, window_shape=length, axis=0
    )[::stride]
    if len(windows) != len(endpoints):
        raise ValueError(
            f"Window feature alignment mismatch: {len(windows)} != {len(endpoints)}"
        )
    feature_count = int(values.shape[1])
    output = np.empty((len(endpoints), feature_count * 6), dtype=np.float32)
    for start in range(0, len(windows), batch_size):
        stop = min(start + batch_size, len(windows))
        block = np.asarray(windows[start:stop], dtype=np.float32)
        mean = block.mean(axis=2, dtype=np.float32)
        second_moment = np.square(block).mean(axis=2, dtype=np.float32)
        std = np.sqrt(np.maximum(second_moment - np.square(mean), 0.0))
        minimum = block.min(axis=2)
        maximum = block.max(axis=2)
        batch_endpoints = endpoints[start:stop]
        last = np.asarray(values[batch_endpoints], dtype=np.float32)
        first = np.asarray(values[batch_endpoints - length + 1], dtype=np.float32)
        output[start:stop] = np.concatenate(
            (mean, std, minimum, maximum, last, last - first), axis=1
        )
    if not np.isfinite(output).all():
        raise FloatingPointError("Isolation Forest features contain NaN or infinity")
    return endpoints, output


def _isolation_forest_score(estimator: Any, features: np.ndarray) -> np.ndarray:
    return (-estimator.score_samples(features)).astype(np.float32)


def run_isolation_forest_baseline(
    config: dict[str, Any],
    seed: int,
    stage: str = "formal",
) -> Path:
    try:
        import joblib
        from sklearn.ensemble import IsolationForest
    except ImportError as error:
        raise RuntimeError(
            "Isolation Forest requires the optional scikit-learn dependency"
        ) from error

    parameters = config["baselines"]["isolation_forest"]
    expected_features = ["mean", "std", "min", "max", "last", "delta"]
    if parameters["window_features"] != expected_features:
        raise ValueError(
            "Isolation Forest window_features must match the frozen feature order"
        )
    seed_everything(seed)
    manifest = load_data_manifest(config)
    run_dir = run_directory(config, stage, "isolation_forest", seed)
    score_dir = run_dir / "scores"
    if run_dir.exists():
        raise FileExistsError(f"Isolation Forest run already exists: {run_dir}")
    score_dir.mkdir(parents=True)
    save_json(run_dir / "resolved_config.json", config)
    save_json(run_dir / "environment.json", software_device_manifest())

    length = int(config["window"]["length"])
    fit_stride = int(config["window"]["fit_stride"])
    evaluation_stride = int(config["window"]["evaluation_stride"])
    feature_batch_size = int(parameters["window_feature_batch_size"])

    fitting_parts = []
    for segment in segments_for_split(manifest, "fit"):
        values = np.load(segment["x"], mmap_mode="r")
        _, features = _window_summary_features(
            values, length, fit_stride, feature_batch_size
        )
        fitting_parts.append(features)
    fitting = np.concatenate(fitting_parts, axis=0)
    del fitting_parts
    if fitting.shape != (165_093, 79 * 6):
        raise ValueError(
            f"Expected fitting feature shape (165093, 474), got {fitting.shape}"
        )

    estimator = IsolationForest(
        n_estimators=int(parameters["n_estimators"]),
        max_samples=int(parameters["max_samples"]),
        contamination=parameters["contamination"],
        n_jobs=int(parameters["n_jobs"]),
        random_state=seed,
    )
    estimator.fit(fitting)
    model_path = run_dir / "isolation_forest.joblib"
    joblib.dump(estimator, model_path, compress=3)
    fit_score = _isolation_forest_score(estimator, fitting)
    save_json(
        run_dir / "run_manifest.json",
        {
            "model": "isolation_forest",
            "seed": seed,
            "stage": stage,
            "fit_windows": len(fitting),
            "feature_dimension": int(fitting.shape[1]),
            "feature_order": expected_features,
            "estimator_parameters": {
                "n_estimators": int(parameters["n_estimators"]),
                "max_samples": int(parameters["max_samples"]),
                "contamination": parameters["contamination"],
                "n_jobs": int(parameters["n_jobs"]),
                "random_state": seed,
            },
            "test_labels_used_for_training_or_selection": 0,
            "fitting_score_summary": {
                "minimum": float(fit_score.min()),
                "median": float(np.median(fit_score)),
                "maximum": float(fit_score.max()),
            },
        },
    )
    del fitting, fit_score

    calibration_segment = segments_for_split(manifest, "calibration")[0]
    calibration_values = np.load(calibration_segment["x"], mmap_mode="r")
    calibration_endpoints, calibration_features = _window_summary_features(
        calibration_values, length, evaluation_stride, feature_batch_size
    )
    calibration_score = _isolation_forest_score(estimator, calibration_features)
    del calibration_features
    calibration = {
        **_metadata_arrays(calibration_segment, calibration_endpoints),
        "raw_isolation_forest": calibration_score,
        "z_isolation_forest": calibration_score,
        "fused_isolation_forest": calibration_score,
    }
    np.savez_compressed(score_dir / "calibration_scores.npz", **calibration)

    threshold_grid = [float(q) for q in config["scoring"]["threshold_grid"]]
    method = config["scoring"]["quantile_method"]
    thresholds: dict[str, Any] = {"isolation_forest": {}}
    for quantile in threshold_grid:
        threshold = float(np.quantile(calibration_score, quantile, method=method))
        alerts = calibration_score > threshold
        ties = calibration_score == threshold
        thresholds["isolation_forest"][str(quantile)] = {
            "threshold": threshold,
            "calibration_alert_count": int(alerts.sum()),
            "calibration_alert_rate": float(alerts.mean()),
            "threshold_tie_count": int(ties.sum()),
            "threshold_tie_rate": float(ties.mean()),
            "quantile_method": method,
            "decision_rule": "score > threshold",
        }

    test_files = []
    for segment in segments_for_split(manifest, "test"):
        values = np.load(segment["x"], mmap_mode="r")
        endpoints, features = _window_summary_features(
            values, length, evaluation_stride, feature_batch_size
        )
        score = _isolation_forest_score(estimator, features)
        del features
        arrays = {
            **_metadata_arrays(segment, endpoints),
            "raw_isolation_forest": score,
            "z_isolation_forest": score,
            "fused_isolation_forest": score,
        }
        output_path = score_dir / f"{segment['name']}_scores.npz"
        np.savez_compressed(output_path, **arrays)
        test_files.append(
            {
                "file_id": segment["name"],
                "path": str(output_path),
                "scored_points": len(score),
                "anomaly_points": int(arrays["y_true"].sum()),
            }
        )

    scoring_manifest = {
        "schema_version": 1,
        "model": "isolation_forest",
        "seed": seed,
        "deterministic_baseline": False,
        "components": ["isolation_forest"],
        "default_combination": "isolation_forest",
        "calibration_score_count": len(calibration_score),
        "component_calibration_statistics": {
            "isolation_forest": {
                "minimum": float(calibration_score.min()),
                "median": float(np.median(calibration_score)),
                "maximum": float(calibration_score.max()),
            }
        },
        "thresholds": thresholds,
        "embedding_diagnostics": None,
        "test_files": test_files,
        "test_score_count": sum(item["scored_points"] for item in test_files),
        "test_labels_used_for_scaling_or_threshold": 0,
    }
    save_json(score_dir / "scoring_manifest.json", scoring_manifest)
    save_json(
        run_dir / "status.json",
        {
            "state": "complete",
            "model": "isolation_forest",
            "seed": seed,
            "fit_windows": 165_093,
            "calibration_scores": len(calibration_score),
            "test_scores": scoring_manifest["test_score_count"],
        },
    )
    return run_dir


def run_persistence_baseline(
    config: dict[str, Any],
    seed_label: int = 17,
    stage: str = "formal",
) -> Path:
    manifest = load_data_manifest(config)
    run_dir = run_directory(config, stage, "persistence", seed_label)
    score_dir = run_dir / "scores"
    if run_dir.exists():
        raise FileExistsError(f"Persistence run already exists: {run_dir}")
    score_dir.mkdir(parents=True)
    save_json(run_dir / "resolved_config.json", config)
    save_json(run_dir / "environment.json", software_device_manifest())
    length = int(config["window"]["length"])
    fit_stride = int(config["window"]["fit_stride"])
    evaluation_stride = int(config["window"]["evaluation_stride"])

    fitting_residuals = []
    for segment in segments_for_split(manifest, "fit"):
        _, residuals = _persistence_residuals(segment, length, fit_stride)
        fitting_residuals.append(residuals)
    fitting = np.concatenate(fitting_residuals, axis=0)
    if len(fitting) != 165_093:
        raise ValueError(f"Expected 165093 fitting residuals, got {len(fitting)}")
    median = np.median(fitting, axis=0)
    mad = np.median(np.abs(fitting - median), axis=0)
    denominator = (
        float(config["scoring"]["mad_scale"]) * mad
        + float(config["scoring"]["epsilon"])
    )
    channel_stats = {
        "median": median.tolist(),
        "mad": mad.tolist(),
        "zero_mad_feature_count": int(np.sum(mad == 0)),
        "zero_mad_features": np.asarray(manifest["feature_names"])[mad == 0].tolist(),
        "fitting_residual_count": len(fitting),
        "score_definition": "mean of five largest nonnegative robust-scaled channel residuals",
    }
    save_json(run_dir / "persistence_channel_statistics.json", channel_stats)

    calibration_segment = segments_for_split(manifest, "calibration")[0]
    calibration_endpoints, calibration_residuals = _persistence_residuals(
        calibration_segment, length, evaluation_stride
    )
    calibration_score = _top_five_score(
        calibration_residuals, median, denominator
    )
    calibration = {
        **_metadata_arrays(calibration_segment, calibration_endpoints),
        "raw_persistence": calibration_score,
        "z_persistence": calibration_score,
        "fused_persistence": calibration_score,
    }
    np.savez_compressed(score_dir / "calibration_scores.npz", **calibration)
    threshold_grid = [float(q) for q in config["scoring"]["threshold_grid"]]
    method = config["scoring"]["quantile_method"]
    thresholds: dict[str, Any] = {"persistence": {}}
    for quantile in threshold_grid:
        threshold = float(np.quantile(calibration_score, quantile, method=method))
        alerts = calibration_score > threshold
        ties = calibration_score == threshold
        thresholds["persistence"][str(quantile)] = {
            "threshold": threshold,
            "calibration_alert_count": int(alerts.sum()),
            "calibration_alert_rate": float(alerts.mean()),
            "threshold_tie_count": int(ties.sum()),
            "threshold_tie_rate": float(ties.mean()),
            "quantile_method": method,
            "decision_rule": "score > threshold",
        }

    test_files = []
    for segment in segments_for_split(manifest, "test"):
        endpoints, residuals = _persistence_residuals(
            segment, length, evaluation_stride
        )
        score = _top_five_score(residuals, median, denominator)
        arrays = {
            **_metadata_arrays(segment, endpoints),
            "raw_persistence": score,
            "z_persistence": score,
            "fused_persistence": score,
        }
        output_path = score_dir / f"{segment['name']}_scores.npz"
        np.savez_compressed(output_path, **arrays)
        test_files.append(
            {
                "file_id": segment["name"],
                "path": str(output_path),
                "scored_points": len(score),
                "anomaly_points": int(arrays["y_true"].sum()),
            }
        )
    scoring_manifest = {
        "schema_version": 1,
        "model": "persistence",
        "seed": seed_label,
        "deterministic_baseline": True,
        "components": ["persistence"],
        "default_combination": "persistence",
        "calibration_score_count": len(calibration_score),
        "component_calibration_statistics": {
            "persistence": {
                "fitting_channel_statistics": str(
                    run_dir / "persistence_channel_statistics.json"
                )
            }
        },
        "thresholds": thresholds,
        "embedding_diagnostics": None,
        "test_files": test_files,
        "test_score_count": sum(item["scored_points"] for item in test_files),
        "test_labels_used_for_scaling_or_threshold": 0,
    }
    save_json(score_dir / "scoring_manifest.json", scoring_manifest)
    save_json(
        run_dir / "status.json",
        {
            "state": "complete",
            "model": "persistence",
            "deterministic": True,
            "fit_residuals": len(fitting),
            "calibration_scores": len(calibration_score),
            "test_scores": scoring_manifest["test_score_count"],
        },
    )
    return run_dir
