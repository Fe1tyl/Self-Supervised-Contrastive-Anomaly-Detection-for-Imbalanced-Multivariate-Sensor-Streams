from __future__ import annotations

import itertools
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import save_json
from .data import WindowDataset, load_data_manifest, segments_for_split
from .model import build_model, scores_for
from .train import run_directory
from .utils import finite_or_raise


def _autocast(config: dict[str, Any], device: torch.device):
    enabled = bool(config["training"]["mixed_precision"]) and device.type == "cuda"
    if not enabled:
        return nullcontext()
    dtype_name = config["training"].get("mixed_precision_dtype", "float16")
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _loader(
    dataset: WindowDataset,
    batch_size: int,
    config: dict[str, Any],
) -> DataLoader:
    workers = int(config["training"]["num_workers"])
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def _load_model(
    config: dict[str, Any],
    model_name: str,
    seed: int,
    stage: str,
    device: torch.device,
) -> tuple[torch.nn.Module, Path]:
    run_dir = run_directory(config, stage, model_name, seed)
    checkpoint_path = run_dir / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint["model_name"] != model_name or int(checkpoint["seed"]) != seed:
        raise ValueError("Checkpoint identity does not match requested run")
    model = build_model(config, 79).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, run_dir


def _prototype(
    model: torch.nn.Module,
    dataset: WindowDataset,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float]]:
    loader = _loader(
        dataset, int(config["training"]["evaluation_batch_size"]), config
    )
    embedding_sum: torch.Tensor | None = None
    observations = 0
    diagnostic_samples: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            x = batch.to(device, non_blocking=True)
            with _autocast(config, device):
                hidden = model.encode_sequence(x)
                embeddings = model.embed(hidden)
            embeddings_float = embeddings.float()
            batch_sum = embeddings_float.sum(dim=0).cpu()
            embedding_sum = (
                batch_sum if embedding_sum is None else embedding_sum + batch_sum
            )
            observations += len(x)
            if sum(len(item) for item in diagnostic_samples) < 4_096:
                diagnostic_samples.append(embeddings_float.cpu().numpy())
    assert embedding_sum is not None
    prototype = embedding_sum.numpy().astype(np.float64) / observations
    prototype /= np.linalg.norm(prototype)
    sample = np.concatenate(diagnostic_samples, axis=0)[:4_096].astype(np.float64)
    dimension_std = sample.std(axis=0)
    covariance = np.cov(sample, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    normalized_eigenvalues = eigenvalues / max(eigenvalues.sum(), 1e-12)
    effective_rank = float(
        np.exp(
            -np.sum(
                normalized_eigenvalues
                * np.log(np.maximum(normalized_eigenvalues, 1e-12))
            )
        )
    )
    pair_sample = sample[: min(1_024, len(sample))]
    cosine = pair_sample @ pair_sample.T
    mean_off_diagonal = float(
        (cosine.sum() - np.trace(cosine))
        / max(1, len(pair_sample) * (len(pair_sample) - 1))
    )
    diagnostics = {
        "training_embedding_count": observations,
        "diagnostic_sample_count": len(sample),
        "dimension_std_min": float(dimension_std.min()),
        "dimension_std_mean": float(dimension_std.mean()),
        "dimension_std_max": float(dimension_std.max()),
        "effective_rank": effective_rank,
        "mean_pairwise_cosine": mean_off_diagonal,
    }
    return prototype.astype(np.float32), diagnostics


def _score_dataset(
    model: torch.nn.Module,
    dataset: WindowDataset,
    components: Sequence[str],
    prototype: torch.Tensor | None,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, np.ndarray]:
    loader = _loader(
        dataset, int(config["training"]["evaluation_batch_size"]), config
    )
    values: dict[str, list[np.ndarray]] = {
        "y_true": [],
        "timestamp": [],
        "row": [],
        "endpoint": [],
    }
    for component in components:
        values[f"raw_{component}"] = []
    with torch.inference_mode():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            with _autocast(config, device):
                outputs = model(x)
            if "representation" in components:
                assert prototype is not None
                score = 1.0 - (outputs["embedding"].float() * prototype).sum(dim=1)
                values["raw_representation"].append(score.cpu().numpy())
            if "reconstruction" in components:
                score = (
                    outputs["reconstruction"][:, -1].float() - x[:, -1].float()
                ).abs().mean(dim=1)
                values["raw_reconstruction"].append(score.cpu().numpy())
            if "prediction" in components:
                score = (
                    outputs["prediction"][:, -1].float() - x[:, -1].float()
                ).abs().mean(dim=1)
                values["raw_prediction"].append(score.cpu().numpy())
            values["y_true"].append(batch["y"].numpy())
            values["timestamp"].append(batch["time"].numpy())
            values["row"].append(batch["row"].numpy())
            values["endpoint"].append(batch["endpoint"].numpy())
    result = {name: np.concatenate(chunks) for name, chunks in values.items()}
    finite_or_raise(
        "raw_scores",
        [result[f"raw_{component}"] for component in components],
    )
    return result


def _component_combinations(
    components: Sequence[str],
) -> list[tuple[str, ...]]:
    return [
        combination
        for size in range(1, len(components) + 1)
        for combination in itertools.combinations(components, size)
    ]


def _combination_key(components: Sequence[str]) -> str:
    return "+".join(components)


def _add_standardized_and_fused(
    arrays: dict[str, np.ndarray],
    components: Sequence[str],
    statistics: dict[str, Any],
) -> None:
    for component in components:
        raw = arrays[f"raw_{component}"].astype(np.float64)
        median = float(statistics[component]["median"])
        denominator = float(statistics[component]["scaled_mad_plus_epsilon"])
        arrays[f"z_{component}"] = np.maximum(0.0, (raw - median) / denominator).astype(
            np.float32
        )
    for combination in _component_combinations(components):
        key = _combination_key(combination)
        arrays[f"fused_{key}"] = np.mean(
            np.stack([arrays[f"z_{name}"] for name in combination], axis=0),
            axis=0,
            dtype=np.float64,
        ).astype(np.float32)


def score_model(
    config: dict[str, Any],
    model_name: str,
    seed: int,
    stage: str = "formal",
) -> Path:
    manifest = load_data_manifest(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, run_dir = _load_model(config, model_name, seed, stage, device)
    score_dir = run_dir / "scores"
    if score_dir.exists():
        raise FileExistsError(f"Scores already exist: {score_dir}")
    score_dir.mkdir(parents=True)
    components = scores_for(model_name)
    length = int(config["window"]["length"])
    stride = int(config["window"]["evaluation_stride"])

    prototype_array: np.ndarray | None = None
    prototype_tensor: torch.Tensor | None = None
    embedding_diagnostics: dict[str, Any] | None = None
    if "representation" in components:
        fit_dataset = WindowDataset(
            segments_for_split(manifest, "fit"),
            length,
            int(config["window"]["fit_stride"]),
        )
        prototype_array, embedding_diagnostics = _prototype(
            model, fit_dataset, config, device
        )
        np.save(score_dir / "normal_prototype.npy", prototype_array)
        prototype_tensor = torch.from_numpy(prototype_array).to(device)

    calibration_dataset = WindowDataset(
        segments_for_split(manifest, "calibration"),
        length,
        stride,
        include_metadata=True,
    )
    calibration = _score_dataset(
        model,
        calibration_dataset,
        components,
        prototype_tensor,
        config,
        device,
    )
    mad_scale = float(config["scoring"]["mad_scale"])
    epsilon = float(config["scoring"]["epsilon"])
    statistics: dict[str, Any] = {}
    for component in components:
        raw = calibration[f"raw_{component}"].astype(np.float64)
        median = float(np.median(raw))
        mad = float(np.median(np.abs(raw - median)))
        if not np.isfinite(mad) or mad <= 0:
            raise ValueError(f"Calibration MAD is non-positive for {component}: {mad}")
        statistics[component] = {
            "median": median,
            "mad": mad,
            "scaled_mad": mad_scale * mad,
            "scaled_mad_plus_epsilon": mad_scale * mad + epsilon,
            "count": len(raw),
        }
    _add_standardized_and_fused(calibration, components, statistics)

    threshold_grid = [float(q) for q in config["scoring"]["threshold_grid"]]
    method = config["scoring"]["quantile_method"]
    thresholds: dict[str, Any] = {}
    for combination in _component_combinations(components):
        key = _combination_key(combination)
        fused = calibration[f"fused_{key}"].astype(np.float64)
        thresholds[key] = {}
        for quantile in threshold_grid:
            threshold = float(np.quantile(fused, quantile, method=method))
            alerts = fused > threshold
            ties = fused == threshold
            thresholds[key][str(quantile)] = {
                "threshold": threshold,
                "calibration_alert_count": int(alerts.sum()),
                "calibration_alert_rate": float(alerts.mean()),
                "threshold_tie_count": int(ties.sum()),
                "threshold_tie_rate": float(ties.mean()),
                "quantile_method": method,
                "decision_rule": "score > threshold",
            }
    np.savez_compressed(score_dir / "calibration_scores.npz", **calibration)

    test_files: list[dict[str, Any]] = []
    for segment in segments_for_split(manifest, "test"):
        dataset = WindowDataset(
            [segment], length, stride, include_metadata=True
        )
        arrays = _score_dataset(
            model, dataset, components, prototype_tensor, config, device
        )
        _add_standardized_and_fused(arrays, components, statistics)
        output_path = score_dir / f"{segment['name']}_scores.npz"
        np.savez_compressed(output_path, **arrays)
        test_files.append(
            {
                "file_id": segment["name"],
                "path": str(output_path),
                "scored_points": len(arrays["y_true"]),
                "anomaly_points": int(arrays["y_true"].sum()),
            }
        )
    if sum(item["scored_points"] for item in test_files) != 401_370:
        raise ValueError("Test score count does not equal 401370")
    scoring_manifest = {
        "schema_version": 1,
        "model": model_name,
        "seed": seed,
        "checkpoint": str(run_dir / "best.pt"),
        "components": list(components),
        "default_combination": _combination_key(components),
        "calibration_score_count": len(calibration["y_true"]),
        "component_calibration_statistics": statistics,
        "thresholds": thresholds,
        "embedding_diagnostics": embedding_diagnostics,
        "test_files": test_files,
        "test_score_count": sum(item["scored_points"] for item in test_files),
        "test_labels_used_for_prototype_scaling_or_threshold": 0,
    }
    save_json(score_dir / "scoring_manifest.json", scoring_manifest)
    return score_dir


def load_score_manifest(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / "scores" / "scoring_manifest.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))
