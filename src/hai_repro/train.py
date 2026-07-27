from __future__ import annotations

import csv
import json
import math
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from .config import save_json
from .data import (
    WindowDataset,
    load_data_manifest,
    segments_for_split,
)
from .losses import compute_objective
from .model import (
    MultiObjectiveTCN,
    build_model,
    parameter_counts,
    trainable_parameters,
)
from .utils import finite_or_raise, seed_everything, software_device_manifest


HISTORY_FIELDS = [
    "epoch",
    "train_total",
    "train_raw_contrastive",
    "train_raw_reconstruction",
    "train_raw_prediction",
    "train_weighted_contrastive",
    "train_weighted_reconstruction",
    "train_weighted_prediction",
    "calibration_total",
    "calibration_raw_contrastive",
    "calibration_raw_reconstruction",
    "calibration_raw_prediction",
    "calibration_weighted_contrastive",
    "calibration_weighted_reconstruction",
    "calibration_weighted_prediction",
    "learning_rate",
    "mean_gradient_norm",
    "epoch_seconds",
    "peak_gpu_memory_bytes",
]


def run_directory(
    config: dict[str, Any], stage: str, model_name: str, seed: int
) -> Path:
    return (
        Path(config["artifacts_dir"])
        / stage
        / model_name
        / f"seed_{seed}"
    )


def _loader(
    dataset: WindowDataset,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    seed: int,
    config: dict[str, Any],
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    workers = int(config["training"]["num_workers"])
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        generator=generator,
    )


def _autocast(config: dict[str, Any], device: torch.device):
    enabled = bool(config["training"]["mixed_precision"]) and device.type == "cuda"
    if not enabled:
        return nullcontext()
    dtype_name = config["training"].get("mixed_precision_dtype", "float16")
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _loss_accumulator() -> dict[str, float]:
    return {
        "total": 0.0,
        "raw_contrastive": 0.0,
        "raw_reconstruction": 0.0,
        "raw_prediction": 0.0,
        "weighted_contrastive": 0.0,
        "weighted_reconstruction": 0.0,
        "weighted_prediction": 0.0,
    }


def _accumulate(
    totals: dict[str, float], values: dict[str, torch.Tensor], batch_size: int
) -> None:
    for name in totals:
        totals[name] += float(values[name].detach().float().item()) * batch_size


def _averages(totals: dict[str, float], observations: int) -> dict[str, float]:
    return {name: value / observations for name, value in totals.items()}


def _augmentation_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device.type)
    generator.manual_seed(seed)
    return generator


def evaluate_loss(
    model: MultiObjectiveTCN,
    loader: DataLoader,
    model_name: str,
    config: dict[str, Any],
    continuous_mask: torch.Tensor,
    device: torch.device,
    augmentation_seed: int,
) -> dict[str, float]:
    model.eval()
    totals = _loss_accumulator()
    observations = 0
    generator = _augmentation_generator(device, augmentation_seed)
    with torch.inference_mode():
        for batch in loader:
            x = batch.to(device, non_blocking=True)
            with _autocast(config, device):
                _, values = compute_objective(
                    model,
                    x,
                    model_name,
                    config,
                    continuous_mask,
                    generator,
                )
            _accumulate(totals, values, len(x))
            observations += len(x)
    averages = _averages(totals, observations)
    finite_or_raise(
        "calibration_loss",
        [np.asarray(list(averages.values()), dtype=np.float64)],
    )
    return averages


def _scheduler(
    optimizer: AdamW, optimizer_steps: int, warmup_fraction: float
) -> LambdaLR:
    warmup_steps = max(1, int(round(optimizer_steps * warmup_fraction)))

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, optimizer_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return LambdaLR(optimizer, multiplier)


def _write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint_payload(
    model: MultiObjectiveTCN,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_epoch: int,
    best_calibration_loss: float,
    epochs_without_improvement: int,
    history: list[dict[str, Any]],
    model_name: str,
    seed: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model_name": model_name,
        "seed": seed,
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_calibration_loss": best_calibration_loss,
        "epochs_without_improvement": epochs_without_improvement,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "history": history,
        "model_config": config["model"],
        "feature_count": 79,
    }


def train_model(
    config: dict[str, Any],
    model_name: str,
    seed: int,
    *,
    maximum_epochs: int | None = None,
    stage: str = "formal",
    resume: bool = False,
) -> Path:
    seed_everything(seed)
    manifest = load_data_manifest(config)
    if len(manifest["feature_names"]) != 79:
        raise ValueError("Prepared data do not contain 79 process variables")
    destination = run_directory(config, stage, model_name, seed)
    status_path = destination / "status.json"
    if destination.exists() and not resume:
        if status_path.exists():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("state") == "complete":
                raise FileExistsError(f"Completed run already exists: {destination}")
        raise FileExistsError(
            f"Run directory already exists; use --resume after inspection: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    save_json(destination / "resolved_config.json", config)
    save_json(destination / "environment.json", software_device_manifest())
    save_json(
        status_path,
        {
            "state": "running",
            "stage": stage,
            "model": model_name,
            "seed": seed,
            "started_unix": time.time(),
        },
    )

    length = int(config["window"]["length"])
    fit_dataset = WindowDataset(
        segments_for_split(manifest, "fit"),
        length,
        int(config["window"]["fit_stride"]),
    )
    calibration_dataset = WindowDataset(
        segments_for_split(manifest, "calibration"),
        length,
        int(config["window"]["evaluation_stride"]),
    )
    if len(fit_dataset) != 165_093 or len(calibration_dataset) != 95_634:
        raise ValueError("Unexpected fitting/calibration window counts")
    micro_batch = int(config["training"]["micro_batch_size"])
    effective_batch = int(config["training"]["effective_batch_size"])
    if effective_batch % micro_batch:
        raise ValueError("effective_batch_size must be divisible by micro_batch_size")
    accumulation_steps = effective_batch // micro_batch
    train_loader = _loader(
        fit_dataset, micro_batch, True, True, seed, config
    )
    calibration_loader = _loader(
        calibration_dataset,
        int(config["training"]["evaluation_batch_size"]),
        False,
        False,
        seed,
        config,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, len(manifest["feature_names"])).to(device)
    parameters = trainable_parameters(model, model_name)
    counts = parameter_counts(model)
    save_json(
        destination / "run_manifest.json",
        {
            "model": model_name,
            "seed": seed,
            "stage": stage,
            "fit_windows": len(fit_dataset),
            "calibration_windows": len(calibration_dataset),
            "micro_batch_size": micro_batch,
            "effective_batch_size": effective_batch,
            "gradient_accumulation_steps": accumulation_steps,
            "checkpoint_criterion": "minimum normal-calibration active total loss",
            "test_labels_used_for_training_or_selection": 0,
            "parameter_counts": counts,
        },
    )
    continuous_mask = torch.tensor(
        np.logical_not(manifest["normalization"]["binary_mask"]),
        dtype=torch.bool,
        device=device,
    )
    optimizer = AdamW(
        parameters,
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    epochs = int(
        maximum_epochs
        if maximum_epochs is not None
        else config["training"]["maximum_epochs"]
    )
    steps_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    scheduler = _scheduler(
        optimizer,
        epochs * steps_per_epoch,
        float(config["training"]["warmup_fraction"]),
    )
    amp_enabled = (
        bool(config["training"]["mixed_precision"]) and device.type == "cuda"
    )
    scaler_enabled = amp_enabled and (
        config["training"].get("mixed_precision_dtype", "float16") == "float16"
    )
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)

    start_epoch = 1
    best_epoch = 0
    best_calibration_loss = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    last_path = destination / "last.pt"
    if resume:
        if not last_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {last_path}")
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_epoch = int(checkpoint["best_epoch"])
        best_calibration_loss = float(checkpoint["best_calibration_loss"])
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        history = checkpoint["history"]

    best_path = destination / "best.pt"
    start_time = time.time()
    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        totals = _loss_accumulator()
        observations = 0
        gradient_norms: list[float] = []
        optimizer.zero_grad(set_to_none=True)
        augmentation_generator = _augmentation_generator(
            device, seed * 1_000_003 + epoch
        )
        for batch_index, batch in enumerate(train_loader, start=1):
            x = batch.to(device, non_blocking=True)
            with _autocast(config, device):
                loss, values = compute_objective(
                    model,
                    x,
                    model_name,
                    config,
                    continuous_mask,
                    augmentation_generator,
                )
            finite_or_raise("training_loss", [loss])
            scaler.scale(loss / accumulation_steps).backward()
            _accumulate(totals, values, len(x))
            observations += len(x)
            should_step = (
                batch_index % accumulation_steps == 0
                or batch_index == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                gradient_norm = clip_grad_norm_(
                    parameters,
                    float(config["training"]["gradient_clip_norm"]),
                )
                if not torch.isfinite(gradient_norm):
                    save_json(
                        status_path,
                        {
                            "state": "failed",
                            "stage": stage,
                            "model": model_name,
                            "seed": seed,
                            "failed_epoch": epoch,
                            "failed_batch": batch_index,
                            "reason": "non-finite gradient norm",
                            "last_safe_epoch": epoch - 1,
                            "mixed_precision_dtype": config["training"].get(
                                "mixed_precision_dtype", "float16"
                            ),
                        },
                    )
                    raise FloatingPointError("gradient_norm contains NaN or infinity")
                gradient_norms.append(float(gradient_norm.detach().item()))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
        train_averages = _averages(totals, observations)
        calibration_averages = evaluate_loss(
            model,
            calibration_loader,
            model_name,
            config,
            continuous_mask,
            device,
            augmentation_seed=seed * 7_919 + 41,
        )
        calibration_total = calibration_averages["total"]
        improved = calibration_total < best_calibration_loss
        if improved:
            best_calibration_loss = calibration_total
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        peak_memory = (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        )
        row = {
            "epoch": epoch,
            **{f"train_{name}": value for name, value in train_averages.items()},
            **{
                f"calibration_{name}": value
                for name, value in calibration_averages.items()
            },
            "learning_rate": optimizer.param_groups[0]["lr"],
            "mean_gradient_norm": float(np.mean(gradient_norms)),
            "epoch_seconds": time.time() - epoch_start,
            "peak_gpu_memory_bytes": peak_memory,
        }
        history.append(row)
        _write_history(destination / "training_history.csv", history)
        payload = _checkpoint_payload(
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_epoch,
            best_calibration_loss,
            epochs_without_improvement,
            history,
            model_name,
            seed,
            config,
        )
        torch.save(payload, last_path)
        if improved:
            shutil.copy2(last_path, best_path)
        save_json(
            status_path,
            {
                "state": "running",
                "stage": stage,
                "model": model_name,
                "seed": seed,
                "last_epoch": epoch,
                "best_epoch": best_epoch,
                "best_calibration_loss": best_calibration_loss,
                "epochs_without_improvement": epochs_without_improvement,
                "elapsed_seconds": time.time() - start_time,
            },
        )
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_total": train_averages["total"],
                    "calibration_total": calibration_total,
                    "best_epoch": best_epoch,
                    "no_improvement": epochs_without_improvement,
                    "seconds": row["epoch_seconds"],
                    "peak_gpu_mib": peak_memory / 1024**2,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if epochs_without_improvement >= int(
            config["training"]["early_stopping_patience"]
        ):
            break

    final_status = {
        "state": "complete",
        "stage": stage,
        "model": model_name,
        "seed": seed,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_calibration_loss": best_calibration_loss,
        "elapsed_seconds": time.time() - start_time,
        "early_stopped": len(history) < epochs,
        "best_checkpoint": str(best_path),
    }
    save_json(status_path, final_status)
    return destination


def smoke_test(
    config: dict[str, Any],
    model_name: str,
    seed: int,
    batches: int = 2,
) -> Path:
    seed_everything(seed)
    manifest = load_data_manifest(config)
    micro_batch = int(config["training"]["micro_batch_size"])
    precision_name = config["training"].get("mixed_precision_dtype", "float32")
    destination = (
        run_directory(config, "smoke", model_name, seed)
        / f"batch_{micro_batch}_{precision_name}"
    )
    if destination.exists():
        raise FileExistsError(f"Smoke directory already exists: {destination}")
    destination.mkdir(parents=True)
    length = int(config["window"]["length"])
    dataset = WindowDataset(
        segments_for_split(manifest, "fit"),
        length,
        int(config["window"]["fit_stride"]),
    )
    loader = _loader(
        dataset,
        micro_batch,
        True,
        True,
        seed,
        config,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, len(manifest["feature_names"])).to(device)
    parameters = trainable_parameters(model, model_name)
    optimizer = AdamW(
        parameters,
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    continuous_mask = torch.tensor(
        np.logical_not(manifest["normalization"]["binary_mask"]),
        dtype=torch.bool,
        device=device,
    )
    gradient_rows: list[dict[str, Any]] = []
    shape_record: dict[str, Any] = {}
    model.train()
    generator = _augmentation_generator(device, seed * 101)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for batch_index, batch in enumerate(loader, start=1):
        if batch_index > batches:
            break
        x = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(config, device):
            loss, values = compute_objective(
                model,
                x,
                model_name,
                config,
                continuous_mask,
                generator,
            )
        loss.backward()
        gradients = {
            name: (
                None
                if parameter.grad is None
                else float(parameter.grad.detach().abs().max().item())
            )
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        missing = [name for name, value in gradients.items() if value is None]
        nonfinite = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and parameter.grad is not None
            and not torch.isfinite(parameter.grad).all()
        ]
        if missing:
            raise RuntimeError(f"Active parameters without gradients: {missing}")
        if nonfinite:
            raise FloatingPointError(f"Non-finite gradients: {nonfinite}")
        gradient_norm = clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        with torch.inference_mode():
            outputs = model(x[:2])
        shape_record = {
            "input": list(x.shape),
            **{name: list(value.shape) for name, value in outputs.items()},
        }
        gradient_rows.append(
            {
                "batch": batch_index,
                "losses": {
                    name: float(value.detach().float().item())
                    for name, value in values.items()
                },
                "gradient_norm": float(gradient_norm),
                "minimum_active_gradient_max_abs": min(
                    value for value in gradients.values() if value is not None
                ),
            }
        )

    model.eval()
    with torch.inference_mode():
        probe = torch.randn(2, length, len(manifest["feature_names"]), device=device)
        baseline = model.encode_sequence(probe)
        changed = probe.clone()
        change_at = length // 2
        changed[:, change_at:] += torch.randn_like(changed[:, change_at:]) * 10.0
        perturbed = model.encode_sequence(changed)
        causal_prefix_difference = float(
            (baseline[:, :change_at] - perturbed[:, :change_at]).abs().max().item()
        )
    if causal_prefix_difference > 1e-6:
        raise AssertionError(
            f"Future perturbation changed causal prefix by {causal_prefix_difference}"
        )
    result = {
        "state": "pass",
        "model": model_name,
        "seed": seed,
        "batches": batches,
        "shape_record": shape_record,
        "gradient_checks": gradient_rows,
        "causal_prefix_max_abs_difference": causal_prefix_difference,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
        "parameter_counts": parameter_counts(model),
        "test_labels_used": 0,
    }
    save_json(destination / "smoke_result.json", result)
    return destination
