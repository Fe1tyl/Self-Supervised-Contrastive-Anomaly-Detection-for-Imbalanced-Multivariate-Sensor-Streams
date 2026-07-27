from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def seed_everything(seed: int, deterministic: bool = True) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def software_device_manifest() -> dict[str, Any]:
    device: dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
    }
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        device.update(
            {
                "name": torch.cuda.get_device_name(index),
                "total_memory_bytes": props.total_memory,
                "capability": list(torch.cuda.get_device_capability(index)),
            }
        )
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    package_versions = {}
    for package in (
        "numpy",
        "pandas",
        "scipy",
        "torch",
        "PyYAML",
        "tqdm",
        "pytest",
        "scikit-learn",
    ):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": device,
        "git_commit": git_commit,
        "packages": package_versions,
        "determinism": {
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
    }


def write_sha256_manifest(root: str | Path, output_name: str = "sha256_manifest.json") -> Path:
    root_path = Path(root)
    output_path = root_path / output_name
    entries: list[dict[str, Any]] = []
    for path in sorted(p for p in root_path.rglob("*") if p.is_file()):
        if path == output_path:
            continue
        entries.append(
            {
                "path": path.relative_to(root_path).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    output_path.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def finite_or_raise(name: str, arrays: Iterable[np.ndarray | torch.Tensor]) -> None:
    for index, value in enumerate(arrays):
        if isinstance(value, torch.Tensor):
            valid = bool(torch.isfinite(value).all().item())
        else:
            valid = bool(np.isfinite(value).all())
        if not valid:
            raise FloatingPointError(f"{name}[{index}] contains NaN or infinity")
