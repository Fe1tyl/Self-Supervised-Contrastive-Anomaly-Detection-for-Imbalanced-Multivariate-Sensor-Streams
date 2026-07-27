from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(PROJECT_ROOT)
    return resolve_paths(config)


def resolve_paths(config: dict[str, Any]) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    root = Path(resolved["_project_root"])
    archive = Path(resolved["dataset"]["archive"])
    if not archive.is_absolute():
        archive = (root / archive).resolve()
    cache_dir = Path(resolved["dataset"]["cache_dir"])
    if not cache_dir.is_absolute():
        cache_dir = (root / cache_dir).resolve()
    artifacts_dir = Path(resolved["artifacts_dir"])
    if not artifacts_dir.is_absolute():
        artifacts_dir = (root / artifacts_dir).resolve()
    resolved["dataset"]["archive"] = str(archive)
    resolved["dataset"]["cache_dir"] = str(cache_dir)
    resolved["artifacts_dir"] = str(artifacts_dir)
    return resolved


def save_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

