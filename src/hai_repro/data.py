from __future__ import annotations

import bisect
import gzip
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import save_json
from .utils import finite_or_raise, sha256_file


EXPECTED_ROWS = {
    "train1": 216_001,
    "train2": 226_801,
    "train3": 478_801,
    "test1": 43_201,
    "test2": 118_801,
    "test3": 108_001,
    "test4": 39_601,
    "test5": 92_401,
}
EXPECTED_TEST_ANOMALIES = {
    "test1": 629,
    "test2": 3_449,
    "test3": 1_535,
    "test4": 1_157,
    "test5": 2_177,
}
EXPECTED_TEST_EVENTS = {
    "test1": 5,
    "test2": 20,
    "test3": 8,
    "test4": 5,
    "test5": 12,
}


@dataclass(frozen=True)
class Segment:
    split: str
    name: str
    source: str
    start: int
    stop: int

    @property
    def rows(self) -> int:
        return self.stop - self.start


def _member(prefix: str, name: str) -> str:
    return f"{prefix.rstrip('/')}/{name}.csv.gz"


def _iter_csv(
    archive: Path,
    member: str,
    chunksize: int = 50_000,
    usecols: Sequence[str] | None = None,
) -> Iterator[pd.DataFrame]:
    with zipfile.ZipFile(archive) as zipped:
        with zipped.open(member, "r") as compressed:
            with gzip.GzipFile(fileobj=compressed, mode="rb") as stream:
                reader = pd.read_csv(stream, chunksize=chunksize, usecols=usecols)
                yield from reader


def _event_count(labels: np.ndarray) -> int:
    binary = np.asarray(labels, dtype=np.uint8)
    if binary.size == 0:
        return 0
    starts = binary.astype(np.int8)
    return int(binary[0] == 1) + int(np.sum(np.diff(starts) == 1))


def _merge_moments(
    count: int,
    mean: np.ndarray,
    m2: np.ndarray,
    values: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    batch_count = values.shape[0]
    if batch_count == 0:
        return count, mean, m2
    batch_mean = values.mean(axis=0, dtype=np.float64)
    centered = values - batch_mean
    batch_m2 = np.einsum("ij,ij->j", centered, centered, dtype=np.float64)
    if count == 0:
        return batch_count, batch_mean, batch_m2
    delta = batch_mean - mean
    total = count + batch_count
    merged_mean = mean + delta * (batch_count / total)
    merged_m2 = m2 + batch_m2 + delta * delta * (count * batch_count / total)
    return total, merged_mean, merged_m2


def _segments(config: dict[str, Any]) -> list[Segment]:
    cut = int(config["dataset"]["train3_fit_rows"])
    return [
        Segment("fit", "train1", "train1", 0, EXPECTED_ROWS["train1"]),
        Segment("fit", "train2", "train2", 0, EXPECTED_ROWS["train2"]),
        Segment("fit", "train3_fit", "train3", 0, cut),
        Segment("calibration", "train3_calibration", "train3", cut, EXPECTED_ROWS["train3"]),
        *[
            Segment("test", name, name, 0, rows)
            for name, rows in EXPECTED_ROWS.items()
            if name.startswith("test")
        ],
    ]


def _inspect_source(
    config: dict[str, Any],
) -> tuple[list[str], list[str], dict[str, Any]]:
    archive = Path(config["dataset"]["archive"])
    prefix = config["dataset"]["archive_member_prefix"]
    time_column = config["dataset"]["time_column"]
    label_columns = config["dataset"]["label_columns"]
    fit_cut = int(config["dataset"]["train3_fit_rows"])

    columns: list[str] | None = None
    features: list[str] | None = None
    count = 0
    mean = np.empty(0, dtype=np.float64)
    m2 = np.empty(0, dtype=np.float64)
    binary_mask = np.empty(0, dtype=bool)
    fitting_minimum = np.empty(0, dtype=np.float64)
    fitting_maximum = np.empty(0, dtype=np.float64)
    file_summaries: dict[str, Any] = {}

    for name in EXPECTED_ROWS:
        rows = 0
        anomalies = 0
        labels_for_events: list[np.ndarray] = []
        fit_seen = 0
        for chunk in _iter_csv(archive, _member(prefix, name)):
            if columns is None:
                columns = chunk.columns.tolist()
                missing = {time_column, *label_columns} - set(columns)
                if missing:
                    raise ValueError(f"Missing required columns: {sorted(missing)}")
                features = [
                    column
                    for column in columns
                    if column != time_column and column not in label_columns
                ]
                if len(features) != 79:
                    raise ValueError(f"Expected 79 process variables, got {len(features)}")
                mean = np.zeros(len(features), dtype=np.float64)
                m2 = np.zeros(len(features), dtype=np.float64)
                binary_mask = np.ones(len(features), dtype=bool)
                fitting_minimum = np.full(len(features), np.inf, dtype=np.float64)
                fitting_maximum = np.full(len(features), -np.inf, dtype=np.float64)
            elif chunk.columns.tolist() != columns:
                raise ValueError(f"Column mismatch in {name}")

            values = chunk[features].to_numpy(dtype=np.float64, copy=False)
            finite_or_raise(f"{name}.features", [values])
            labels = chunk[config["dataset"]["label_column"]].to_numpy(
                dtype=np.uint8, copy=False
            )
            if not np.isin(labels, [0, 1]).all():
                raise ValueError(f"Non-binary attack labels in {name}")
            rows += len(chunk)
            anomalies += int(labels.sum())
            if name.startswith("test"):
                labels_for_events.append(labels.copy())

            if name in {"train1", "train2", "train3"}:
                if name == "train3":
                    remaining = max(0, fit_cut - fit_seen)
                    fit_values = values[:remaining]
                    fit_seen += len(values)
                else:
                    fit_values = values
                if len(fit_values):
                    count, mean, m2 = _merge_moments(count, mean, m2, fit_values)
                    binary_mask &= np.all(
                        (fit_values == 0.0) | (fit_values == 1.0), axis=0
                    )
                    fitting_minimum = np.minimum(
                        fitting_minimum, fit_values.min(axis=0)
                    )
                    fitting_maximum = np.maximum(
                        fitting_maximum, fit_values.max(axis=0)
                    )

        if rows != EXPECTED_ROWS[name]:
            raise ValueError(f"{name}: expected {EXPECTED_ROWS[name]} rows, got {rows}")
        event_count = (
            _event_count(np.concatenate(labels_for_events))
            if labels_for_events
            else 0
        )
        if name.startswith("test"):
            if anomalies != EXPECTED_TEST_ANOMALIES[name]:
                raise ValueError(
                    f"{name}: expected {EXPECTED_TEST_ANOMALIES[name]} anomalies, "
                    f"got {anomalies}"
                )
            if event_count != EXPECTED_TEST_EVENTS[name]:
                raise ValueError(
                    f"{name}: expected {EXPECTED_TEST_EVENTS[name]} events, "
                    f"got {event_count}"
                )
        elif anomalies != 0:
            raise ValueError(f"{name} is not normal-only")
        file_summaries[name] = {
            "rows": rows,
            "anomalies": anomalies,
            "events": event_count,
        }

    assert columns is not None and features is not None
    if count != 825_842:
        raise ValueError(f"Expected 825842 fitting rows, got {count}")
    variance = m2 / count
    standard_deviation = np.sqrt(np.maximum(variance, 0.0))
    continuous_mask = ~binary_mask
    constant_mask = continuous_mask & (fitting_minimum == fitting_maximum)
    standard_deviation[constant_mask] = 0.0
    scale = standard_deviation.copy()
    scale[constant_mask] = 1.0
    normalization = {
        "fit_rows": count,
        "feature_names": features,
        "binary_features": np.asarray(features)[binary_mask].tolist(),
        "continuous_features": np.asarray(features)[continuous_mask].tolist(),
        "constant_nonbinary_features": np.asarray(features)[constant_mask].tolist(),
        "constant_feature_policy": config["dataset"]["constant_feature_policy"],
        "binary_mask": binary_mask.tolist(),
        "constant_mask": constant_mask.tolist(),
        "mean": mean.tolist(),
        "fitting_minimum": fitting_minimum.tolist(),
        "fitting_maximum": fitting_maximum.tolist(),
        "population_standard_deviation": standard_deviation.tolist(),
        "applied_scale": scale.tolist(),
    }
    return columns, features, {
        "files": file_summaries,
        "normalization": normalization,
    }


def _write_segment(
    config: dict[str, Any],
    segment: Segment,
    features: Sequence[str],
    normalization: dict[str, Any],
) -> dict[str, Any]:
    archive = Path(config["dataset"]["archive"])
    prefix = config["dataset"]["archive_member_prefix"]
    cache = Path(config["dataset"]["cache_dir"])
    destination = cache / segment.split
    destination.mkdir(parents=True, exist_ok=True)

    x_path = destination / f"{segment.name}_x.npy"
    y_path = destination / f"{segment.name}_y.npy"
    time_path = destination / f"{segment.name}_time.npy"
    row_path = destination / f"{segment.name}_row.npy"

    x_output = np.lib.format.open_memmap(
        x_path, mode="w+", dtype=np.float32, shape=(segment.rows, len(features))
    )
    y_output = np.lib.format.open_memmap(
        y_path, mode="w+", dtype=np.uint8, shape=(segment.rows,)
    )
    time_output = np.lib.format.open_memmap(
        time_path, mode="w+", dtype=np.int64, shape=(segment.rows,)
    )
    row_output = np.lib.format.open_memmap(
        row_path, mode="w+", dtype=np.int64, shape=(segment.rows,)
    )

    means = np.asarray(normalization["mean"], dtype=np.float64)
    scales = np.asarray(normalization["applied_scale"], dtype=np.float64)
    binary = np.asarray(normalization["binary_mask"], dtype=bool)
    continuous = ~binary

    source_offset = 0
    output_offset = 0
    for chunk in _iter_csv(archive, _member(prefix, segment.source)):
        chunk_start = source_offset
        chunk_stop = source_offset + len(chunk)
        overlap_start = max(chunk_start, segment.start)
        overlap_stop = min(chunk_stop, segment.stop)
        source_offset = chunk_stop
        if overlap_start >= overlap_stop:
            continue
        local_start = overlap_start - chunk_start
        local_stop = overlap_stop - chunk_start
        selected = chunk.iloc[local_start:local_stop]
        values = selected[list(features)].to_numpy(dtype=np.float64, copy=True)
        values[:, continuous] = (
            values[:, continuous] - means[continuous]
        ) / scales[continuous]
        finite_or_raise(f"{segment.name}.normalized", [values])
        count = len(selected)
        target = slice(output_offset, output_offset + count)
        x_output[target] = values.astype(np.float32)
        y_output[target] = selected[config["dataset"]["label_column"]].to_numpy(
            dtype=np.uint8, copy=False
        )
        timestamps = pd.to_datetime(
            selected[config["dataset"]["time_column"]], errors="raise"
        ).to_numpy(dtype="datetime64[s]").astype(np.int64)
        time_output[target] = timestamps
        row_output[target] = np.arange(overlap_start, overlap_stop, dtype=np.int64)
        output_offset += count

    if output_offset != segment.rows:
        raise ValueError(
            f"{segment.name}: wrote {output_offset} rows, expected {segment.rows}"
        )
    for array in (x_output, y_output, time_output, row_output):
        array.flush()
    del x_output, y_output, time_output, row_output
    return {
        "split": segment.split,
        "name": segment.name,
        "source": segment.source,
        "source_start_inclusive": segment.start,
        "source_stop_exclusive": segment.stop,
        "rows": segment.rows,
        "x": str(x_path),
        "y": str(y_path),
        "time": str(time_path),
        "row": str(row_path),
    }


def _normalization_audit(
    segments: Sequence[dict[str, Any]],
    normalization: dict[str, Any],
    chunk_size: int = 100_000,
) -> dict[str, Any]:
    continuous = ~np.asarray(normalization["binary_mask"], dtype=bool)
    constant = np.asarray(normalization["constant_mask"], dtype=bool)
    count = 0
    sums = np.zeros(int(continuous.sum()), dtype=np.float64)
    sums_sq = np.zeros(int(continuous.sum()), dtype=np.float64)
    nan_count = 0
    inf_count = 0
    for item in segments:
        if item["split"] != "fit":
            continue
        values = np.load(item["x"], mmap_mode="r")
        for start in range(0, len(values), chunk_size):
            block = np.asarray(values[start : start + chunk_size, continuous])
            nan_count += int(np.isnan(block).sum())
            inf_count += int(np.isinf(block).sum())
            sums += block.sum(axis=0, dtype=np.float64)
            sums_sq += np.einsum("ij,ij->j", block, block, dtype=np.float64)
            count += len(block)
    means = sums / count
    stds = np.sqrt(np.maximum(sums_sq / count - means * means, 0.0))
    continuous_feature_indices = np.flatnonzero(continuous)
    nonconstant_positions = ~constant[continuous_feature_indices]
    nonconstant_stds = stds[nonconstant_positions]
    constant_stds = stds[~nonconstant_positions]
    return {
        "continuous_feature_count": int(continuous.sum()),
        "binary_feature_count": int((~continuous).sum()),
        "constant_nonbinary_feature_count": int(constant.sum()),
        "max_abs_mean": float(np.max(np.abs(means))),
        "min_nonconstant_population_std": float(np.min(nonconstant_stds)),
        "max_nonconstant_population_std": float(np.max(nonconstant_stds)),
        "max_constant_population_std": float(np.max(constant_stds))
        if len(constant_stds)
        else 0.0,
        "nan_count": nan_count,
        "inf_count": inf_count,
    }


def prepare_data(config: dict[str, Any], force: bool = False) -> dict[str, Any]:
    cache = Path(config["dataset"]["cache_dir"])
    manifest_path = cache / "data_manifest.json"
    if manifest_path.exists() and not force:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    cache.mkdir(parents=True, exist_ok=True)
    archive = Path(config["dataset"]["archive"])
    if not archive.is_file():
        raise FileNotFoundError(archive)
    archive_hash = sha256_file(archive)
    expected_hash = config["dataset"].get("expected_archive_sha256")
    if expected_hash and archive_hash.upper() != expected_hash.upper():
        raise ValueError(
            f"Archive SHA-256 mismatch: expected {expected_hash}, got {archive_hash}"
        )

    columns, features, audit = _inspect_source(config)
    written = [
        _write_segment(config, segment, features, audit["normalization"])
        for segment in _segments(config)
    ]
    normalization_audit = _normalization_audit(written, audit["normalization"])
    length = int(config["window"]["length"])
    fit_stride = int(config["window"]["fit_stride"])
    evaluation_stride = int(config["window"]["evaluation_stride"])
    for item in written:
        stride = fit_stride if item["split"] == "fit" else evaluation_stride
        item["window_length"] = length
        item["stride"] = stride
        item["windows"] = max(0, math.floor((item["rows"] - length) / stride) + 1)
    split_windows = {
        split: sum(item["windows"] for item in written if item["split"] == split)
        for split in ("fit", "calibration", "test")
    }
    expected_windows = {"fit": 165_093, "calibration": 95_634, "test": 401_370}
    if split_windows != expected_windows:
        raise ValueError(
            f"Window count mismatch: expected {expected_windows}, got {split_windows}"
        )
    manifest = {
        "schema_version": 1,
        "archive": str(archive),
        "archive_sha256": archive_hash,
        "archive_member_prefix": config["dataset"]["archive_member_prefix"],
        "columns": columns,
        "feature_names": features,
        "label_columns": config["dataset"]["label_columns"],
        "files": audit["files"],
        "normalization": audit["normalization"],
        "normalization_audit": normalization_audit,
        "segments": written,
        "split_rows": {
            split: sum(item["rows"] for item in written if item["split"] == split)
            for split in ("fit", "calibration", "test")
        },
        "split_windows": split_windows,
        "window_generation": "after row-level split, independently within each file/segment",
        "score_alignment": "window final timestamp",
    }
    save_json(manifest_path, manifest)
    return manifest


def load_data_manifest(config: dict[str, Any]) -> dict[str, Any]:
    path = Path(config["dataset"]["cache_dir"]) / "data_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Prepared-data manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


class WindowDataset(Dataset[torch.Tensor | dict[str, torch.Tensor]]):
    def __init__(
        self,
        segments: Sequence[dict[str, Any]],
        length: int,
        stride: int,
        include_metadata: bool = False,
    ) -> None:
        self.segments = list(segments)
        self.length = int(length)
        self.stride = int(stride)
        self.include_metadata = include_metadata
        self.arrays = [np.load(item["x"], mmap_mode="r") for item in self.segments]
        self.labels = (
            [np.load(item["y"], mmap_mode="r") for item in self.segments]
            if include_metadata
            else []
        )
        self.timestamps = (
            [np.load(item["time"], mmap_mode="r") for item in self.segments]
            if include_metadata
            else []
        )
        self.rows = (
            [np.load(item["row"], mmap_mode="r") for item in self.segments]
            if include_metadata
            else []
        )
        self.endpoints = [
            np.arange(self.length - 1, len(array), self.stride, dtype=np.int64)
            for array in self.arrays
        ]
        self.cumulative: list[int] = []
        running = 0
        for endpoints in self.endpoints:
            running += len(endpoints)
            self.cumulative.append(running)

    def __len__(self) -> int:
        return self.cumulative[-1] if self.cumulative else 0

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        file_index = bisect.bisect_right(self.cumulative, index)
        previous = self.cumulative[file_index - 1] if file_index else 0
        local_index = index - previous
        return file_index, int(self.endpoints[file_index][local_index])

    def __getitem__(self, index: int) -> torch.Tensor | dict[str, torch.Tensor]:
        file_index, endpoint = self._locate(index)
        start = endpoint - self.length + 1
        window = np.array(
            self.arrays[file_index][start : endpoint + 1],
            dtype=np.float32,
            copy=True,
        )
        x = torch.from_numpy(window)
        if not self.include_metadata:
            return x
        return {
            "x": x,
            "y": torch.tensor(int(self.labels[file_index][endpoint]), dtype=torch.uint8),
            "time": torch.tensor(
                int(self.timestamps[file_index][endpoint]), dtype=torch.int64
            ),
            "row": torch.tensor(int(self.rows[file_index][endpoint]), dtype=torch.int64),
            "file_index": torch.tensor(file_index, dtype=torch.int16),
            "endpoint": torch.tensor(endpoint, dtype=torch.int64),
        }


def segments_for_split(
    manifest: dict[str, Any], split: str
) -> list[dict[str, Any]]:
    return [item for item in manifest["segments"] if item["split"] == split]
