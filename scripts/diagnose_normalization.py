from __future__ import annotations

import json
from pathlib import Path

import numpy as np


manifest = json.loads(
    Path("data/cache/hai-21.03/data_manifest.json").read_text(encoding="utf-8")
)
features = manifest["feature_names"]
continuous = ~np.asarray(manifest["normalization"]["binary_mask"], dtype=bool)
constant = np.asarray(manifest["normalization"]["constant_mask"], dtype=bool)
arrays = [
    np.load(segment["x"], mmap_mode="r")
    for segment in manifest["segments"]
    if segment["split"] == "fit"
]
for index, name in enumerate(features):
    if not continuous[index]:
        continue
    values = np.concatenate(
        [np.asarray(array[:, index], dtype=np.float64) for array in arrays]
    )
    standard_deviation = float(values.std())
    if standard_deviation < 0.99 or standard_deviation > 1.01:
        print(
            json.dumps(
                {
                    "index": index,
                    "feature": name,
                    "declared_constant": bool(constant[index]),
                    "stored_std": standard_deviation,
                    "stored_unique": len(np.unique(values)),
                    "raw_std": manifest["normalization"][
                        "population_standard_deviation"
                    ][index],
                    "applied_scale": manifest["normalization"]["applied_scale"][
                        index
                    ],
                }
            )
        )
