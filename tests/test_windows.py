from pathlib import Path

import numpy as np

from hai_repro.data import WindowDataset


def test_windows_stay_inside_each_file(tmp_path: Path) -> None:
    segments = []
    for index, rows in enumerate((11, 12)):
        x = np.full((rows, 2), index, dtype=np.float32)
        path = tmp_path / f"x{index}.npy"
        np.save(path, x)
        segments.append({"x": str(path), "name": f"f{index}"})
    dataset = WindowDataset(segments, length=4, stride=2)
    assert len(dataset) == 4 + 5
    for index in range(4):
        assert np.all(dataset[index].numpy() == 0)
    for index in range(4, 9):
        assert np.all(dataset[index].numpy() == 1)
