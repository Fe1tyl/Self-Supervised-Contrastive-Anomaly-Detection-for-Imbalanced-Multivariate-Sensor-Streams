# HAI 21.03 Unified Reproduction

This project is the clean-room, executable reproduction package for the paper
“Self-Supervised Contrastive Anomaly Detection for Imbalanced Multivariate
Sensor Streams.” It does not import any of the hand-entered result tables from
the manuscript. Every reported number is recomputed from HAI 21.03 source data,
saved scores, and explicit metric code.

## Frozen protocol

- Dataset: HAI 21.03 (`hai-master.zip`, SHA-256 recorded at preparation time).
- Model fitting: all of `train1`, all of `train2`, and the first 383,040 rows of
  `train3` (825,842 observations).
- Normal calibration: the last 95,761 rows of `train3`.
- Tests: `test1`--`test5`, evaluated separately and pooled without allowing an
  event to cross a file boundary.
- Inputs: 79 process variables; 128-second causal windows.
- Strides: five for model fitting and one for calibration/test.
- Labels and scores: aligned only to the last timestamp in each window.
- Normalization: continuous-feature mean and population standard deviation are
  fitted on model-fitting observations only. Features whose fitting values are
  entirely in `{0, 1}` are retained as binary variables. A non-binary feature
  that is constant in model fitting is centered and assigned unit scale, which
  matches standard scaler behavior while retaining all 79 declared inputs.
  Constant channels and their zero fitting variance are reported separately.
- Seeds: 17, 42, and 2026.
- Operating threshold: `Q_0.995` of normal-calibration scores, NumPy
  `method="higher"`, with the strict rule `score > threshold`.

The expected window counts are 165,093 model-fitting windows, 95,634
calibration windows, and 401,370 test windows.

## Reproduction stages

Run commands from this directory. The `PYTHON` placeholder below must point to
an environment containing PyTorch, NumPy, pandas, PyYAML, and pytest.

```powershell
$PYTHON = "C:\path\to\python.exe"
& $PYTHON -m hai_repro.cli prepare --config configs/hai2103.json
& $PYTHON -m pytest -q
& $PYTHON -m hai_repro.cli smoke --config configs/hai2103.json --seed 17
& $PYTHON -m hai_repro.cli train --config configs/hai2103.json --model full --seed 17
& $PYTHON -m hai_repro.cli score --config configs/hai2103.json --model full --seed 17
& $PYTHON -m hai_repro.cli evaluate --config configs/hai2103.json --model full --seed 17
& $PYTHON -m hai_repro.cli isolation-forest --config configs/hai2103.json --seed 17
& $PYTHON -m hai_repro.cli evaluate --config configs/hai2103.json --model isolation_forest --seed 17
```

The Isolation Forest baseline represents each normalized window with per-channel
mean, standard deviation, minimum, maximum, final value, and final-minus-first
change (474 fixed features). Its estimator parameters are frozen in the main
configuration before test evaluation.

The smoke test is intentionally separate from formal training. Formal results
are written under `artifacts/formal/`; smoke outputs go under
`artifacts/smoke/` and cannot be mistaken for paper results.

## Evidence package

Each formal run contains:

- resolved configuration, software/device information, and source hashes;
- epoch-level loss history and the selected checkpoint;
- calibration component statistics and thresholds;
- timestamp-level raw component scores, standardized scores, fused scores,
  labels, predictions, and file/row identifiers;
- reference and predicted interval tables;
- point, overlap-hit event, duration-coverage, event-coverage, false-alarm, and
  delay metrics;
- a SHA-256 manifest for the complete run directory.

No performance target is used to tune, filter, or accept a run. A run passes
integrity checks when its protocol, shapes, alignment, numerical values, and
artifact counts are correct, even if the scientific hypothesis is unsupported.
