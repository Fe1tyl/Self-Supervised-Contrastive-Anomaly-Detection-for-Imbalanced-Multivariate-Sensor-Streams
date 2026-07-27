# HAI 21.03 Unified Reproduction Release

This archive is the companion evidence package for *Self-Supervised Contrastive
Anomaly Detection for Imbalanced Multivariate Sensor Streams: A HAI 21.03
Benchmark Study*.

## Scope

- HAI 21.03, 79 process variables, 128-step causal windows.
- Split before normalization and window generation.
- Model-fitting/calibration/test windows: 165,093 / 95,634 / 401,370.
- Seeds: 17, 42, and 2026.
- Primary threshold: normal-calibration Q0.995, NumPy `higher`, strict `score > threshold`.
- Coverage-aware event recall at 10% and 50% reference-duration coverage.
- Maximum-cardinality one-to-one overlap matching for event counts.

## Contents

`hai_unified_reproduction/` contains the versioned source, frozen configuration,
tests, every formal checkpoint/model file, timestamp-level scores and predictions,
per-file metrics, event intervals and matches, environment records, and per-run
SHA-256 manifests. `artifacts/failed/` retains the excluded float16 failure for
audit. `paper_artifacts/` contains the generated tables and threshold figure.

The source HAI archive and extracted dataset cache are intentionally excluded.
Download HAI 21.03 from the repository cited in the paper. The frozen expected
archive SHA-256 is
`3697AF1B7ED3B653C4959535DB2B4541A1EE18104A0003DD5C5E9E3927A97B3A`.

## Main result

The unified rerun does not support the original complementarity hypothesis.
The full model obtained Macro AP 0.6037 +/- 0.1190. Removing the contrastive
objective obtained 0.7356 +/- 0.0087, while the contrastive-only model obtained
0.0297 +/- 0.0134. These are HAI 21.03 benchmark findings, not cross-dataset or
state-of-the-art claims.

## Verification

Start with `release_manifest.json`, then compare the nested per-run manifests.
The aggregate entry points are:

- `hai_unified_reproduction/artifacts/summary/model_summary.csv`
- `hai_unified_reproduction/artifacts/summary/seed_level_metrics.csv`
- `hai_unified_reproduction/artifacts/summary/seed_file_metrics.csv`
- `paper_artifacts/paper_primary_table.csv`
- `paper_artifacts/paper_ablation_table.csv`
- `paper_artifacts/paper_threshold_sensitivity.csv`

Run the test suite from `hai_unified_reproduction/` with `PYTHONPATH=src` before
re-executing the CLI commands documented in its README.
