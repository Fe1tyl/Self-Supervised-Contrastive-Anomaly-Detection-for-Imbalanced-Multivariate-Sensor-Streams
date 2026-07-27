from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .config import load_config
from .baselines import run_isolation_forest_baseline, run_persistence_baseline
from .aggregate import aggregate_completed_runs
from .data import prepare_data
from .evaluate import evaluate_saved_scores
from .score import score_model
from .train import smoke_test, train_model


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hai-repro",
        description="Unified HAI 21.03 experiment pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="audit and preprocess HAI 21.03")
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--force", action="store_true")

    smoke = subparsers.add_parser("smoke", help="run two-batch integrity smoke test")
    smoke.add_argument("--config", required=True)
    smoke.add_argument("--model", default="full")
    smoke.add_argument("--seed", type=int, default=17)
    smoke.add_argument("--batches", type=int, default=2)

    train = subparsers.add_parser("train", help="run or resume model training")
    train.add_argument("--config", required=True)
    train.add_argument("--model", required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--maximum-epochs", type=int)
    train.add_argument("--stage", choices=["formal", "pilot"], default="formal")
    train.add_argument("--resume", action="store_true")

    score = subparsers.add_parser("score", help="generate raw component scores")
    score.add_argument("--config", required=True)
    score.add_argument("--model", required=True)
    score.add_argument("--seed", type=int, required=True)
    score.add_argument("--stage", choices=["formal", "pilot"], default="formal")

    evaluate = subparsers.add_parser(
        "evaluate", help="evaluate saved scores and export evidence"
    )
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--seed", type=int, required=True)
    evaluate.add_argument("--components")
    evaluate.add_argument("--quantile", type=float)
    evaluate.add_argument("--stage", choices=["formal", "pilot"], default="formal")
    evaluate.add_argument("--no-timestamp-export", action="store_true")

    persistence = subparsers.add_parser(
        "persistence", help="run the deterministic persistence baseline"
    )
    persistence.add_argument("--config", required=True)
    persistence.add_argument("--seed-label", type=int, default=17)
    persistence.add_argument("--stage", choices=["formal", "pilot"], default="formal")

    isolation_forest = subparsers.add_parser(
        "isolation-forest", help="run the classical Isolation Forest baseline"
    )
    isolation_forest.add_argument("--config", required=True)
    isolation_forest.add_argument("--seed", type=int, required=True)
    isolation_forest.add_argument(
        "--stage", choices=["formal", "pilot"], default="formal"
    )

    aggregate = subparsers.add_parser(
        "aggregate", help="aggregate completed evaluations into paper tables"
    )
    aggregate.add_argument("--config", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "prepare":
        result = prepare_data(config, force=args.force)
        print(
            json.dumps(
                {
                    "manifest": str(
                        Path(config["dataset"]["cache_dir"]) / "data_manifest.json"
                    ),
                    "split_rows": result["split_rows"],
                    "split_windows": result["split_windows"],
                    "normalization_audit": result["normalization_audit"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "smoke":
        output = smoke_test(config, args.model, args.seed, args.batches)
        print(output)
    elif args.command == "train":
        output = train_model(
            config,
            args.model,
            args.seed,
            maximum_epochs=args.maximum_epochs,
            stage=args.stage,
            resume=args.resume,
        )
        print(output)
    elif args.command == "score":
        output = score_model(config, args.model, args.seed, args.stage)
        print(output)
    elif args.command == "evaluate":
        components = (
            tuple(value.strip() for value in args.components.split(","))
            if args.components
            else None
        )
        output = evaluate_saved_scores(
            config,
            args.model,
            args.seed,
            components=components,
            quantile=args.quantile,
            stage=args.stage,
            export_timestamp_scores=not args.no_timestamp_export,
        )
        print(output)
    elif args.command == "persistence":
        output = run_persistence_baseline(
            config, seed_label=args.seed_label, stage=args.stage
        )
        print(output)
    elif args.command == "isolation-forest":
        output = run_isolation_forest_baseline(config, args.seed, args.stage)
        print(output)
    elif args.command == "aggregate":
        output = aggregate_completed_runs(config)
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
