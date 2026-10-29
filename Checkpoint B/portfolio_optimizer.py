"""
End-to-end pipeline for the biotech portfolio optimizer.

Runs each stage in order, stopping at the first failure:

    ingest      data_ingestion.py       prices, news, SEC filings and facts
    features    feature_engineering.py  point-in-time features + sentiment
    preprocess  preprocessing.py        chronological split + scaling
    tune        tune_lstm.py            Optuna search -> best_params.json
    train       lstm_model.py           final LSTM + out-of-sample backtest
    simulate    monte_carlo.py          10,000-path Monte Carlo

Each stage is its own script and can still be run on its own; this file only
sequences them and forwards the relevant options.

Usage:
    python portfolio_optimizer.py                        # everything
    python portfolio_optimizer.py --skip tune            # reuse existing best_params.json
    python portfolio_optimizer.py --from train           # resume from a stage
    python portfolio_optimizer.py --only simulate --paths 2000
    python portfolio_optimizer.py --refresh --start 2018-01-01 --trials 60

The SEC requires a contact identity for the ingest stage:
    export EDGAR_IDENTITY="Your Name your.email@example.com"
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

STAGES = {
    "ingest": "data_ingestion.py",
    "features": "feature_engineering.py",
    "preprocess": "preprocessing.py",
    "tune": "tune_lstm.py",
    "train": "lstm_model.py",
    "simulate": "monte_carlo.py",
}

log = logging.getLogger("portfolio_optimizer")


def stage_args(stage, args):
    """Command-line options forwarded to each stage (only those that were set)."""
    forwarded = {
        "ingest": [
            ("--tickers", args.tickers), ("--start", args.start),
            ("--refresh", args.refresh), ("--skip-sec", args.skip_sec),
            ("--skip-news", args.skip_news),
        ],
        "features": [("--sentiment", args.sentiment)],
        "preprocess": [("--train-frac", args.train_frac)],
        "tune": [("--trials", args.trials)],
        "train": [("--default-params", args.default_params)],
        "simulate": [("--paths", args.paths), ("--pool-start", args.pool_start)],
    }[stage]

    out = []
    for flag, value in forwarded:
        if value is None or value is False:
            continue
        if value is True:
            out.append(flag)
        elif isinstance(value, list):
            out += [flag, *value]
        else:
            out += [flag, str(value)]
    return out


def selected_stages(args):
    names = list(STAGES)
    if args.only:
        return args.only
    start = names.index(args.from_stage) if args.from_stage else 0
    return [s for s in names[start:] if s not in (args.skip or [])]


def run_stage(stage, args):
    cmd = [sys.executable, str(BASE_DIR / STAGES[stage]), *stage_args(stage, args)]
    log.info("=== %s: %s", stage, " ".join(cmd[1:]))
    started = time.time()
    result = subprocess.run(cmd, cwd=BASE_DIR)
    elapsed = time.time() - started
    if result.returncode != 0:
        log.error("%s failed (exit %d) after %.0fs", stage, result.returncode, elapsed)
        sys.exit(result.returncode)
    log.info("=== %s done in %.0fs", stage, elapsed)


def check_prerequisites(stages, args):
    if "ingest" in stages and not args.skip_sec and not os.environ.get("EDGAR_IDENTITY"):
        sys.exit(
            "EDGAR_IDENTITY is not set; the SEC requires a contact identity:\n"
            '  export EDGAR_IDENTITY="Your Name your.email@example.com"\n'
            "or pass --skip-sec, or --skip ingest to reuse downloaded data."
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_argument_group("stage selection")
    group.add_argument("--only", nargs="+", choices=STAGES, help="Run just these stages")
    group.add_argument("--from", dest="from_stage", choices=STAGES, help="Start from this stage")
    group.add_argument("--skip", nargs="+", choices=STAGES, help="Skip these stages")

    group = parser.add_argument_group("forwarded options")
    group.add_argument("--tickers", nargs="+", help="ingest: tickers (default: built-in universe)")
    group.add_argument("--start", help="ingest: history start date")
    group.add_argument("--refresh", action="store_true", help="ingest: re-download cached data")
    group.add_argument("--skip-sec", action="store_true", help="ingest: skip SEC download")
    group.add_argument("--skip-news", action="store_true", help="ingest: skip news download")
    group.add_argument("--sentiment", choices=["finbert", "vader", "none"], help="features: sentiment model")
    group.add_argument("--train-frac", type=float, help="preprocess: fraction of dates for training")
    group.add_argument("--trials", type=int, help="tune: number of Optuna trials")
    group.add_argument("--default-params", action="store_true", help="train: ignore best_params.json")
    group.add_argument("--paths", type=int, help="simulate: number of Monte Carlo paths")
    group.add_argument("--pool-start", help="simulate: first day of history to bootstrap from")

    args = parser.parse_args()
    if args.only and (args.from_stage or args.skip):
        parser.error("--only cannot be combined with --from or --skip")
    return args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    stages = selected_stages(args)
    check_prerequisites(stages, args)
    log.info("Running stages: %s", ", ".join(stages))

    started = time.time()
    for stage in stages:
        run_stage(stage, args)
    log.info("Pipeline finished in %.1f min. Results in %s",
             (time.time() - started) / 60, BASE_DIR / "data" / "results")


if __name__ == "__main__":
    main()
