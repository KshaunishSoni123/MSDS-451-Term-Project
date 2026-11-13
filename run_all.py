"""
Run Checkpoint B (LSTM portfolio) and Checkpoint C (Bollinger overlay) in order.

Stages, in run order:

    Checkpoint B  (via Checkpoint B/portfolio_optimizer.py)
        ingest       prices, news, SEC data for the 20 stocks
        features     point-in-time features + sentiment
        eda          feature and coverage plots
        preprocess   chronological split + scaling
        tune         Optuna search -> best_params.json
        train        LSTM ensemble + out-of-sample backtest -> quarterly weights
        simulate     LSTM Monte Carlo

    Checkpoint C
        bb-ingest    20 years of prices for the standalone Bollinger backtest
        bb-backtest  bollinger_trading.py: standalone strategy, per-ticker report
        bb-simulate  bollinger_monte_carlo.py: Monte Carlo of the standalone strategy
        overlay      overlay_backtest.py: Bollinger trading on top of the LSTM weights

Each stage is a separate script and stops the run on failure. Prices are
cached; pass --refresh to download fresh data in both checkpoints.

Usage:
    ./run_all.sh                          # full refit: every stage
    ./run_all.sh --refresh                # same, re-downloading prices
    ./run_all.sh --skip tune simulate bb-simulate   # quick run with current params
    ./run_all.sh --from train             # retrain the LSTM, then rerun Checkpoint C
    ./run_all.sh --only overlay           # overlay on the current LSTM weights
    ./run_all.sh --trials 40 --max-weight 0.2 --k 2.5
    ./run_all.sh --bond-limit 0.2 --tilt 0.5   # fix the anchor instead of tuning it

Options not listed below (e.g. --trials, --seeds, --max-weight, --tilt,
--bond-limit, --paths, --no-refit) are passed to Checkpoint B's portfolio_optimizer.py; see
``Checkpoint B/portfolio_optimizer.py --help``.
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
B_DIR = ROOT / "Checkpoint B"
C_DIR = ROOT / "Checkpoint C"

B_STAGES = ["ingest", "features", "eda", "preprocess", "tune", "train", "simulate"]
C_STAGES = {
    "bb-ingest": "data_ingestion.py",
    "bb-backtest": "bollinger_trading.py",
    "bb-simulate": "bollinger_monte_carlo.py",
    "overlay": "overlay_backtest.py",
}
STAGES = B_STAGES + list(C_STAGES)

log = logging.getLogger("run_all")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Stages: " + " ".join(STAGES),
    )
    group = parser.add_argument_group("stage selection")
    group.add_argument("--only", nargs="+", choices=STAGES, metavar="STAGE", help="Run just these stages")
    group.add_argument("--from", dest="from_stage", choices=STAGES, metavar="STAGE",
                       help="Start from this stage")
    group.add_argument("--skip", nargs="+", choices=STAGES, metavar="STAGE", help="Skip these stages")

    group = parser.add_argument_group("shared")
    group.add_argument("--refresh", action="store_true",
                       help="Re-download prices in both checkpoints")

    group = parser.add_argument_group("Checkpoint C (bb-backtest, bb-simulate, overlay)")
    group.add_argument("--window", type=int, help="Bollinger SMA / std window (default 20)")
    group.add_argument("--k", type=float, help="Entry threshold in standard deviations (default 2.0)")
    group.add_argument("--commission-bps", type=float,
                       help="Brokerage commission per unit traded, in bps (default 5)")
    group.add_argument("--slippage-bps", type=float, help="Slippage per unit traded, in bps (default 5)")
    group.add_argument("--long-exposure", type=float, help="Multiplier on a long signal (default 1.5)")
    group.add_argument("--short-exposure", type=float, help="Multiplier on a short signal (default 0.5)")
    group.add_argument("--bb-paths", type=int, help="bb-simulate: number of Monte Carlo paths")
    group.add_argument("--cash-reserve", type=float,
                       help="overlay: share kept in SHY to fund long tilts (default 0.10)")
    group.add_argument("--management-fee", type=float, help="overlay: annual management fee (default 0.01)")
    group.add_argument("--performance-fee", type=float,
                       help="overlay: performance fee rate over HWM + hurdle (default 0.20)")
    group.add_argument("--hurdle-rate", type=float, help="overlay: annual hurdle rate (default 0.05)")

    args, b_extra = parser.parse_known_args()
    if args.only and (args.from_stage or args.skip):
        parser.error("--only cannot be combined with --from or --skip")
    return args, b_extra


def selected_stages(args):
    if args.only:
        return [s for s in STAGES if s in args.only]
    start = STAGES.index(args.from_stage) if args.from_stage else 0
    return [s for s in STAGES[start:] if s not in (args.skip or [])]


def c_args(stage, args):
    if stage == "bb-ingest":
        out = ["--universe", "core"]
        return out + (["--refresh"] if args.refresh else [])
    out = []
    for flag, value in (("--window", args.window), ("--k", args.k),
                        ("--commission-bps", args.commission_bps),
                        ("--slippage-bps", args.slippage_bps),
                        ("--long-exposure", args.long_exposure),
                        ("--short-exposure", args.short_exposure)):
        if value is not None:
            out += [flag, str(value)]
    if stage == "bb-simulate" and args.bb_paths is not None:
        out += ["--paths", str(args.bb_paths)]
    if stage == "overlay":
        for flag, value in (("--cash-reserve", args.cash_reserve),
                            ("--management-fee", args.management_fee),
                            ("--performance-fee", args.performance_fee),
                            ("--hurdle-rate", args.hurdle_rate)):
            if value is not None:
                out += [flag, str(value)]
    return out


def run(cmd, cwd, label):
    log.info("=== %s: %s", label, " ".join(str(c) for c in cmd[1:]))
    started = time.time()
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        log.error("%s failed (exit %d) after %.0fs", label, result.returncode, time.time() - started)
        sys.exit(result.returncode)
    log.info("=== %s done in %.0fs", label, time.time() - started)


def main():
    args, b_extra = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stages = selected_stages(args)
    b_stages = [s for s in stages if s in B_STAGES]
    if b_extra and not b_stages:
        sys.exit(f"Unrecognized options {b_extra}: no Checkpoint B stage selected to receive them")
    log.info("Running stages: %s", ", ".join(stages))

    started = time.time()
    if b_stages:
        cmd = [sys.executable, "portfolio_optimizer.py", "--only", *b_stages, *b_extra]
        if args.refresh and "ingest" in b_stages:
            cmd.append("--refresh")
        run(cmd, B_DIR, "Checkpoint B")
    for stage in stages:
        if stage in C_STAGES:
            run([sys.executable, C_STAGES[stage], *c_args(stage, args)], C_DIR, stage)

    log.info("All done in %.1f min. Results in %s and %s", (time.time() - started) / 60,
             B_DIR / "data" / "results", C_DIR / "data" / "results")


if __name__ == "__main__":
    main()
