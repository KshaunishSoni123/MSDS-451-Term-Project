"""
Monte Carlo backtest of the Bollinger Band mean-reversion strategy.

The historical backtest (bollinger_trading.py) is a single path, so it says
little about how much of its result is luck. This script builds many
synthetic 20-year price histories from the real data and runs the same
strategy (BollingerMeanReversion) on every one of them.

Path generation: stationary block bootstrap (Politis & Romano, 1994)
    * Daily log returns of every traded ticker plus XBI are resampled in
      blocks of consecutive real trading days. Block lengths are geometric
      with mean ``--block-length``.
    * Whole dates are drawn jointly across tickers, which keeps the
      cross-sectional correlation (biotech names tend to move together).
    * Within a block the real day-to-day ordering is kept, which preserves
      short-horizon autocorrelation and volatility clustering. That
      short-horizon structure is what a 20-day mean-reversion signal trades
      on. Trends longer than a block are broken up, so the 200-day regime
      filter sees fewer long trends than in real history.
    * Tickers that listed after a sampled date (e.g. MRNA before 2018) have no
      return that day; it is filled with a random day from that ticker's own
      history.
    * Every path is ``trend_window`` warm-up days plus ``--years`` * 252
      trading days. Metrics cover only the post-warm-up period.

Each path is traded as the equal-weight portfolio from bollinger_trading.py
(1/N capital per ticker) and compared on the same path with equal-weight
buy-and-hold and with XBI.

Outputs (data/results/):
    bb_mc_paths.parquet       one row per path: strategy, buy-hold and XBI metrics
    bb_mc_summary.csv         percentiles of each metric across paths
    bb_mc_equity_bands.parquet  percentiles of portfolio equity over time

Usage:
    python bollinger_monte_carlo.py                       # 10,000 paths x 20 years
    python bollinger_monte_carlo.py --paths 1000 --seed 7
    python bollinger_monte_carlo.py --k 2.5 --no-trend-filter --block-length 21
"""

import argparse
import logging
import time

import numpy as np
import pandas as pd

from bollinger_trading import (
    BENCHMARK_TICKERS, RESULTS_DIR, TRADING_DAYS, BollingerMeanReversion,
    add_exposure_args, load_prices,
)

log = logging.getLogger("bollinger_monte_carlo")

QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]


# ---------------------------------------------------------------------------
# Path generation
# ---------------------------------------------------------------------------

def stationary_bootstrap_indices(rng, n_paths, n_days, n_source, mean_block):
    """(n_paths, n_days) indices into the source dates.

    Each day either continues the current block (next real date) or, with
    probability 1/mean_block, starts a new block at a random date.
    """
    idx = np.empty((n_paths, n_days), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n_source, n_paths)
    restart = rng.random((n_paths, n_days)) < 1.0 / mean_block
    fresh = rng.integers(0, n_source, (n_paths, n_days))
    for t in range(1, n_days):
        nxt = idx[:, t - 1] + 1
        idx[:, t] = np.where(restart[:, t] | (nxt >= n_source), fresh[:, t], nxt)
    return idx


def simulate_log_returns(rng, log_rets, n_paths, n_days, mean_block):
    """Bootstrap (n_days, n_paths, n_tickers) log returns from a (dates, tickers)
    array that may contain NaN where a ticker was not yet listed."""
    idx = stationary_bootstrap_indices(rng, n_paths, n_days, len(log_rets), mean_block)
    sim = log_rets[idx.T]                                  # (days, paths, tickers)
    for j in range(log_rets.shape[1]):
        holes = np.isnan(sim[:, :, j])
        if holes.any():
            valid = log_rets[~np.isnan(log_rets[:, j]), j]
            sim[:, :, j][holes] = valid[rng.integers(0, len(valid), holes.sum())]
    return sim


# ---------------------------------------------------------------------------
# Vectorized strategy on (time, columns) price arrays
# ---------------------------------------------------------------------------

def rolling_mean_std(x, window):
    """Rolling mean and population std along axis 0 for NaN-free arrays;
    the first window-1 rows are NaN, matching pandas min_periods=window."""
    c1 = np.cumsum(x, axis=0)
    c2 = np.cumsum(x * x, axis=0)
    s1 = c1.copy()
    s2 = c2.copy()
    s1[window:] -= c1[:-window]
    s2[window:] -= c2[:-window]
    mean = s1 / window
    var = np.maximum(s2 / window - mean ** 2, 0.0)
    mean[: window - 1] = np.nan
    var[: window - 1] = np.nan
    return mean, np.sqrt(var)


def strategy_returns(strategy, price):
    """Daily net strategy returns (time, columns) for synthetic price paths,
    with the same indicator, signal and cost rules as
    BollingerMeanReversion.backtest."""
    sma, sigma = rolling_mean_std(price, strategy.window)
    trend, _ = rolling_mean_std(price, strategy.trend_window)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (price - sma) / np.where(sigma > 0, sigma, np.nan)
    signal = strategy.positions_array(z, price, trend)
    pos = strategy.exposure_array(signal, z)

    asset = np.zeros_like(price)
    asset[1:] = price[1:] / price[:-1] - 1
    held = np.zeros_like(pos)
    held[1:] = pos[:-1]
    turnover = np.abs(np.diff(pos, axis=0, prepend=0.0))
    net = held * asset - turnover * strategy.cost_bps / 1e4
    entries = (signal != 0) & (np.diff(signal, axis=0, prepend=0) != 0)
    return net, asset, pos, signal, entries


# ---------------------------------------------------------------------------
# Metrics on (time, paths) return arrays
# ---------------------------------------------------------------------------

def path_metrics(r, prefix):
    """Annualized metrics for each column of a (time, paths) return array."""
    years = r.shape[0] / TRADING_DAYS
    equity = np.cumprod(1 + r, axis=0)
    final = equity[-1]
    mean, std = r.mean(axis=0), r.std(axis=0, ddof=1)
    drawdown = (equity / np.maximum.accumulate(equity, axis=0) - 1).min(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cagr = np.where(final > 0, final ** (1 / years) - 1, -1.0)
        sharpe = np.where(std > 0, mean / std * np.sqrt(TRADING_DAYS), np.nan)
    return {
        f"{prefix}_terminal_wealth": final,
        f"{prefix}_cagr": cagr,
        f"{prefix}_ann_vol": std * np.sqrt(TRADING_DAYS),
        f"{prefix}_sharpe": sharpe,
        f"{prefix}_max_drawdown": drawdown,
    }


# ---------------------------------------------------------------------------
# Simulation driver
# ---------------------------------------------------------------------------

def run(strategy, px, tickers, n_paths, years, mean_block, batch_size, seed):
    rng = np.random.default_rng(seed)
    cols = tickers + (["XBI"] if "XBI" in px else [])
    log_rets = np.log(px[cols]).diff().iloc[1:].to_numpy()
    # Drop dates where nothing traded (e.g. leading NaN rows)
    log_rets = log_rets[~np.isnan(log_rets).all(axis=1)]

    warmup = max(strategy.trend_window, strategy.window)
    n_days = warmup + years * TRADING_DAYS
    n_tick = len(tickers)
    log.info("Simulating %d paths x %d days (%d warm-up) x %d tickers from %d real days",
             n_paths, n_days, warmup, n_tick, len(log_rets))

    rows, strat_eq, bh_eq = [], [], []
    started = time.time()
    for b0 in range(0, n_paths, batch_size):
        nb = min(batch_size, n_paths - b0)
        sim = simulate_log_returns(rng, log_rets, nb, n_days, mean_block)
        price = 100 * np.exp(np.cumsum(sim, axis=0))                  # (days, paths, cols)

        # Trade every (path, ticker) column at once
        stock_px = price[:, :, :n_tick].reshape(n_days, nb * n_tick)
        net, asset, pos, signal, entries = strategy_returns(strategy, stock_px)
        net = net[warmup:].reshape(-1, nb, n_tick)
        asset = asset[warmup:].reshape(-1, nb, n_tick)
        pos = pos[warmup:].reshape(-1, nb, n_tick)
        signal = signal[warmup:].reshape(-1, nb, n_tick)
        entries = entries[warmup:].reshape(-1, nb, n_tick)

        port = net.mean(axis=2)                                       # equal weight
        buy_hold = asset.mean(axis=2)
        batch = {"path": np.arange(b0, b0 + nb)}
        batch.update(path_metrics(port, "strategy"))
        batch.update(path_metrics(buy_hold, "buy_hold"))
        if "XBI" in cols:
            xbi = np.exp(sim[warmup:, :, -1]) - 1
            batch.update(path_metrics(xbi, "xbi"))
        batch["n_trades"] = entries.sum(axis=(0, 2))
        batch["n_long"] = (entries & (signal > 0)).sum(axis=(0, 2))
        batch["n_short"] = (entries & (signal < 0)).sum(axis=(0, 2))
        batch["time_tilted"] = (signal != 0).mean(axis=(0, 2))
        batch["avg_exposure"] = pos.mean(axis=(0, 2))
        rows.append(pd.DataFrame(batch))

        strat_eq.append(np.cumprod(1 + port, axis=0).astype(np.float32))
        bh_eq.append(np.cumprod(1 + buy_hold, axis=0).astype(np.float32))

        done = b0 + nb
        if done == n_paths or done % (batch_size * 10) == 0:
            log.info("  %d / %d paths (%.0fs)", done, n_paths, time.time() - started)

    paths = pd.concat(rows, ignore_index=True)
    strat_eq = np.concatenate(strat_eq, axis=1)
    bh_eq = np.concatenate(bh_eq, axis=1)
    bands = {}
    for name, eq in (("strategy", strat_eq), ("buy_hold", bh_eq)):
        for q, values in zip(QUANTILES, np.quantile(eq, QUANTILES, axis=1)):
            bands[f"{name}_p{int(q * 100):02d}"] = values
    bands = pd.DataFrame(bands)
    bands.index.name = "day"
    return paths, bands


def summarize(paths):
    metrics = [c for c in paths if c != "path"]
    summary = paths[metrics].quantile(QUANTILES).T
    summary.columns = [f"p{int(q * 100):02d}" for q in QUANTILES]
    summary.insert(0, "mean", paths[metrics].mean())

    probs = {
        "P(strategy CAGR > 0)": (paths["strategy_cagr"] > 0).mean(),
        "P(strategy CAGR > buy-hold CAGR)": (paths["strategy_cagr"] > paths["buy_hold_cagr"]).mean(),
        "P(strategy Sharpe > buy-hold Sharpe)": (paths["strategy_sharpe"] > paths["buy_hold_sharpe"]).mean(),
        "P(strategy max DD shallower than buy-hold)":
            (paths["strategy_max_drawdown"] > paths["buy_hold_max_drawdown"]).mean(),
    }
    if "xbi_cagr" in paths:
        probs["P(strategy CAGR > XBI CAGR)"] = (paths["strategy_cagr"] > paths["xbi_cagr"]).mean()
        probs["P(strategy Sharpe > XBI Sharpe)"] = (paths["strategy_sharpe"] > paths["xbi_sharpe"]).mean()
    return summary, pd.Series(probs, name="probability")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--paths", type=int, default=10_000, help="Number of simulated paths")
    parser.add_argument("--years", type=int, default=20, help="Years per path (after warm-up)")
    parser.add_argument("--block-length", type=float, default=63,
                        help="Mean bootstrap block length in trading days")
    parser.add_argument("--batch-size", type=int, default=100, help="Paths simulated per batch")
    parser.add_argument("--seed", type=int, default=451, help="Random seed")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Tickers to trade (default: every non-benchmark ticker in prices.parquet)")
    parser.add_argument("--min-history", type=int, default=250,
                        help="Skip tickers with fewer price days than this")
    parser.add_argument("--window", type=int, default=20, help="Bollinger SMA / std window")
    parser.add_argument("--k", type=float, default=2.0, help="Entry threshold in standard deviations")
    parser.add_argument("--trend-window", type=int, default=200, help="Regime filter SMA window")
    parser.add_argument("--no-trend-filter", action="store_true", help="Disable the 200-day SMA filter")
    parser.add_argument("--long-only", action="store_true", help="Disable short entries")
    parser.add_argument("--cost-bps", type=float, default=5.0, help="Cost per unit traded, in bps")
    add_exposure_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    px = load_prices()
    candidates = [t.upper() for t in args.tickers] if args.tickers else [
        t for t in px.columns if t not in BENCHMARK_TICKERS
    ]
    tickers = [t for t in candidates if t in px and px[t].count() >= args.min_history]
    if not tickers:
        raise SystemExit("No tickers to simulate; run data_ingestion.py first")

    strategy = BollingerMeanReversion(
        window=args.window, k=args.k, trend_window=args.trend_window,
        use_trend_filter=not args.no_trend_filter, allow_short=not args.long_only,
        cost_bps=args.cost_bps, baseline=args.baseline,
        long_exposure=args.long_exposure, short_exposure=args.short_exposure,
    )
    log.info("%s on %d tickers", strategy, len(tickers))

    paths, bands = run(strategy, px, tickers, args.paths, args.years,
                       args.block_length, args.batch_size, args.seed)
    summary, probs = summarize(paths)

    paths.to_parquet(RESULTS_DIR / "bb_mc_paths.parquet", index=False)
    pd.concat([summary, probs.to_frame().rename(columns={"probability": "mean"})]).to_csv(
        RESULTS_DIR / "bb_mc_summary.csv")
    bands.to_parquet(RESULTS_DIR / "bb_mc_equity_bands.parquet")
    log.info("Wrote results to %s", RESULTS_DIR)

    show = [f"{p}_{m}" for m in ("cagr", "sharpe", "max_drawdown", "terminal_wealth")
            for p in ("strategy", "buy_hold", "xbi") if f"{p}_{m}" in paths]
    show += ["n_trades", "time_tilted", "avg_exposure"]
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 140):
        print(f"\nMonte Carlo: {args.paths:,} paths x {args.years} years")
        print(summary.loc[show].to_string())
        print()
        print(probs.to_string())


if __name__ == "__main__":
    main()
