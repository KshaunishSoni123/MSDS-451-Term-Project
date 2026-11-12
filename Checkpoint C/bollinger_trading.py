"""
Bollinger Band mean-reversion trading and backtest simulation.

Strategy (applied to each ticker independently, on adjusted close P_t):

    SMA_t     = n-day simple moving average of P_t            (middle band)
    sigma_t   = n-day rolling standard deviation of P_t
    Z_t       = (P_t - SMA_t) / sigma_t                       (z-score)
    TREND_t   = 200-day simple moving average of P_t          (regime filter)

    Enter long   when Z_t < -k   and  P_t > TREND_t   (price below the lower band, uptrend)
    Enter short  when Z_t > +k   and  P_t < TREND_t   (price above the upper band, downtrend)
    Exit long    when Z_t crosses back up through 0   (price returns to the SMA)
    Exit short   when Z_t crosses back down through 0

``Z_t < -k`` is the same rule as ``P_t - SMA_t < -k * sigma_t``, i.e. the
price closing below the lower Bollinger Band. The trend filter applies to
entries only; an open position is held until Z_t crosses 0.

Beta baseline: the rules above produce a signal S_t in {-1, 0, +1}, and the
position held is

    exposure_t = baseline        if S_t = 0
                 long_exposure   if S_t = +1
                 short_exposure  if S_t = -1

With the defaults (1.0 / 1.5 / 0.5) the stock is held at 100% while Z is
near 0, raised to 150% on a long signal and cut to 50% on a short signal.
``--baseline 0 --long-exposure 1 --short-exposure -1`` gives the pure
long/short strategy.

Execution: signals are computed from the close of day t and filled at that
close, so the position earns the return from t to t+1. Every change in
position pays ``cost_bps`` per unit traded (100% -> 150% is 0.5 units).

Inputs:  data/raw/prices.parquet   (written by data_ingestion.py)
Outputs: data/results/bb_metrics.csv          per-ticker and portfolio metrics
         data/results/bb_trades.csv           every round-trip trade
         data/results/bb_daily.parquet        daily positions and returns per ticker
         data/results/bb_portfolio.parquet    equal-weight portfolio vs benchmarks
         data/results/bb_ticker_report.csv    latest bands + closed-trade stats per ticker
         data/results/bb_summary.csv          total return, CAGR, Sharpe, closed trades
         data/results/bb_sweep.csv            parameter grid (with --sweep)

Usage:
    python bollinger_trading.py
    python bollinger_trading.py --tickers VRTX REGN AMGN --k 2.5 --window 20
    python bollinger_trading.py --no-trend-filter --cost-bps 20
    python bollinger_trading.py --sweep
"""

import argparse
import logging
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PRICES_PATH = ROOT / "data" / "raw" / "prices.parquet"
RESULTS_DIR = ROOT / "data" / "results"

TRADING_DAYS = 252

# Benchmarks written by data_ingestion.py; never traded by the strategy
BENCHMARK_TICKERS = {"XBI", "IBB", "XLV", "SPY", "TLT", "SHY", "^VIX"}

log = logging.getLogger("bollinger_trading")


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def performance_metrics(returns, rf=None):
    """Annualized summary statistics for a daily simple-return series.

    ``rf`` (a daily return series, e.g. SHY) makes Sharpe use excess returns,
    matching Checkpoint B; without it Sharpe uses raw returns.
    """
    r = returns.dropna()
    if r.empty:
        return {}
    excess = r - rf.reindex(r.index).fillna(0) if rf is not None else r
    equity = (1 + r).cumprod()
    years = len(r) / TRADING_DAYS
    vol = r.std() * np.sqrt(TRADING_DAYS)
    downside = r.clip(upper=0).pow(2).mean() ** 0.5 * np.sqrt(TRADING_DAYS)
    drawdown = equity / equity.cummax() - 1
    total = equity.iloc[-1] - 1
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 and equity.iloc[-1] > 0 else np.nan
    return {
        "start": r.index[0].date(),
        "end": r.index[-1].date(),
        "total_return": total,
        "cagr": cagr,
        "ann_vol": vol,
        "sharpe": excess.mean() / excess.std() * np.sqrt(TRADING_DAYS) if excess.std() > 0 else np.nan,
        "sortino": r.mean() * TRADING_DAYS / downside if downside > 0 else np.nan,
        "max_drawdown": drawdown.min(),
        "calmar": cagr / -drawdown.min() if drawdown.min() < 0 else np.nan,
    }


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Output of ``BollingerMeanReversion.backtest`` for one ticker."""
    ticker: str
    daily: pd.DataFrame       # indicators, signal, position, gross/net returns, equity
    trades: pd.DataFrame      # one row per round trip
    metrics: dict = field(default_factory=dict)


@dataclass
class BollingerMeanReversion:
    """Bollinger Band z-score mean reversion with a long-term SMA regime filter.

    Parameters
    ----------
    window : int
        Lookback for the middle band (SMA) and its standard deviation.
    k : float
        Entry threshold in standard deviations (the band width).
    trend_window : int
        Lookback for the regime-filter SMA. Longs are only opened above it,
        shorts only below it.
    use_trend_filter : bool
        Turn the regime filter off to compare against the unfiltered strategy.
    allow_short : bool
        Set False for a long-only version.
    cost_bps : float
        Transaction cost per unit of position traded, in basis points.
    baseline : float
        Exposure held when there is no signal (1.0 = fully invested).
    long_exposure : float
        Exposure held on a long signal (Z < -k).
    short_exposure : float
        Exposure held on a short signal (Z > +k); negative means net short.
    """
    window: int = 20
    k: float = 2.0
    trend_window: int = 200
    use_trend_filter: bool = True
    allow_short: bool = True
    cost_bps: float = 10.0
    baseline: float = 1.0
    long_exposure: float = 1.5
    short_exposure: float = 0.5

    def __post_init__(self):
        if self.window < 2:
            raise ValueError("window must be at least 2")
        if self.k <= 0:
            raise ValueError("k must be positive")
        if not self.short_exposure <= self.baseline <= self.long_exposure:
            raise ValueError("need short_exposure <= baseline <= long_exposure")

    # -- indicators ---------------------------------------------------------

    def indicators(self, price):
        """Bands, z-score and regime SMA for a price series indexed by date.

        Uses population std (ddof=0), matching Bollinger's definition and the
        bb_* columns from data_ingestion.py. Every value at date t uses only
        prices up to and including t.
        """
        price = price.dropna().astype(float)
        sma = price.rolling(self.window, min_periods=self.window).mean()
        sigma = price.rolling(self.window, min_periods=self.window).std(ddof=0)
        trend = price.rolling(self.trend_window, min_periods=self.trend_window).mean()
        z = (price - sma) / sigma.replace(0, np.nan)
        return pd.DataFrame({
            "price": price,
            "sma": sma,
            "sigma": sigma,
            "upper": sma + self.k * sigma,
            "lower": sma - self.k * sigma,
            "z": z,
            "trend_sma": trend,
        })

    # -- signals ------------------------------------------------------------

    def positions(self, ind):
        """Signal state (+1 long, -1 short, 0 neutral) at each close."""
        pos = self.positions_array(
            ind["z"].to_numpy()[:, None],
            ind["price"].to_numpy()[:, None],
            ind["trend_sma"].to_numpy()[:, None],
        )
        return pd.Series(pos[:, 0], index=ind.index, name="signal")

    def positions_array(self, z, price, trend):
        """Trading rules on (time, series) arrays; one column per ticker/path.
        Returns the signal state; ``exposure_array`` turns it into a position.

        The rules depend on the position already held (the zero-cross exit
        only applies inside a trade), so this walks forward through time, but
        each step is vectorized across columns. The historical backtest and
        the Monte Carlo simulation both use this one implementation.
        NaN comparisons are False, so no entries happen during warm-up.
        """
        n_time, n_cols = z.shape
        pos = np.zeros((n_time, n_cols), dtype=np.int8)
        current = np.zeros(n_cols, dtype=np.int8)

        if self.use_trend_filter:
            long_ok = price > trend            # False while trend is NaN
            short_ok = price < trend
        else:
            long_ok = short_ok = np.ones((n_time, n_cols), dtype=bool)
        if not self.allow_short:
            short_ok = np.zeros((n_time, n_cols), dtype=bool)

        for t in range(n_time):
            zt = z[t]
            # Exit when Z crosses back through 0 (the price reaches the SMA)
            current[((current == 1) & (zt >= 0)) | ((current == -1) & (zt <= 0))] = 0
            # Entry, possibly on the same bar as an exit
            flat = current == 0
            go_long = flat & (zt < -self.k) & long_ok[t]
            go_short = flat & ~go_long & (zt > self.k) & short_ok[t]
            current[go_long] = 1
            current[go_short] = -1
            pos[t] = current

        return pos

    def exposure_array(self, signal, z):
        """Position size for each signal state, from the first day with a
        valid z-score onward (flat before that, so the warm-up is not traded)."""
        active = np.maximum.accumulate(~np.isnan(z), axis=0)
        level = np.where(signal > 0, self.long_exposure,
                         np.where(signal < 0, self.short_exposure, self.baseline))
        return np.where(active, level, 0.0)

    # -- backtest -----------------------------------------------------------

    def backtest(self, price, ticker=""):
        """Simulate the strategy on one price series."""
        daily = self.indicators(price)
        daily["signal"] = self.positions(daily)
        daily["position"] = self.exposure_array(
            daily["signal"].to_numpy()[:, None], daily["z"].to_numpy()[:, None]
        )[:, 0]

        held = daily["position"].shift(1).fillna(0)          # position over (t-1, t]
        daily["asset_return"] = daily["price"].pct_change()
        daily["turnover"] = daily["position"].diff().abs().fillna(daily["position"].abs())
        daily["gross_return"] = held * daily["asset_return"]
        daily["cost"] = daily["turnover"] * self.cost_bps / 1e4
        daily["net_return"] = (daily["gross_return"] - daily["cost"]).fillna(0)
        daily["equity"] = (1 + daily["net_return"]).cumprod()

        trades = self.extract_trades(daily)
        metrics = self.summarize(daily, trades)
        return BacktestResult(ticker, daily, trades, metrics)

    def extract_trades(self, daily):
        """Signal round trips (entry to zero cross). Returns are the tilt's
        contribution, (exposure - baseline) * price move, net of costs on both legs.
        A trade still open on the last date is marked ``exit_reason='open'``
        and valued at that close."""
        pos = daily["signal"]
        price = daily["price"]
        trades, side, entry_date = [], 0, None

        def close_trade(exit_date, reason):
            entry_px, exit_px = price[entry_date], price[exit_date]
            tilt = (self.long_exposure if side == 1 else self.short_exposure) - self.baseline
            gross = tilt * (exit_px / entry_px - 1)
            trades.append({
                "side": "long" if side == 1 else "short",
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_price": entry_px,
                "exit_price": exit_px,
                "entry_z": daily.at[entry_date, "z"],
                "exit_z": daily.at[exit_date, "z"],
                "holding_days": pos.index.get_loc(exit_date) - pos.index.get_loc(entry_date),
                "gross_return": gross,
                "net_return": gross - 2 * abs(tilt) * self.cost_bps / 1e4,
                "exit_reason": reason,
            })

        for date, p in pos.items():
            if side != 0 and p != side:
                close_trade(date, "z_cross_zero")
                side = 0
            if side == 0 and p != 0:
                side, entry_date = p, date
        if side != 0:
            close_trade(pos.index[-1], "open")

        return pd.DataFrame(trades)

    def summarize(self, daily, trades):
        # Measure from the first date the strategy could trade
        live = daily["net_return"][daily["z"].first_valid_index():]
        m = performance_metrics(live)
        m["buy_hold_cagr"] = performance_metrics(daily["asset_return"][live.index])["cagr"]
        closed = trades[trades["exit_reason"] != "open"] if not trades.empty else trades
        m.update({
            "n_trades": len(trades),
            "n_long": int((trades["side"] == "long").sum()) if len(trades) else 0,
            "n_short": int((trades["side"] == "short").sum()) if len(trades) else 0,
            "n_closed": len(closed),
            "net_pnl": closed["net_return"].sum() if len(closed) else 0.0,
            "win_rate": (closed["net_return"] > 0).mean() if len(closed) else np.nan,
            "avg_trade_return": closed["net_return"].mean() if len(closed) else np.nan,
            "avg_holding_days": closed["holding_days"].mean() if len(closed) else np.nan,
            "time_tilted": (daily.loc[live.index, "signal"] != 0).mean(),
            "avg_exposure": daily.loc[live.index, "position"].mean(),
        })
        return m


# ---------------------------------------------------------------------------
# Multi-ticker simulation
# ---------------------------------------------------------------------------

def load_prices(path=PRICES_PATH):
    if not path.exists():
        raise SystemExit(f"{path} not found; run data_ingestion.py first")
    prices = pd.read_parquet(path, columns=["date", "ticker", "adj_close"])
    return prices.pivot(index="date", columns="ticker", values="adj_close").sort_index()


def run_portfolio(strategy, px, tickers):
    """Backtest each ticker and combine them into an equal-weight portfolio.

    Each ticker is a sleeve of 1/N capital, where N is the number of tickers
    with a live signal (past the warm-up window) on that day, so names that
    IPO'd later join the portfolio once they have enough history.
    """
    results = {t: strategy.backtest(px[t], ticker=t) for t in tickers}

    net = pd.DataFrame({t: r.daily["net_return"] for t, r in results.items()})
    live = pd.DataFrame({t: r.daily["z"].notna() for t, r in results.items()})
    live = live.reindex(net.index).fillna(False).astype(bool)
    portfolio = net.where(live).mean(axis=1).dropna()

    asset = px[tickers].pct_change().where(live.reindex(px.index).fillna(False))
    bench = pd.DataFrame({
        "strategy": portfolio,
        "equal_weight_buy_hold": asset.mean(axis=1).reindex(portfolio.index),
    })
    for b in ("XBI", "SPY"):
        if b in px:
            bench[b] = px[b].pct_change().reindex(portfolio.index)
    return results, bench


def collect(results, bench, strategy):
    metrics = pd.DataFrame({t: r.metrics for t, r in results.items()}).T
    metrics.index.name = "ticker"
    for name in bench:
        row = performance_metrics(bench[name].fillna(0))
        if name == "strategy":
            trades = pd.concat([r.trades for r in results.values()], ignore_index=True)
            closed = trades[trades["exit_reason"] != "open"]
            row.update({
                "n_trades": len(trades),
                "n_long": int((trades["side"] == "long").sum()),
                "n_short": int((trades["side"] == "short").sum()),
                "n_closed": len(closed),
                "win_rate": (closed["net_return"] > 0).mean(),
                "avg_trade_return": closed["net_return"].mean(),
                "avg_holding_days": closed["holding_days"].mean(),
            })
        metrics.loc[f"PORTFOLIO:{name}"] = pd.Series(row)

    trades = pd.concat(
        [r.trades.assign(ticker=t) for t, r in results.items() if not r.trades.empty],
        ignore_index=True,
    )
    trades = trades[["ticker"] + [c for c in trades if c != "ticker"]]
    daily = pd.concat(
        [r.daily.assign(ticker=t).reset_index() for t, r in results.items()],
        ignore_index=True,
    )
    for key, value in asdict(strategy).items():
        metrics[key] = value
    return metrics, trades, daily


def ticker_report(results, metrics):
    """Latest Bollinger Bands and closed-trade stats for each ticker.

    Bands are as of each ticker's last price date. Trade stats count closed
    round trips only; net PnL is the sum of their net returns (the tilt's
    contribution, as a fraction of the ticker's sleeve, not compounded).
    """
    rows = {}
    for t, r in results.items():
        last = r.daily["z"].last_valid_index()
        bands = r.daily.loc[last] if last is not None else pd.Series(dtype=float)
        m = metrics.loc[t]
        rows[t] = {
            "as_of": last.date() if last is not None else None,
            "price": bands.get("price", np.nan),
            "lower_band": bands.get("lower", np.nan),
            "middle_sma": bands.get("sma", np.nan),
            "upper_band": bands.get("upper", np.nan),
            "z": bands.get("z", np.nan),
            "total_trades": int(m["n_closed"]),
            "net_pnl_pct": 100 * m["net_pnl"],
            "win_rate_pct": 100 * m["win_rate"],
            "avg_days_held": m["avg_holding_days"],
        }
    report = pd.DataFrame(rows).T
    report.index.name = "ticker"
    return report


def overall_summary(metrics):
    """Total return %, CAGR %, Sharpe and closed trades for the strategy
    portfolio and each benchmark (benchmarks have no trades)."""
    port = metrics.loc[metrics.index.str.startswith("PORTFOLIO:")]
    summary = pd.DataFrame({
        "total_return_pct": 100 * port["total_return"].astype(float),
        "cagr_pct": 100 * port["cagr"].astype(float),
        "sharpe": port["sharpe"].astype(float),
        "closed_trades": port["n_closed"],
    })
    summary.index = summary.index.str.replace("PORTFOLIO:", "", regex=False)
    summary.index.name = "portfolio"
    return summary


def sweep(px, tickers, base, windows, ks):
    """Portfolio metrics over a grid of (window, k), with and without the
    trend filter. This is in-sample: use it to see how sensitive the
    strategy is, not to pick the single best row."""
    rows = []
    for window, k, use_filter in product(windows, ks, (True, False)):
        strat = BollingerMeanReversion(
            window=window, k=k, trend_window=base.trend_window,
            use_trend_filter=use_filter, allow_short=base.allow_short,
            cost_bps=base.cost_bps, baseline=base.baseline,
            long_exposure=base.long_exposure, short_exposure=base.short_exposure,
        )
        results, bench = run_portfolio(strat, px, tickers)
        m = performance_metrics(bench["strategy"])
        m["n_trades"] = sum(len(r.trades) for r in results.values())
        rows.append({"window": window, "k": k, "trend_filter": use_filter, **m})
        log.info("window=%d k=%.2f filter=%s  sharpe=%.2f  cagr=%.2f%%",
                 window, k, use_filter, m["sharpe"], 100 * m["cagr"])
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def add_exposure_args(parser):
    parser.add_argument("--baseline", type=float, default=1.0,
                        help="Exposure with no signal (0 for the pure long/short strategy)")
    parser.add_argument("--long-exposure", type=float, default=1.5,
                        help="Exposure on a long signal (Z < -k)")
    parser.add_argument("--short-exposure", type=float, default=0.5,
                        help="Exposure on a short signal (Z > +k); -1 for a full short")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Tickers to trade (default: every non-benchmark ticker in prices.parquet)")
    parser.add_argument("--window", type=int, default=20, help="Bollinger SMA / std window")
    parser.add_argument("--k", type=float, default=2.0, help="Entry threshold in standard deviations")
    parser.add_argument("--trend-window", type=int, default=200, help="Regime filter SMA window")
    parser.add_argument("--no-trend-filter", action="store_true", help="Disable the 200-day SMA filter")
    parser.add_argument("--long-only", action="store_true", help="Disable short entries")
    parser.add_argument("--cost-bps", type=float, default=10.0, help="Cost per unit traded, in bps")
    add_exposure_args(parser)
    parser.add_argument("--start", default=None, help="Only trade from this date, YYYY-MM-DD")
    parser.add_argument("--min-history", type=int, default=250,
                        help="Skip tickers with fewer price days than this")
    parser.add_argument("--sweep", action="store_true",
                        help="Also run a (window, k, filter) parameter grid")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    px = load_prices()
    if args.start:
        # Keep enough earlier history to warm up the trend SMA
        first = px.index.searchsorted(pd.Timestamp(args.start))
        px = px.iloc[max(0, first - args.trend_window):]

    candidates = [t.upper() for t in args.tickers] if args.tickers else [
        t for t in px.columns if t not in BENCHMARK_TICKERS
    ]
    missing = sorted(set(candidates) - set(px.columns))
    if missing:
        log.warning("No prices for %s; run data_ingestion.py with these tickers", missing)
    tickers = [t for t in candidates if t in px and px[t].count() >= args.min_history]
    skipped = sorted(set(candidates) - set(tickers) - set(missing))
    if skipped:
        log.info("Skipping %d tickers with < %d days of history: %s",
                 len(skipped), args.min_history, skipped)
    if not tickers:
        raise SystemExit("No tickers to trade")

    strategy = BollingerMeanReversion(
        window=args.window, k=args.k, trend_window=args.trend_window,
        use_trend_filter=not args.no_trend_filter, allow_short=not args.long_only,
        cost_bps=args.cost_bps, baseline=args.baseline,
        long_exposure=args.long_exposure, short_exposure=args.short_exposure,
    )
    log.info("Backtesting %d tickers with %s", len(tickers), strategy)

    results, bench = run_portfolio(strategy, px, tickers)
    metrics, trades, daily = collect(results, bench, strategy)

    metrics.to_csv(RESULTS_DIR / "bb_metrics.csv")
    trades.to_csv(RESULTS_DIR / "bb_trades.csv", index=False)
    daily.to_parquet(RESULTS_DIR / "bb_daily.parquet", index=False)
    bench.to_parquet(RESULTS_DIR / "bb_portfolio.parquet")
    report = ticker_report(results, metrics)
    summary = overall_summary(metrics)
    report.to_csv(RESULTS_DIR / "bb_ticker_report.csv")
    summary.to_csv(RESULTS_DIR / "bb_summary.csv")
    log.info("Wrote results to %s", RESULTS_DIR)

    with pd.option_context("display.float_format", "{:,.2f}".format, "display.width", 160):
        print(f"\nPer-ticker Bollinger Bands (window={strategy.window}, k={strategy.k}) "
              "and closed-trade stats")
        print(report.to_string())
        print("\nOverall performance")
        print(summary.to_string(na_rep="-"))

    if args.sweep:
        grid = sweep(px, tickers, strategy, windows=[10, 20, 30, 50], ks=[1.5, 2.0, 2.5, 3.0])
        grid.to_csv(RESULTS_DIR / "bb_sweep.csv", index=False)
        with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 140):
            print("\nParameter sweep (in-sample, sorted by Sharpe)")
            print(grid[["window", "k", "trend_filter", "cagr", "sharpe", "max_drawdown", "n_trades"]]
                  .head(10).to_string(index=False))


if __name__ == "__main__":
    main()
