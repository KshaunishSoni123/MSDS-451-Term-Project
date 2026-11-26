"""
Head-to-head comparison of our ETF with XBI and SPY.

"Our ETF" is the full strategy from overlay_backtest.py: the Checkpoint B LSTM
portfolio over the XBI holdings, rebalanced quarterly, with the Bollinger
overlay trading on top, net of trading costs, the management fee and the
performance fee. The version before fund fees is shown too, so the cost of the
fee schedule is visible. XBI (biotech sector) and SPY (broad market) are the
two benchmarks an investor could buy instead.

All series cover the same out-of-sample test period. Sharpe ratios and alphas
use returns in excess of SHY, as everywhere else in the project.

Tables (data/results/):
    fund_comparison.csv       return and risk for each series
    fund_relative.csv         our ETF vs each benchmark (and XBI vs SPY): beta,
                              correlation, alpha, tracking error, information
                              ratio, up/down capture
    fund_calendar_returns.csv return by calendar year (first/last are partial)
    fund_comparison.png       growth of $1M and drawdowns

Usage:
    python fund_comparison.py          # after overlay_backtest.py
"""

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bollinger_trading import RESULTS_DIR, TRADING_DAYS, load_prices, performance_metrics

ROOT = Path(__file__).resolve().parent
CHECKPOINT_B = ROOT.parent / "Checkpoint B"
CASH = "SHY"

# overlay_returns.parquet column -> display name
SERIES = {
    "lstm_bollinger": "Our ETF",
    "lstm_bollinger_gross": "Our ETF before fund fees",
    "xbi": "XBI",
    "spy": "SPY",
}
FUND, GROSS = SERIES["lstm_bollinger"], SERIES["lstm_bollinger_gross"]
PAIRS = [(FUND, "SPY"), (FUND, "XBI"), ("XBI", "SPY")]
COLORS = {FUND: "#1f5fa8", GROSS: "#1f5fa8", "XBI": "#c2571a", "SPY": "#5b6770"}

log = logging.getLogger("fund_comparison")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def load_returns(checkpoint_b):
    path = RESULTS_DIR / "overlay_returns.parquet"
    if not path.exists():
        raise SystemExit(f"{path} not found; run overlay_backtest.py first")
    raw = pd.read_parquet(path)
    missing = sorted(set(SERIES) - set(raw.columns))
    if missing:
        raise SystemExit(f"overlay_returns.parquet is missing {missing}; rerun overlay_backtest.py")
    returns = raw[list(SERIES)].rename(columns=SERIES)
    px = load_prices(checkpoint_b / "data" / "raw" / "prices.parquet")
    rf = px[CASH].pct_change(fill_method=None).reindex(returns.index).fillna(0.0)
    return returns, rf


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def absolute_metrics(returns, rf, capital):
    rows = {}
    for name, r in returns.items():
        m = performance_metrics(r, rf=rf)
        monthly = (1 + r).resample("ME").prod() - 1
        rows[name] = {
            "total_return_pct": 100 * m["total_return"],
            "cagr_pct": 100 * m["cagr"],
            "ann_vol_pct": 100 * m["ann_vol"],
            "sharpe": m["sharpe"],
            "sortino": m["sortino"],
            "max_drawdown_pct": 100 * m["max_drawdown"],
            "calmar": m["calmar"],
            "worst_day_pct": 100 * r.min(),
            "best_day_pct": 100 * r.max(),
            "positive_months_pct": 100 * (monthly > 0).mean(),
            "ending_value_usd": capital * (1 + m["total_return"]),
        }
    out = pd.DataFrame(rows).T
    out.index.name = "series"
    return out


def relative_metrics(returns, rf):
    """Each fund vs each benchmark: CAPM beta and alpha on excess returns,
    active return statistics, and monthly up/down capture."""
    excess = returns.sub(rf, axis=0)
    monthly = (1 + returns).resample("ME").prod() - 1
    rows = {}
    for fund, bench in PAIRS:
        f, b = excess[fund], excess[bench]
        beta = f.cov(b) / b.var()
        alpha = (f.mean() - beta * b.mean()) * TRADING_DAYS
        active = returns[fund] - returns[bench]
        te = active.std() * np.sqrt(TRADING_DAYS)
        up, down = monthly[bench] > 0, monthly[bench] < 0
        cagr = {k: (1 + returns[k]).prod() ** (TRADING_DAYS / len(returns)) - 1 for k in (fund, bench)}
        rows[f"{fund} vs {bench}"] = {
            "beta": beta,
            "correlation": returns[fund].corr(returns[bench]),
            "alpha_ann_pct": 100 * alpha,
            "cagr_difference_pct": 100 * (cagr[fund] - cagr[bench]),
            "tracking_error_pct": 100 * te,
            "information_ratio": active.mean() * TRADING_DAYS / te,
            "up_capture_pct": 100 * monthly.loc[up, fund].mean() / monthly.loc[up, bench].mean(),
            "down_capture_pct": 100 * monthly.loc[down, fund].mean() / monthly.loc[down, bench].mean(),
            "months_outperformed_pct": 100 * (monthly[fund] > monthly[bench]).mean(),
        }
    out = pd.DataFrame(rows).T
    out.index.name = "comparison"
    return out


def calendar_returns(returns):
    yearly = (1 + returns).groupby(returns.index.year).prod() - 1
    days = returns.groupby(returns.index.year).size()
    out = 100 * yearly
    out.index = [f"{y}{' (partial)' if days[y] < 240 else ''}" for y in yearly.index]
    out.index.name = "year"
    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot(returns, capital, path):
    wealth = capital * (1 + returns).cumprod()
    start = returns.index[0] - pd.Timedelta(days=1)
    wealth.loc[start] = capital
    wealth = wealth.sort_index()
    drawdown = wealth / wealth.cummax() - 1

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                  gridspec_kw={"height_ratios": [2.2, 1]})
    for name in returns:
        style = {"color": COLORS[name], "lw": 2.2 if name == FUND else 1.4,
                 "ls": "--" if name == GROSS else "-"}
        ax.plot(wealth.index, wealth[name] / 1e6, label=name, **style)
        if name != GROSS:
            ax2.plot(drawdown.index, 100 * drawdown[name], **style)
    for name in returns:
        ax.annotate(f"${wealth[name].iloc[-1] / 1e6:.2f}M", (wealth.index[-1], wealth[name].iloc[-1] / 1e6),
                    xytext=(4, 0), textcoords="offset points", va="center", fontsize=8,
                    color=COLORS[name])
    ax.axhline(capital / 1e6, color="black", lw=0.6, alpha=0.4)
    ax.set_ylabel("Portfolio value ($M)")
    ax.set_title(f"Growth of ${capital / 1e6:.0f}M: our ETF vs XBI and SPY "
                 f"({returns.index[0]:%b %Y} - {returns.index[-1]:%b %Y})")
    ax.legend(loc="upper left", frameon=False)
    ax2.set_ylabel("Drawdown (%)")
    ax2.axhline(0, color="black", lw=0.6, alpha=0.4)
    for a in (ax, ax2):
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint-b", type=Path, default=CHECKPOINT_B,
                        help="Checkpoint B folder (for SHY prices)")
    parser.add_argument("--capital", type=float, default=1_000_000, help="Starting value for $ figures")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    returns, rf = load_returns(args.checkpoint_b)

    absolute = absolute_metrics(returns, rf, args.capital)
    relative = relative_metrics(returns, rf)
    calendar = calendar_returns(returns)
    absolute.to_csv(RESULTS_DIR / "fund_comparison.csv")
    relative.to_csv(RESULTS_DIR / "fund_relative.csv")
    calendar.to_csv(RESULTS_DIR / "fund_calendar_returns.csv")
    plot(returns, args.capital, RESULTS_DIR / "fund_comparison.png")
    log.info("Wrote fund comparison tables and figure to %s", RESULTS_DIR)

    with pd.option_context("display.float_format", "{:,.2f}".format, "display.width", 200):
        print(f"\nOur ETF vs XBI and SPY, {returns.index[0].date()} -> {returns.index[-1].date()} "
              f"(Sharpe and alpha in excess of SHY; our ETF is net of all fees)")
        print(absolute.T.to_string())
        print("\nRelative to each benchmark")
        print(relative.T.to_string())
        print("\nCalendar-year returns (%)")
        print(calendar.to_string())


if __name__ == "__main__":
    main()
