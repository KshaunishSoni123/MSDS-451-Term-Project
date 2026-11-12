"""
Bollinger Band overlay on the Checkpoint B LSTM portfolio.

The LSTM (Checkpoint B, lstm_model.py) sets the baseline portfolio once a
quarter: weights over the 20 biotech stocks plus SHY and TLT. Between
rebalances the Bollinger mean-reversion signal from bollinger_trading.py
resizes each stock relative to its baseline holding:

    holding_i = multiplier_i * base_i

    multiplier = long_exposure   (1.5)  on a long signal  (Z < -k, uptrend)
                 baseline        (1.0)  with no signal
                 short_exposure  (0.5)  on a short signal (Z > +k, downtrend)

base_i is the stock's LSTM weight at the last rebalance, drifted with its own
price since then, i.e. what the portfolio would hold without the overlay.

Cash reserve: at each rebalance every LSTM weight is scaled by
(1 - cash_reserve) and the difference is held in SHY (default 10%), so the
overlay always has cash to fund long tilts even when the LSTM holds almost
no SHY. base_i above is the scaled weight.

Rules:
    * Trades happen only when a stock's signal changes (and at the quarterly
      rebalance). Between those, positions drift with prices; nothing is
      re-sized daily.
    * Every tilt is funded from SHY, the cash proxy (the LSTM's own SHY plus
      the cash reserve): buying the extra 50% sells SHY, and cutting to 50%
      puts the proceeds into SHY. TLT and the other stocks are not touched.
    * SHY cannot go negative. If there is not enough SHY for a buy, the buy
      is scaled down to what SHY can pay for (logged as ``shy_limited``).
    * Stocks the LSTM does not hold are never traded (1.5 x 0 = 0).
    * At each rebalance the base resets to the new LSTM weights and any
      active multiplier is applied to the new base.
    * Costs: ``--cost-bps`` per unit traded on every leg (the stock and the
      SHY side), charged against the next day's return. This matches
      Checkpoint B's run_holding_period, so the overlay-off run reproduces
      Checkpoint B's LSTM backtest.

Trade PnL is the tilt's contribution in dollars: (holding - neutral holding)
x (stock return - SHY return), since SHY funds the tilt, minus the costs of
opening and closing it. Costs from resizing an open tilt at a quarterly
rebalance are counted in portfolio returns but not in any trade.

Sharpe ratios are on returns in excess of SHY, as in Checkpoint B.

Inputs (from Checkpoint B):
    data/results/backtest_weights.parquet   LSTM weights at each rebalance
    data/results/backtest_returns.parquet   LSTM / equal-weight / XBI daily returns
    data/raw/prices.parquet                 prices the LSTM was backtested on

Outputs (data/results/):
    overlay_returns.parquet      daily returns: overlay, LSTM with the cash reserve,
                                 LSTM only, equal weight, XBI
    overlay_weights.parquet      daily portfolio weights with the overlay
    overlay_trades.csv           every tilt round trip
    overlay_ticker_report.csv    latest bands + trade stats per ticker
    overlay_summary.csv          total return, CAGR, Sharpe, closed trades

Usage:
    python overlay_backtest.py
    python overlay_backtest.py --k 2.5 --cost-bps 20
    python overlay_backtest.py --cash-reserve 0.05
    python overlay_backtest.py --checkpoint-b ../"Checkpoint B"
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from bollinger_trading import (
    RESULTS_DIR, BollingerMeanReversion, add_exposure_args, load_prices,
    performance_metrics,
)

ROOT = Path(__file__).resolve().parent
CHECKPOINT_B = ROOT.parent / "Checkpoint B"
CASH = "SHY"
SAFE_HAVENS = ["SHY", "TLT"]

log = logging.getLogger("overlay_backtest")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def load_checkpoint_b(path):
    """LSTM weights, Checkpoint B's daily returns, prices and the test calendar
    (the first rebalance date followed by every return date)."""
    data = path / "data"
    weights = pd.read_parquet(data / "results" / "backtest_weights.parquet").fillna(0.0)
    b_returns = pd.read_parquet(data / "results" / "backtest_returns.parquet")
    px = load_prices(data / "raw" / "prices.parquet")

    first = weights.index[0]
    if b_returns.index[0] <= first:
        raise SystemExit("backtest_returns.parquet must start after the first rebalance")
    calendar = b_returns.index.insert(0, first)
    missing = sorted(set(weights.columns) - set(px.columns))
    if missing or CASH not in weights:
        raise SystemExit(f"Prices or weights missing for {missing or [CASH]}")
    return weights, b_returns, px, calendar


def signal_states(strategy, px, stocks, calendar):
    """Bollinger signal (+1/0/-1) for each stock on each calendar date, from
    indicators built on the full price history (so the 200-day trend SMA is
    warm before the test period starts)."""
    signals, indicators = {}, {}
    for t in stocks:
        ind = strategy.indicators(px[t].loc[:calendar[-1]])
        indicators[t] = ind
        signals[t] = strategy.positions(ind).reindex(calendar).fillna(0).astype(int)
    return pd.DataFrame(signals), indicators


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate(weights, returns, signals, strategy, capital, overlay=True, reserve=0.0):
    """Daily simulation of the LSTM portfolio, with or without the overlay.

    ``reserve`` scales every LSTM weight by (1 - reserve) at each rebalance
    and holds the rest in SHY.

    Returns (daily returns, daily weights, trade log).
    """
    assets = list(weights.columns)
    stocks = list(signals.columns)
    n = len(assets)
    s_idx = np.array([assets.index(s) for s in stocks])
    cash = assets.index(CASH)
    bps = strategy.cost_bps / 1e4
    calendar = returns.index
    rebalance = set(weights.index)

    R = returns[assets].to_numpy()
    S = signals.to_numpy() if overlay else np.zeros((len(calendar), len(stocks)), dtype=int)
    level = {1: strategy.long_exposure, 0: strategy.baseline, -1: strategy.short_exposure}

    H = np.zeros(n)                 # dollar holdings
    base = np.zeros(n)              # LSTM holdings drifted since the last rebalance
    state = np.zeros(len(stocks), dtype=int)
    value, pending_cost = float(capital), 0.0
    daily_ret = np.zeros(len(calendar))
    daily_w = np.zeros((len(calendar), n))
    open_trades, trades = {}, []

    for t, date in enumerate(calendar):
        # -- returns earned over (t-1, t] on yesterday's holdings -----------
        if t > 0:
            tilt = H[s_idx] - strategy.baseline * base[s_idx]
            for j, tr in open_trades.items():
                tr["pnl"] += tilt[j] * (R[t, s_idx[j]] - R[t, cash])
            grown = H * (1 + R[t])
            daily_ret[t] = grown.sum() / value - 1 - pending_cost
            new_value = value * (1 + daily_ret[t])
            scale = new_value / grown.sum()
            H, base, value = grown * scale, base * (1 + R[t]) * scale, new_value

        # -- trades at the close of t ---------------------------------------
        old = H.copy()
        is_rebalance = date in rebalance
        if is_rebalance:
            base = weights.loc[date, assets].to_numpy() * value * (1 - reserve)
            base[cash] += reserve * value
            ref = base.copy()               # sized against the fresh base
            changed = np.ones(len(stocks), dtype=bool)
        else:
            ref = H.copy()
            changed = S[t] != state

        desired = ref.copy()
        mult = np.array([level[int(s)] for s in S[t]])
        desired[s_idx[changed]] = mult[changed] * base[s_idx[changed]]
        delta = desired - ref
        delta[cash] = 0.0
        buys, sells = delta[delta > 0].sum(), -delta[delta < 0].sum()
        fill = min(1.0, (ref[cash] + sells) / buys) if buys > 1e-12 else 1.0
        H = ref + np.where(delta > 0, delta * fill, delta)
        H[cash] = value - (H.sum() - H[cash])

        traded = np.abs(H - old)
        pending_cost = traded.sum() / value * bps

        # -- trade log (only signal changes open or close a trade) ----------
        for j in np.flatnonzero(S[t] != state):
            i = s_idx[j]
            neutral = strategy.baseline * base[i]
            if j in open_trades:
                tr = open_trades.pop(j)
                tr["pnl"] -= 2 * abs(neutral - old[i]) * bps
                tr.update(exit_date=date, holding_days=t - tr.pop("_t"), exit_reason="signal")
                trades.append(tr)
            if S[t, j] != 0 and base[i] > 0:
                open_trades[j] = {
                    "ticker": stocks[j], "side": "long" if S[t, j] > 0 else "short",
                    "entry_date": date, "exit_date": None, "_t": t,
                    "entry_weight": H[i] / value, "base_weight": base[i] / value,
                    "shy_limited": bool(delta[i] > 0 and fill < 1),
                    "pnl": -2 * abs(H[i] - neutral) * bps,
                }
        state = S[t].copy()
        daily_w[t] = H / value

    for j, tr in open_trades.items():
        tr.update(exit_date=calendar[-1], holding_days=len(calendar) - 1 - tr.pop("_t"),
                  exit_reason="open")
        trades.append(tr)

    ret = pd.Series(daily_ret[1:], index=calendar[1:])
    w = pd.DataFrame(daily_w, index=calendar, columns=assets)
    cols = ["ticker", "side", "entry_date", "exit_date", "holding_days", "base_weight",
            "entry_weight", "shy_limited", "pnl", "exit_reason"]
    log_ = pd.DataFrame(trades, columns=cols).rename(columns={"pnl": "net_pnl"})
    return ret, w, log_.sort_values(["entry_date", "ticker"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def ticker_report(stocks, indicators, signals, weights, daily_w, trades):
    """Latest bands, current sizing and closed-trade stats for each stock."""
    closed = trades[trades["exit_reason"] != "open"]
    last_lstm = weights.iloc[-1]
    rows = {}
    for t in stocks:
        ind = indicators[t].loc[:signals.index[-1]]
        last = ind["z"].last_valid_index()
        band = ind.loc[last]
        c = closed[closed["ticker"] == t]
        state = int(signals[t].iloc[-1])
        rows[t] = {
            "as_of": last.date(),
            "price": band["price"],
            "lower_band": band["lower"],
            "middle_sma": band["sma"],
            "upper_band": band["upper"],
            "z": band["z"],
            "signal": {1: "long", -1: "short", 0: "-"}[state],
            "lstm_weight_pct": 100 * last_lstm[t],
            "weight_now_pct": 100 * daily_w[t].iloc[-1],
            "total_trades": len(c),
            "net_pnl_usd": c["net_pnl"].sum(),
            "win_rate_pct": 100 * (c["net_pnl"] > 0).mean() if len(c) else np.nan,
            "avg_days_held": c["holding_days"].mean() if len(c) else np.nan,
        }
    report = pd.DataFrame(rows).T
    report.index.name = "ticker"
    return report


def overall_summary(series, rf, n_closed):
    rows = {}
    for name, r in series.items():
        m = performance_metrics(r, rf=rf)
        rows[name] = {
            "total_return_pct": 100 * m["total_return"],
            "cagr_pct": 100 * m["cagr"],
            "sharpe": m["sharpe"],
            "closed_trades": n_closed.get(name, np.nan),
        }
    summary = pd.DataFrame(rows).T.astype(float)
    summary["closed_trades"] = summary["closed_trades"].astype("Int64")
    summary.index.name = "portfolio"
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint-b", type=Path, default=CHECKPOINT_B,
                        help="Checkpoint B folder with the LSTM backtest outputs")
    parser.add_argument("--capital", type=float, default=1_000_000,
                        help="Starting portfolio value, for dollar PnL")
    parser.add_argument("--window", type=int, default=20, help="Bollinger SMA / std window")
    parser.add_argument("--k", type=float, default=2.0, help="Entry threshold in standard deviations")
    parser.add_argument("--trend-window", type=int, default=200, help="Regime filter SMA window")
    parser.add_argument("--no-trend-filter", action="store_true", help="Disable the 200-day SMA filter")
    parser.add_argument("--long-only", action="store_true", help="Disable short (0.5x) signals")
    parser.add_argument("--cost-bps", type=float, default=10.0, help="Cost per unit traded, in bps")
    parser.add_argument("--cash-reserve", type=float, default=0.10,
                        help="Share of the portfolio kept in SHY to fund long tilts (default 0.10)")
    add_exposure_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    weights, b_returns, px, calendar = load_checkpoint_b(args.checkpoint_b)
    assets = list(weights.columns)
    stocks = [a for a in assets if a not in SAFE_HAVENS]
    strategy = BollingerMeanReversion(
        window=args.window, k=args.k, trend_window=args.trend_window,
        use_trend_filter=not args.no_trend_filter, allow_short=not args.long_only,
        cost_bps=args.cost_bps, baseline=args.baseline,
        long_exposure=args.long_exposure, short_exposure=args.short_exposure,
    )
    log.info("LSTM base: %d stocks + %s, %d rebalances %s -> %s; overlay %s",
             len(stocks), SAFE_HAVENS, len(weights), weights.index[0].date(),
             calendar[-1].date(), strategy)

    returns = px[assets].reindex(calendar).pct_change(fill_method=None).fillna(0.0)
    signals, indicators = signal_states(strategy, px, stocks, calendar)

    if not 0 <= args.cash_reserve < 1:
        raise SystemExit("--cash-reserve must be in [0, 1)")
    ret_overlay, w_overlay, trades = simulate(weights, returns, signals, strategy, args.capital,
                                              reserve=args.cash_reserve)
    # Same cash reserve, no overlay: isolates what the Bollinger trades add
    ret_reserve, w_reserve, _ = simulate(weights, returns, signals, strategy, args.capital,
                                         overlay=False, reserve=args.cash_reserve)
    ret_lstm, w_lstm, _ = simulate(weights, returns, signals, strategy, args.capital, overlay=False)

    # The overlay-off run should reproduce Checkpoint B's own LSTM backtest
    gap = (ret_lstm - b_returns["lstm"]).abs().max()
    (log.info if gap < 1e-6 else log.warning)(
        "LSTM-only vs Checkpoint B backtest: max daily return difference %.2e", gap)

    series = {
        "lstm_bollinger": ret_overlay,
        f"lstm_cash_{100 * args.cash_reserve:.0f}pct": ret_reserve,
        "lstm_only": ret_lstm,
        "equal_weight": b_returns["equal_weight"],
        "xbi": b_returns["xbi"],
    }
    rf = returns[CASH].iloc[1:]
    closed = trades[trades["exit_reason"] != "open"]
    summary = overall_summary(series, rf, {"lstm_bollinger": len(closed)})
    report = ticker_report(stocks, indicators, signals, weights, w_overlay, trades)

    pd.DataFrame(series).to_parquet(RESULTS_DIR / "overlay_returns.parquet")
    w_overlay.to_parquet(RESULTS_DIR / "overlay_weights.parquet")
    trades.to_csv(RESULTS_DIR / "overlay_trades.csv", index=False)
    report.to_csv(RESULTS_DIR / "overlay_ticker_report.csv")
    summary.to_csv(RESULTS_DIR / "overlay_summary.csv")
    log.info("Wrote overlay results to %s", RESULTS_DIR)

    with pd.option_context("display.float_format", "{:,.2f}".format, "display.width", 200):
        print(f"\nPer-ticker Bollinger Bands (window={strategy.window}, k={strategy.k}) "
              "and closed-trade stats (net PnL on "
              f"${args.capital:,.0f} starting capital)")
        print(report.to_string())
        print(f"\nOverall performance {calendar[1].date()} -> {calendar[-1].date()} "
              "(Sharpe in excess of SHY)")
        print(summary.to_string(na_rep="-"))
        limited = int(trades["shy_limited"].sum())
        print(f"\nCash reserve {100 * args.cash_reserve:.0f}%. Average SHY weight: overlay "
              f"{100 * w_overlay[CASH].mean():.1f}% (min {max(0.0, 100 * w_overlay[CASH].min()):.1f}%), "
              f"LSTM only {100 * w_lstm[CASH].mean():.1f}%. "
              f"Entries cut short by SHY running out: {limited} of {len(trades)}.")


if __name__ == "__main__":
    main()
