"""
Monte Carlo simulation of the LSTM portfolio over simulated future markets.

Generates ``N_PATHS`` one-year futures starting from the last real trading day
and runs the trained LSTM ensemble through each one with quarterly
rebalancing, alongside equal-weight (biotech only) and XBI benchmarks on the
same paths.

Simulation method:
    * Stationary block bootstrap (Politis & Romano, 1994) of historical daily
      returns. Each sampled day carries every asset plus XBI/SPY/TLT/SHY
      together, so cross-asset correlation, fat tails and (within blocks)
      volatility clustering are preserved. Block lengths are geometric with
      mean ``MEAN_BLOCK`` days.
    * VIX follows a mean-reverting AR(1) in logs, driven by the residual from
      the same bootstrapped day, so it stays tied to the simulated returns
      without drifting to implausible levels.
    * At each rebalance the LSTM sees features recomputed from the simulated
      prices (momentum, volatility, beta, regime, market cap ratios, HMM bear
      probability) with the same formulas, scalers and HMM as the real
      pipeline. The HMM filter continues from its real state on the last day.
      Slow-moving fundamentals, 8-K counts and eligibility are held at their
      last real values.
    * The first rebalance uses the real feature history, so it is the same
      portfolio in every path; later rebalances react to each simulated path.

Outputs:
    data/results/monte_carlo_paths.parquet    Per-path metrics per strategy
    data/results/monte_carlo_summary.csv      Distribution summary
    data/results/monte_carlo_weights.parquet  Mean/std LSTM weights per rebalance

Usage:
    python monte_carlo.py
    python monte_carlo.py --paths 10000 --pool-start 2024-06-14
"""

import argparse
import logging

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import regime_model
from event_calendar import CPI_RELEASES, FOMC_DECISIONS
from feature_engineering import scheduled_event_features
from lstm_model import (
    BENCHMARK, HOLD_DAYS, LOOKBACK, MODEL_DIR, RAW_DIR, RESULTS_DIR, RISK_FREE,
    TRADING_DAYS, equal_weights, load_inputs, load_model, performance, predict,
    predict_weights, run_holding_period, to_tensors,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FEATURE_DIR = MODEL_DIR.parent / "features"

N_PATHS = 10_000
HORIZON = TRADING_DAYS     # one simulated year
MEAN_BLOCK = 21            # mean bootstrap block length (trading days)
HISTORY = 400              # real days prepended so rolling features are complete
CHUNK = 50                 # paths simulated at once (memory scales with universe size)
SEED = 451

MARKET_SERIES = [BENCHMARK, "SPY", "TLT", RISK_FREE]
VIX = "^VIX"

# Asset features recomputed from simulated prices; everything else is held
PRICE_FEATURES = [
    "mom_12_1", "mom_6_1", "ret_1m", "vol_63d", "vol_252d", "downside_vol_63d",
    "beta_xbi_252d", "jump_count_252d", "dist_52w_high", "log_dollar_volume_63d",
]
MARKET_CAP_FEATURES = ["log_market_cap", "book_to_market", "cash_to_market_cap"]

log = logging.getLogger("monte_carlo")


# ---------------------------------------------------------------------------
# Rolling-window helpers on (paths, time, series) tensors
# ---------------------------------------------------------------------------

def shift(x, k):
    return torch.cat([torch.full_like(x[:, :k], float("nan")), x[:, :-k]], dim=1)


# Like pandas rolling(w, min_periods): NaNs are skipped, and a window needs at
# least ``minp`` valid values (stocks listed recently have partial history).

def _window_sum(x, w):
    cs = torch.cumsum(x, dim=1)
    out = cs.clone()
    out[:, w:] = cs[:, w:] - cs[:, :-w]
    return out


def rolling_sum(x, w, minp):
    valid = ~torch.isnan(x)
    total = _window_sum(torch.nan_to_num(x), w)
    count = _window_sum(valid.double(), w)
    return torch.where(count >= minp, total, float("nan")), count


def rolling_mean(x, w, minp):
    total, count = rolling_sum(x, w, minp)
    return total / count


def rolling_std(x, w, minp):
    """Sample standard deviation (ddof=1), matching pandas."""
    total, count = rolling_sum(x, w, minp)
    total_sq, _ = rolling_sum(x * x, w, minp)
    var = (total_sq - total * total / count) / (count - 1)
    return var.clamp(min=0).sqrt()


def rolling_max(x, w, minp):
    filled = torch.where(torch.isnan(x), float("-inf"), x)
    xp = F.pad(filled.transpose(1, 2), (w - 1, 0), value=float("-inf"))
    out = F.max_pool1d(xp, w, stride=1).transpose(1, 2)
    _, count = rolling_sum(x, w, minp)
    return torch.where(count >= minp, out, float("nan"))


# ---------------------------------------------------------------------------
# Feature recomputation (mirrors feature_engineering.py)
# ---------------------------------------------------------------------------

def price_features(px, close, volume, market):
    """px/close/volume (P, T, N), market (P, T, 1) adjusted prices."""
    ann = np.sqrt(TRADING_DAYS)
    r = px / shift(px, 1) - 1
    m = market / shift(market, 1) - 1
    w63, w252 = (63, 42), (252, 126)       # (window, min_periods) as in feature_engineering
    cov = rolling_mean(r * m, *w252) - rolling_mean(r, *w252) * rolling_mean(m, *w252)
    var_m = rolling_mean(m * m, *w252) - rolling_mean(m, *w252) ** 2
    jumps = torch.where(torch.isnan(r), float("nan"), (r.abs() > 0.10).double())
    return {
        "mom_12_1": shift(px, 21) / shift(px, 252) - 1,
        "mom_6_1": shift(px, 21) / shift(px, 126) - 1,
        "ret_1m": px / shift(px, 21) - 1,
        "vol_63d": rolling_std(r, *w63) * ann,
        "vol_252d": rolling_std(r, *w252) * ann,
        "downside_vol_63d": rolling_mean(r.clamp(max=0) ** 2, *w63).sqrt() * ann,
        "beta_xbi_252d": cov / var_m,
        "jump_count_252d": rolling_sum(jumps, *w252)[0],
        "dist_52w_high": px / rolling_max(px, *w252) - 1,
        "log_dollar_volume_63d": torch.log(rolling_mean(close * volume, *w63)),
    }


def market_cap_features(close, fundamentals):
    """close (P, T, N); fundamentals: dict of (N,) last reported raw values."""
    market_cap = fundamentals["shares_outstanding"] * close
    liquid = fundamentals["cash"] + torch.nan_to_num(fundamentals["short_term_investments"])
    return {
        "log_market_cap": torch.log(market_cap),
        "book_to_market": fundamentals["equity"] / market_cap,
        "cash_to_market_cap": liquid / market_cap,
    }


def regime_features(xbi, spy, tlt, vix):
    """Inputs (P, T, 1) -> dict of (P, T) regime features."""
    def ret(x, n=63):
        return x / shift(x, n) - 1

    r_xbi = xbi / shift(xbi, 1) - 1
    feats = {
        "vix": vix,
        "vix_change_21d": vix - shift(vix, 21),
        "vix_zscore_252d": (vix - rolling_mean(vix, 252, 126)) / rolling_std(vix, 252, 126),
        "xbi_ret_63d": ret(xbi),
        "tlt_ret_63d": ret(tlt),
        "xbi_minus_tlt_63d": ret(xbi) - ret(tlt),
        "xbi_minus_spy_63d": ret(xbi) - ret(spy),
        "xbi_vol_21d": rolling_std(r_xbi, 21, 15) * np.sqrt(TRADING_DAYS),
        "xbi_drawdown_252d": xbi / rolling_max(xbi, 252, 126) - 1,
        "spy_drawdown_252d": spy / rolling_max(spy, 252, 126) - 1,
    }
    return {k: v[..., 0] for k, v in feats.items()}


def event_features(last_date, n_days):
    """FOMC / CPI flags and countdowns on the simulated business-day calendar.

    The event calendar is published in advance, so these are deterministic and
    identical on every path. Returns a dict of (1, n_days) tensors.
    """
    dates = pd.bdate_range(last_date + pd.offsets.BDay(), periods=n_days)
    events = pd.concat([
        scheduled_event_features("fomc", FOMC_DECISIONS, dates),
        scheduled_event_features("cpi", CPI_RELEASES, dates),
    ], axis=1)
    return {k: torch.tensor(events[k].to_numpy())[None] for k in events}


def scale(raw, fitted):
    """Apply the training clip bounds + StandardScaler; NaN -> 0 (train mean)."""
    names = fitted["features"]
    lower = torch.tensor(fitted["lower"][names].to_numpy())
    upper = torch.tensor(fitted["upper"][names].to_numpy())
    mean = torch.tensor(fitted["scaler"].mean_)
    std = torch.tensor(fitted["scaler"].scale_)
    x = torch.stack([raw[n] for n in names], dim=-1)
    x = torch.maximum(torch.minimum(x, upper), lower)
    return torch.nan_to_num((x - mean) / std).float()


# ---------------------------------------------------------------------------
# Market data and bootstrap
# ---------------------------------------------------------------------------

def load_market(dates):
    prices = pd.read_parquet(RAW_DIR / "prices.parquet")
    prices["date"] = prices["date"].astype("datetime64[ns]")
    return {
        field: prices.pivot(index="date", columns="ticker", values=field).reindex(dates)
        for field in ("adj_close", "close", "volume")
    }


def last_fundamentals(tickers):
    """Last raw (unscaled) feature row per asset."""
    feats = pd.read_parquet(FEATURE_DIR / "stock_features.parquet")
    return feats[feats["date"] == feats["date"].max()].set_index("ticker").reindex(tickers)


def bootstrap_pool(adj, series, pool_start):
    """Daily returns for all series and VIX AR(1) residuals on common days.

    Assets that were not yet listed on a sampled day get XBI's return for that
    day (a beta-1 proxy), so the pool can reach back before every asset
    existed without understating young stocks' volatility to zero.
    """
    rets = adj[series].pct_change(fill_method=None)
    log_vix = np.log(adj[VIX])
    d_log_vix = log_vix.diff()
    market = rets[MARKET_SERIES]
    keep = (rets.index >= pool_start) & market.notna().all(axis=1) & d_log_vix.notna() & log_vix.shift(1).notna()
    pool = rets[keep].apply(lambda col: col.fillna(rets.loc[keep, BENCHMARK]))

    # d log VIX_t = a + b * log VIX_{t-1} + e_t   (b < 0: mean reversion)
    lv, dlv = log_vix.shift(1)[keep], d_log_vix[keep]
    b, a = np.polyfit(lv, dlv, 1)
    resid = (dlv - (a + b * lv)).to_numpy()
    log.info(
        "Bootstrap pool %s -> %s (%d days); VIX half-life %.0f days, long-run level %.1f",
        pool.index[0].date(), pool.index[-1].date(), len(pool),
        np.log(0.5) / np.log(1 + b), np.exp(-a / b),
    )
    return pool.to_numpy(), resid, (a, b)


def stationary_bootstrap(rng, n_days, n_paths, horizon, mean_block):
    idx = np.empty((n_paths, horizon), dtype=np.int64)
    idx[:, 0] = rng.integers(n_days, size=n_paths)
    for t in range(1, horizon):
        jump = rng.random(n_paths) < 1 / mean_block
        idx[:, t] = np.where(jump, rng.integers(n_days, size=n_paths), (idx[:, t - 1] + 1) % n_days)
    return idx


# ---------------------------------------------------------------------------
# Validation: simulated-feature code must reproduce the real pipeline
# ---------------------------------------------------------------------------

def validate_features(wide, tickers, fundamentals, hmm, n_check=50):
    adj_all = wide["adj_close"]
    pick = lambda df, names: torch.tensor(df[names].iloc[-HISTORY:].to_numpy())[None]
    market = lambda s: pick(adj_all, [s])

    asset = price_features(pick(adj_all, tickers), pick(wide["close"], tickers),
                           pick(wide["volume"], tickers), market(BENCHMARK))
    regime = regime_features(market(BENCHMARK), market("SPY"), market("TLT"), market(VIX))

    real_stock = pd.read_parquet(FEATURE_DIR / "stock_features.parquet")
    real_regime = pd.read_parquet(FEATURE_DIR / "regime_features.parquet").set_index("date")
    dates = adj_all.index[-n_check:]

    errors = {}
    for name, values in asset.items():
        real = real_stock.pivot(index="date", columns="ticker", values=name).reindex(index=dates, columns=tickers)
        errors[name] = np.nanmax(np.abs(values[0, -n_check:].numpy() - real.to_numpy()))
    for name, values in regime.items():
        errors[name] = np.nanmax(np.abs(values[0, -n_check:].numpy() - real_regime[name].reindex(dates).to_numpy()))
    last = real_stock[real_stock["date"] == dates[-1]].set_index("ticker").reindex(tickers)
    for name, values in market_cap_features(pick(wide["close"], tickers)[:, -1:], fundamentals).items():
        errors[name] = np.nanmax(np.abs(values[0, 0].numpy() - last[name].to_numpy()))

    # HMM filter on the full real history vs the probabilities preprocessing saved
    obs = regime_model.observations(adj_all).to_numpy()
    probs = regime_model.filtered_probs(obs, hmm)
    saved = pd.read_parquet(FEATURE_DIR / "hmm_regime.parquet").set_index("date")["hmm_bear_prob"]
    errors["hmm_bear_prob"] = np.nanmax(np.abs(regime_model.bear_prob(probs, hmm) - saved.reindex(adj_all.index).to_numpy()))

    worst = max(errors, key=errors.get)
    log.info("Feature validation vs real pipeline: max abs difference %.2e (%s)", errors[worst], worst)
    if errors[worst] > 1e-6:
        raise RuntimeError(f"Simulated features do not match the real pipeline: {errors}")
    return probs[-1]          # HMM state at the last real day


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate(args):
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    data = load_inputs()
    model, ckpt = load_model()
    scalers = joblib.load(MODEL_DIR / "scalers.joblib")
    hmm = scalers["hmm"]
    tickers = ckpt["tickers"]
    is_safe = ckpt["is_safe"]
    n = len(tickers)
    series = tickers + [s for s in MARKET_SERIES if s not in tickers]
    k_asset = [series.index(s) for s in tickers]
    k_xbi, k_spy, k_tlt, k_rf = (series.index(s) for s in MARKET_SERIES)

    wide = load_market(data["dates"])
    last = last_fundamentals(tickers)
    fundamentals = {
        k: torch.tensor(last[k].to_numpy(dtype=float))
        for k in ("shares_outstanding", "equity", "cash", "short_term_investments")
    }
    held = {
        k: torch.tensor(last[k].to_numpy(dtype=float))
        for k in scalers["stock"]["features"]
        if k not in PRICE_FEATURES + MARKET_CAP_FEATURES
    }
    hmm_state = validate_features(wide, tickers, fundamentals, hmm)

    pool, vix_resid, (a, b) = bootstrap_pool(wide["adj_close"], series, args.pool_start)

    hist_adj = wide["adj_close"][series].iloc[-HISTORY:].to_numpy()
    hist_close = wide["close"][tickers].iloc[-HISTORY:].to_numpy()
    hist_volume = wide["volume"][tickers].iloc[-HISTORY:].to_numpy()
    hist_vix = wide["adj_close"][VIX].iloc[-HISTORY:].to_numpy()
    sim_volume = np.nanmean(hist_volume[-63:], axis=0)

    # Eligibility is held at the last real day
    tradable = data["tradable"][-1]
    log.info("Eligible assets at simulation start: %d of %d", tradable.sum(), n)
    ew = equal_weights(tradable, is_safe)

    # First rebalance uses the real feature history and is the same in every path
    t = to_tensors(data)
    w0 = predict_weights(model, t, np.array([len(data["dates"]) - 1]))[0]

    rebalance_days = list(range(0, HORIZON, HOLD_DAYS))          # 0, 63, 126, 189
    n_feature_days = rebalance_days[-1]                           # sim days 1..189
    pos = slice(HISTORY, HISTORY + n_feature_days)
    sim_events = event_features(data["dates"][-1], n_feature_days)

    results, weight_log = [], {d: [] for d in rebalance_days}
    for start in range(0, args.paths, CHUNK):
        p = min(CHUNK, args.paths - start)
        idx = stationary_bootstrap(rng, len(pool), p, HORIZON, MEAN_BLOCK)
        R = pool[idx]                                             # (p, H, K)

        # Simulated price levels appended to real history
        growth = np.cumprod(1 + R, axis=1)
        adj = np.concatenate([np.broadcast_to(hist_adj, (p, *hist_adj.shape)),
                              hist_adj[-1] * growth], axis=1)
        close = np.concatenate([np.broadcast_to(hist_close, (p, *hist_close.shape)),
                                hist_close[-1] * growth[..., k_asset]], axis=1)
        volume = np.concatenate([np.broadcast_to(hist_volume, (p, *hist_volume.shape)),
                                 np.broadcast_to(sim_volume, (p, HORIZON, n))], axis=1)
        log_vix = np.empty((p, HORIZON))
        prev = np.full(p, np.log(hist_vix[-1]))
        for step in range(HORIZON):
            prev = prev + a + b * prev + vix_resid[idx[:, step]]
            log_vix[:, step] = prev
        vix = np.concatenate([np.broadcast_to(hist_vix, (p, HISTORY)), np.exp(log_vix)], axis=1)

        adj_t, close_t, vol_t = (torch.from_numpy(np.ascontiguousarray(x)) for x in (adj, close, volume))
        vix_t = torch.from_numpy(vix)[..., None]

        raw = {k: v[:, pos] for k, v in price_features(
            adj_t[..., k_asset], close_t, vol_t, adj_t[..., [k_xbi]]).items()}
        raw.update(market_cap_features(close_t[:, pos], fundamentals))
        raw.update({k: v.expand(p, n_feature_days, n) for k, v in held.items()})
        regime_raw = {k: v[:, pos] for k, v in regime_features(
            adj_t[..., [k_xbi]], adj_t[..., [k_spy]], adj_t[..., [k_tlt]], vix_t).items()}

        # HMM filter continues from the real state on each simulated path
        sim_obs = np.stack([np.log1p(R[..., k_xbi]), log_vix], axis=-1)
        sim_probs = regime_model.filtered_probs(sim_obs, hmm, init=np.broadcast_to(hmm_state, (p, 2)))
        regime_raw.update({k: v.expand(p, n_feature_days) for k, v in sim_events.items()})
        regime_raw["hmm_bear_prob"] = torch.from_numpy(regime_model.bear_prob(sim_probs, hmm)[:, :n_feature_days])

        asset_x = scale(raw, scalers["stock"])                    # (p, D, N, Fa)
        regime_x = scale(regime_raw, scalers["regime"])           # (p, D, Fr)
        tradable_t = torch.from_numpy(np.broadcast_to(tradable, (p, n)).copy())

        # Quarterly rebalancing through the simulated year
        rf = R[..., k_rf]
        R_assets = R[..., k_asset]
        port = {"lstm": [], "equal_weight": []}
        prev_w = {k: np.zeros((p, n)) for k in port}
        for d in rebalance_days:
            if d == 0:
                w = np.broadcast_to(w0, (p, n)).copy()
            else:
                win = slice(d - LOOKBACK, d)                      # sim days d-62..d
                a_x = asset_x[:, win]
                r_x = regime_x[:, win][:, :, None, :].expand(-1, -1, n, -1)
                w = predict(model, torch.cat([a_x, r_x], dim=-1), tradable_t)
            weight_log[d].append(w)
            fwd = R_assets[:, d:d + HOLD_DAYS]
            targets = {"lstm": w, "equal_weight": np.broadcast_to(ew, (p, n))}
            for name, target in targets.items():
                daily, prev_w[name] = run_holding_period(target, prev_w[name], fwd)
                port[name].append(daily)

        returns = {k: np.concatenate(v, axis=1) for k, v in port.items()}
        returns["xbi"] = R[..., k_xbi]
        for name, r in returns.items():
            metrics = pd.DataFrame(performance(r, rf))
            metrics.insert(0, "strategy", name)
            metrics.insert(0, "path", np.arange(start, start + p))
            results.append(metrics)
        if (start // CHUNK) % 20 == 0:
            log.info("Simulated %d / %d paths", start + p, args.paths)

    paths = pd.concat(results, ignore_index=True)
    weights = pd.concat({
        f"day_{d}": pd.DataFrame({
            "mean": np.concatenate(ws).mean(0),
            "std": np.concatenate(ws).std(0),
        }, index=tickers)
        for d, ws in weight_log.items()
    }, names=["rebalance", "ticker"])
    safe = {d: np.concatenate(ws)[:, is_safe].sum(1) for d, ws in weight_log.items()}
    log.info("Safe-haven weight by rebalance (mean, 5th-95th pct): %s", {
        d: f"{v.mean():.0%} ({np.quantile(v, 0.05):.0%}-{np.quantile(v, 0.95):.0%})" for d, v in safe.items()
    })
    return paths, weights


def summarize(paths):
    def stats(g):
        r, s = g["ann_return"], g["sharpe"]
        worst = r <= r.quantile(0.05)
        return pd.Series({
            "sharpe_mean": s.mean(),
            "sharpe_median": s.median(),
            "sharpe_p05": s.quantile(0.05),
            "sharpe_p95": s.quantile(0.95),
            "return_mean": r.mean(),
            "return_median": r.median(),
            "prob_loss": (r < 0).mean(),
            "var_95_annual": -r.quantile(0.05),
            "es_95_annual": -r[worst].mean(),
            "max_drawdown_median": g["max_drawdown"].median(),
        })

    summary = paths.groupby("strategy").apply(stats)
    sharpe = paths.pivot(index="path", columns="strategy", values="sharpe")
    summary["prob_lstm_beats"] = {
        k: (sharpe["lstm"] > sharpe[k]).mean() for k in sharpe if k != "lstm"
    }
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--paths", type=int, default=N_PATHS)
    parser.add_argument("--pool-start", default="2016-01-04",
                        help="First day of history to bootstrap from")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    paths, weights = simulate(args)
    summary = summarize(paths)

    paths.to_parquet(RESULTS_DIR / "monte_carlo_paths.parquet", index=False)
    weights.to_parquet(RESULTS_DIR / "monte_carlo_weights.parquet")
    summary.to_csv(RESULTS_DIR / "monte_carlo_summary.csv")
    log.info("Monte Carlo summary (%d paths, %d trading days each):\n%s",
             args.paths, HORIZON, summary.round(3).T.to_string())


if __name__ == "__main__":
    main()
