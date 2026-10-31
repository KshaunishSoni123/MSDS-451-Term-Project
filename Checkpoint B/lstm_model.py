"""
Sharpe-optimized LSTM portfolio for the biotech universe.

Reads the scaled datasets from ``preprocessing.py`` and trains an LSTM that
maps the last ``LOOKBACK`` days of features to long-only portfolio weights
over the eligible biotech names plus safe-haven assets (SHY, TLT).

Loss (per decision day t):

    -Sharpe(net of transaction costs, over the next train_hold_days)
    + turnover_penalty      * turnover(w_t, drifted w_{t-HOLD_DAYS})
    + concentration_penalty * normalized HHI(w_t)
    - rank_loss_weight      * corr(log w_t, forward returns)   (cross-sectional)

Turnover is measured against the portfolio the model would have chosen one
rebalance earlier, drifted by realized returns. The rank term rewards ordering
stocks by their future return, a denser signal than portfolio Sharpe alone.
Training on 21-day windows (train_hold_days) is another way to cut target
noise; evaluation always uses the real quarterly (HOLD_DAYS) rebalance.

Architecture: one LSTM is shared across all assets. For each asset it reads
that asset's features plus the market regime features (including the HMM bear
probability) and emits a score; a softmax across eligible assets, capped per
asset (MAX_WEIGHT for stocks, SAFE_MAX_WEIGHT for safe havens), gives the
weights. Each training batch uses a random subset of stocks plus the safe
havens, which speeds training on a large universe and regularizes the model.

The deployed model is an ensemble: N_SEEDS models trained with different
seeds whose weights are averaged, since a single Sharpe-trained network varies
a lot from seed to seed.

Validation uses walk-forward folds inside the training period. The validation
score is the net-of-cost Sharpe of actual quarterly rebalancing over the full
universe, averaged over every possible rebalance phase.

Outputs:
    data/model/lstm.pt                     Ensemble weights + params
    data/results/backtest_returns.parquet  Daily out-of-sample returns
    data/results/backtest_weights.parquet  Weights at each rebalance
    data/results/backtest_metrics.csv      LSTM vs equal weight vs XBI

Usage:
    python lstm_model.py                    # uses best_params.json if present
    python lstm_model.py --default-params   # ignore tuned params
    python lstm_model.py --seeds 10
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from feature_engineering import SAFE_HAVENS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
MODEL_DIR = BASE_DIR / "data" / "model"
RESULTS_DIR = BASE_DIR / "data" / "results"
PARAMS_PATH = MODEL_DIR / "best_params.json"

LOOKBACK = 63          # days of features the LSTM sees
HOLD_DAYS = 63         # quarterly rebalance used for turnover and evaluation
TRADING_DAYS = 252
WARMUP_DAYS = 252      # skip the first year, where 252-day features are empty
RISK_FREE = "SHY"      # cash proxy for excess returns
BENCHMARK = "XBI"
COST_BPS = 10          # transaction cost per unit of turnover
MAX_WEIGHT = 0.05      # per-stock cap (the 5% diversified-fund limit)
SAFE_MAX_WEIGHT = 0.5  # per safe-haven cap; SHY + TLT can reach 100%
N_FOLDS = 3            # walk-forward validation folds
N_SEEDS = 5            # ensemble size
SEED = 451

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

DEFAULT_PARAMS = {
    "hidden": 32,
    "dropout": 0.2,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 32,
    "assets_per_batch": 64,       # stocks sampled per batch (safe havens always in)
    "min_epochs": 5,              # early stopping can't select earlier epochs
    "max_weight": MAX_WEIGHT,
    "turnover_penalty": 0.0,
    "concentration_penalty": 0.0,
    "rank_loss_weight": 0.0,
    "train_hold_days": HOLD_DAYS,
}

log = logging.getLogger("lstm_model")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_inputs():
    """Assemble aligned arrays for the full (train + test) period.

    Returns a dict with:
        dates      (T,) trading days
        tickers    (N,) asset order (biotech names and safe havens)
        is_safe    (N,) asset is a safe haven
        asset_x    (T, N, Fa) scaled asset features, NaN -> 0 (the train mean)
        regime_x   (T, Fr) scaled regime features
        returns    (T, N) daily return realized on day t (NaN before listing)
        rf         (T,) daily risk-free return
        bench      (T,) daily benchmark return
        tradable   (T, N) asset is priced and point-in-time eligible on day t
        cutoff_idx index of the first test date
    """
    stock = pd.concat([pd.read_parquet(MODEL_DIR / f"stock_{s}.parquet") for s in ("train", "test")])
    regime = pd.concat([pd.read_parquet(MODEL_DIR / f"regime_{s}.parquet") for s in ("train", "test")])
    prices = pd.read_parquet(RAW_DIR / "prices.parquet")
    prices["date"] = prices["date"].astype("datetime64[ns]")
    test_start = pd.read_parquet(MODEL_DIR / "stock_test.parquet", columns=["date"])["date"].min()

    tickers = sorted(stock["ticker"].unique())
    dates = pd.DatetimeIndex(sorted(stock["date"].unique()))
    asset_features = [c for c in stock.columns if c not in ("date", "ticker", "eligible")]
    regime_features = [c for c in regime.columns if c != "date"]

    grid = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    panel = stock.set_index(["date", "ticker"]).reindex(grid)
    asset_x = panel[asset_features].to_numpy().reshape(len(dates), len(tickers), len(asset_features))
    eligible = panel["eligible"].fillna(False).astype(bool).to_numpy().reshape(len(dates), len(tickers))
    regime_x = regime.set_index("date").reindex(dates).ffill()[regime_features].to_numpy()

    px = prices.pivot(index="date", columns="ticker", values="adj_close").reindex(dates)
    rets = px.pct_change(fill_method=None)

    return {
        "dates": dates,
        "tickers": tickers,
        "is_safe": np.array([tk in SAFE_HAVENS for tk in tickers]),
        "asset_features": asset_features,
        "regime_features": regime_features,
        "asset_x": np.nan_to_num(asset_x).astype(np.float32),
        "regime_x": np.nan_to_num(regime_x).astype(np.float32),
        "returns": rets[tickers].to_numpy(),
        "rf": rets[RISK_FREE].fillna(0).to_numpy(),
        "bench": rets[BENCHMARK].fillna(0).to_numpy(),
        "tradable": eligible & px[tickers].notna().to_numpy(),
        "cutoff_idx": int(dates.searchsorted(test_start)),
    }


def to_tensors(data):
    return {
        "asset_x": torch.from_numpy(data["asset_x"]).to(DEVICE),
        "regime_x": torch.from_numpy(data["regime_x"]).to(DEVICE),
        "returns": torch.from_numpy(np.nan_to_num(data["returns"]).astype(np.float32)).to(DEVICE),
        "rf": torch.from_numpy(data["rf"].astype(np.float32)).to(DEVICE),
        "tradable": torch.from_numpy(data["tradable"].copy()).to(DEVICE),
        "is_safe": torch.from_numpy(data["is_safe"]).to(DEVICE),
    }


def asset_caps(data, max_weight):
    return torch.where(
        torch.from_numpy(data["is_safe"]), SAFE_MAX_WEIGHT, max_weight
    ).float().to(DEVICE)


def window_inputs(asset_x, regime_x, idx, assets=None, lookback=LOOKBACK):
    """Model input for decisions at day indices ``idx``: (B, L, N, Fa + Fr).

    ``assets`` optionally selects a subset of assets.
    """
    win = idx[:, None] + torch.arange(-lookback + 1, 1, device=idx.device)
    a = asset_x[win] if assets is None else asset_x[win[..., None], assets]
    r = regime_x[win][:, :, None, :].expand(-1, -1, a.shape[2], -1)
    return torch.cat([a, r], dim=-1)


def decision_days(data):
    """Training-period days whose holding window ends before the test period.

    Starts late enough that the previous rebalance (t - HOLD_DAYS) also has a
    full year of features, since the turnover penalty needs it.
    """
    first = WARMUP_DAYS + HOLD_DAYS
    last = data["cutoff_idx"] - HOLD_DAYS - 1
    return np.arange(first, last + 1)


def make_folds(days, n_folds=N_FOLDS):
    """Expanding-window walk-forward folds over ``n_folds + 2`` equal chunks.

    Fold k trains on chunks [0, k+2) and validates on chunk k+2. The last
    HOLD_DAYS training days are dropped so no training holding window overlaps
    the validation period. The final fold is also the final training split.
    """
    chunks = np.array_split(days, n_folds + 2)
    return [
        (np.concatenate(chunks[:k])[:-HOLD_DAYS], chunks[k])
        for k in range(2, n_folds + 2)
    ]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def cap_weights(weights, caps, iters=30):
    """Enforce per-asset caps (broadcastable to weights), redistributing the
    excess pro rata to uncapped assets. Differentiable, so training sees the
    same constraint as deployment."""
    w = weights
    for _ in range(iters):
        excess = (w - caps).clamp(min=0).sum(1, keepdim=True)
        if excess.max() < 1e-6:
            break
        w = torch.minimum(w, caps)
        room = w * (w < caps - 1e-7)
        w = w + excess * room / room.sum(1, keepdim=True).clamp_min(1e-12)
    return w


class PortfolioLSTM(nn.Module):
    def __init__(self, n_features, hidden=32, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x, tradable, caps):
        """x: (B, L, N, F), tradable: (B, N) bool, caps: (N,) -> weights (B, N)."""
        b, l, n, f = x.shape
        seq = x.permute(0, 2, 1, 3).reshape(b * n, l, f)
        _, (h, _) = self.lstm(seq)
        scores = self.head(h[-1]).view(b, n)
        weights = torch.softmax(scores.masked_fill(~tradable, float("-inf")), dim=1)
        return cap_weights(weights, caps)


class Ensemble(nn.Module):
    """Averages the weights of several models (still within every cap)."""

    def __init__(self, models, caps):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.register_buffer("caps", caps)

    def forward(self, x, tradable, caps=None):
        caps = self.caps if caps is None else caps
        return torch.stack([m(x, tradable, caps) for m in self.models]).mean(0)


def predict(model, x, tradable, caps=None):
    """Weights as float64 numpy for inputs on any device."""
    with torch.no_grad():
        args = (x.to(DEVICE), tradable.to(DEVICE))
        w = model(*args) if caps is None else model(*args, caps)
    return w.cpu().double().numpy()


def predict_weights(model, t, days, caps=None, chunk=16):
    model.eval()
    out = []
    for i in range(0, len(days), chunk):
        idx = torch.as_tensor(days[i:i + chunk], device=DEVICE)
        x = window_inputs(t["asset_x"], t["regime_x"], idx)
        out.append(predict(model, x, t["tradable"][idx], caps))
    return np.concatenate(out)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def holding_returns(weights, fwd_returns):
    """Daily returns of a buy-and-hold portfolio.

    weights (B, N) set at the start, fwd_returns (B, H, N) -> (B, H).
    """
    growth = torch.cumprod(1 + fwd_returns, dim=1)
    value = (weights[:, None, :] * growth).sum(-1)
    prev = torch.cat([torch.ones_like(value[:, :1]), value[:, :-1]], dim=1)
    return value / prev - 1


def sharpe_ratio(port_returns, rf):
    """Annualized Sharpe ratio per row of daily returns (B, H) -> (B,)."""
    excess = port_returns - rf
    return excess.mean(1) / (excess.std(1) + 1e-6) * np.sqrt(TRADING_DAYS)


def turnover(weights, prev_weights, past_returns):
    """Trade size sum|w - w_prev drifted| at a rebalance (B,)."""
    drifted = prev_weights * torch.prod(1 + past_returns, dim=1)
    drifted = drifted / drifted.sum(1, keepdim=True)
    return (weights - drifted).abs().sum(1)


def concentration(weights, tradable):
    """Normalized HHI: 0 for equal weight, 1 for a single asset (B,)."""
    n = tradable.sum(1).clamp(min=2)
    return ((weights ** 2).sum(1) - 1 / n) / (1 - 1 / n)


def rank_correlation(weights, fwd_returns, mask):
    """Cross-sectional Pearson correlation of log weights with forward
    returns over ``mask`` assets (B,): a differentiable proxy for rank IC."""
    m = mask.float()
    n = m.sum(1).clamp(min=2)
    x = torch.where(mask, torch.log(weights.clamp_min(1e-8)), 0.0)
    y = torch.where(mask, torch.prod(1 + fwd_returns, dim=1) - 1, 0.0)
    xc = (x - (x * m).sum(1, keepdim=True) / n[:, None]) * m
    yc = (y - (y * m).sum(1, keepdim=True) / n[:, None]) * m
    # eps inside the sqrt: its gradient at 0 is infinite (single-stock days)
    return (xc * yc).sum(1) / (xc.pow(2).sum(1) * yc.pow(2).sum(1) + 1e-12).sqrt()


def batch(t, idx, assets, hold):
    prev = idx - HOLD_DAYS
    fwd = idx[:, None] + torch.arange(1, hold + 1, device=DEVICE)
    past = idx[:, None] + torch.arange(-HOLD_DAYS + 1, 1, device=DEVICE)   # (t-H, t]
    return {
        "x": window_inputs(t["asset_x"], t["regime_x"], idx, assets),
        "tradable": t["tradable"][idx][:, assets],
        "x_prev": window_inputs(t["asset_x"], t["regime_x"], prev, assets),
        "tradable_prev": t["tradable"][prev][:, assets],
        "fwd": t["returns"][fwd][:, :, assets],
        "rf": t["rf"][fwd],
        "past": t["returns"][past][:, :, assets],
        "is_safe": t["is_safe"][assets],
    }


def regularized_loss(model, b, params, caps):
    w = model(b["x"], b["tradable"], caps)
    with torch.no_grad():
        w_prev = model(b["x_prev"], b["tradable_prev"], caps)
    trade = turnover(w, w_prev, b["past"])

    port = holding_returns(w, b["fwd"])
    cost = trade * COST_BPS / 1e4
    port = torch.cat([port[:, :1] - cost[:, None], port[:, 1:]], dim=1)
    sharpe = sharpe_ratio(port, b["rf"]).mean()
    conc = concentration(w, b["tradable"]).mean()
    rank = rank_correlation(w, b["fwd"], b["tradable"] & ~b["is_safe"]).mean()

    loss = (
        -sharpe
        + params["turnover_penalty"] * trade.mean()
        + params["concentration_penalty"] * conc
        - params["rank_loss_weight"] * rank
    )
    return loss, {"sharpe": sharpe.item(), "turnover": trade.mean().item(), "rank": rank.item()}


# ---------------------------------------------------------------------------
# Validation: net Sharpe of real quarterly rebalancing
# ---------------------------------------------------------------------------

def evaluate(model, data, t, days, caps=None, weights=None):
    """Net-of-cost Sharpe of quarterly rebalancing over ``days``.

    Averaged over every rebalance phase (start offset 0..HOLD_DAYS-1). The
    first rebalance of each phase is treated as already held, so costs only
    reflect turnover the model chooses. ``weights`` (len(days), N) can be
    passed instead of a model to score any weighting rule.
    """
    if weights is None:
        weights = predict_weights(model, t, days, caps)
    returns = np.nan_to_num(data["returns"])
    sharpes, trades = [], []
    for offset in range(HOLD_DAYS):
        reb = np.arange(offset, len(days), HOLD_DAYS)
        if len(reb) < 2:
            continue
        prev, daily = weights[reb[0]], []
        for j in reb:
            d = days[j]
            if j != reb[0]:
                trades.append(np.abs(weights[j] - prev).sum())
            r, prev = run_holding_period(weights[j], prev, returns[d + 1: d + HOLD_DAYS + 1])
            daily.append(r)
        daily = np.concatenate(daily)
        start = days[reb[0]] + 1
        excess = daily - data["rf"][start: start + len(daily)]
        sharpes.append(excess.mean() / excess.std(ddof=1) * np.sqrt(TRADING_DAYS))

    return {
        "sharpe": float(np.mean(sharpes)),
        "turnover": float(np.mean(trades)),
        "max_weight": float(weights[:, ~data["is_safe"]].max(1).mean()),
        "safe_weight": float(weights[:, data["is_safe"]].sum(1).mean()),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def sample_assets(rng, data, idx, k):
    """Random stocks eligible somewhere in the batch, plus all safe havens."""
    candidates = np.flatnonzero(data["tradable"][idx].any(0) & ~data["is_safe"])
    picked = rng.choice(candidates, size=min(k, len(candidates)), replace=False)
    return torch.as_tensor(np.sort(np.concatenate([picked, np.flatnonzero(data["is_safe"])])), device=DEVICE)


def train(data, params, train_days, val_days, epochs=60, patience=10, seed=SEED,
          verbose=True, t=None):
    """Train one model with early stopping on validation net Sharpe.

    Returns the best model and its validation metrics.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    t = to_tensors(data) if t is None else t
    caps = asset_caps(data, params["max_weight"])
    min_epochs = min(params["min_epochs"], epochs)

    n_features = t["asset_x"].shape[-1] + t["regime_x"].shape[-1]
    model = PortfolioLSTM(n_features, hidden=params["hidden"], dropout=params["dropout"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])

    best, best_state, stale = None, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(train_days)
        stats = []
        for i in range(0, len(perm), params["batch_size"]):
            idx_np = perm[i:i + params["batch_size"]]
            assets = sample_assets(rng, data, idx_np, params["assets_per_batch"])
            b = batch(t, torch.as_tensor(idx_np, device=DEVICE), assets, params["train_hold_days"])
            loss, s = regularized_loss(model, b, params, caps[assets])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            stats.append(s)

        val = evaluate(model, data, t, val_days, caps)
        if verbose:
            tr = pd.DataFrame(stats).mean()
            log.info(
                "seed %d epoch %3d  train Sharpe %6.3f turnover %.3f rank %+.3f | "
                "val net Sharpe %6.3f turnover %.3f safe %.2f",
                seed, epoch, tr["sharpe"], tr["turnover"], tr["rank"],
                val["sharpe"], val["turnover"], val["safe_weight"],
            )
        if epoch < min_epochs:
            continue
        if best is None or val["sharpe"] > best["sharpe"]:
            best, stale = {**val, "epoch": epoch}, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    return model.eval(), best


def train_ensemble(data, params, train_days, val_days, n_seeds=N_SEEDS, epochs=60,
                   patience=10, verbose=True):
    t = to_tensors(data)
    models, bests = [], []
    for k in range(n_seeds):
        model, best = train(data, params, train_days, val_days, epochs, patience,
                            seed=SEED + k, verbose=verbose, t=t)
        log.info("seed %d: best epoch %d, val net Sharpe %.3f", SEED + k, best["epoch"], best["sharpe"])
        models.append(model)
        bests.append(best)
    ensemble = Ensemble(models, asset_caps(data, params["max_weight"])).to(DEVICE).eval()
    return ensemble, bests, t


def load_params(use_default=False):
    if not use_default and PARAMS_PATH.exists():
        params = {**DEFAULT_PARAMS, **json.loads(PARAMS_PATH.read_text())}
        log.info("Using tuned params from %s", PARAMS_PATH)
    else:
        params = dict(DEFAULT_PARAMS)
        log.info("Using default params")
    log.info("%s", params)
    return params


def save_model(ensemble, data, params):
    torch.save({
        "state_dicts": [{k: v.cpu() for k, v in m.state_dict().items()} for m in ensemble.models],
        "n_features": ensemble.models[0].lstm.input_size,
        "params": params,
        "tickers": data["tickers"],
        "is_safe": data["is_safe"],
        "asset_features": data["asset_features"],
        "regime_features": data["regime_features"],
    }, MODEL_DIR / "lstm.pt")


def load_model():
    ckpt = torch.load(MODEL_DIR / "lstm.pt", weights_only=False)
    p = ckpt["params"]
    models = []
    for state in ckpt["state_dicts"]:
        m = PortfolioLSTM(ckpt["n_features"], hidden=p["hidden"], dropout=p["dropout"])
        m.load_state_dict(state)
        models.append(m)
    caps = torch.where(torch.from_numpy(ckpt["is_safe"]), SAFE_MAX_WEIGHT, p["max_weight"]).float()
    return Ensemble(models, caps).to(DEVICE).eval(), ckpt


# ---------------------------------------------------------------------------
# Backtest and metrics
# ---------------------------------------------------------------------------

def run_holding_period(weights, prev_drifted, fwd_returns, cost_bps=COST_BPS):
    """Buy-and-hold returns with transaction cost charged on day one.

    weights (N,) or (P, N); prev_drifted is the portfolio being replaced.
    Returns (daily returns, end-of-period drifted weights).
    """
    w = weights if weights.ndim == 2 else weights[None]
    prev = prev_drifted if prev_drifted.ndim == 2 else prev_drifted[None]
    fwd = fwd_returns if fwd_returns.ndim == 3 else fwd_returns[None]
    growth = np.cumprod(1 + fwd, axis=1)
    value = (w[:, None, :] * growth).sum(-1)
    daily = value / np.concatenate([np.ones_like(value[:, :1]), value[:, :-1]], axis=1) - 1
    daily[:, 0] -= np.abs(w - prev).sum(1) * cost_bps / 1e4
    drifted = w * growth[:, -1] / value[:, -1:]
    if weights.ndim == 1:
        return daily[0], drifted[0]
    return daily, drifted


def equal_weights(tradable, is_safe):
    """Equal weight over tradable biotech names (the benchmark excludes bonds)."""
    w = (tradable & ~is_safe).astype(float)
    return w / w.sum(-1, keepdims=True)


def performance(returns, rf):
    """Summary statistics for daily returns along the last axis."""
    returns = np.asarray(returns)
    n = returns.shape[-1]
    wealth = np.cumprod(1 + returns, axis=-1)
    excess = returns - rf
    q05 = np.quantile(returns, 0.05, axis=-1, keepdims=True)
    tail = np.where(returns <= q05, returns, np.nan)
    return {
        "ann_return": wealth[..., -1] ** (TRADING_DAYS / n) - 1,
        "ann_vol": returns.std(-1, ddof=1) * np.sqrt(TRADING_DAYS),
        "sharpe": excess.mean(-1) / excess.std(-1, ddof=1) * np.sqrt(TRADING_DAYS),
        "max_drawdown": (wealth / np.maximum.accumulate(wealth, axis=-1) - 1).min(-1),
        "var_95_daily": -q05[..., 0],
        "es_95_daily": -np.nanmean(tail, axis=-1),
    }


def backtest(model, data, t=None):
    """Quarterly-rebalanced out-of-sample backtest over the test period."""
    t = to_tensors(data) if t is None else t
    returns = np.nan_to_num(data["returns"])
    start, end = data["cutoff_idx"] - 1, len(data["dates"]) - 1
    n = len(data["tickers"])

    series = {"lstm": [], "equal_weight": []}
    prev = {k: np.zeros(n) for k in series}
    weight_rows, turnover_rows = [], []
    for d in range(start, end, HOLD_DAYS):
        fwd = returns[d + 1: min(d + HOLD_DAYS, end) + 1]
        w_lstm = predict_weights(model, t, np.array([d]))[0]
        targets = {"lstm": w_lstm, "equal_weight": equal_weights(data["tradable"][d], data["is_safe"])}
        turnover_rows.append(np.abs(w_lstm - prev["lstm"]).sum())
        for name, w in targets.items():
            daily, prev[name] = run_holding_period(w, prev[name], fwd)
            series[name].append(daily)
        weight_rows.append(pd.Series(w_lstm, index=data["tickers"], name=data["dates"][d]))

    dates = data["dates"][start + 1: end + 1]
    daily = pd.DataFrame({k: np.concatenate(v) for k, v in series.items()}, index=dates)
    daily["xbi"] = data["bench"][start + 1: end + 1]
    rf = data["rf"][start + 1: end + 1]
    metrics = pd.DataFrame({c: performance(daily[c].to_numpy(), rf) for c in daily}).T

    weights = pd.DataFrame(weight_rows)
    stocks = weights.loc[:, ~data["is_safe"]]
    # Turnover excludes the initial purchase from cash
    metrics.loc["lstm", "avg_turnover"] = np.mean(turnover_rows[1:])
    metrics.loc["lstm", "avg_max_weight"] = stocks.max(axis=1).mean()
    metrics.loc["lstm", "avg_safe_weight"] = weights.loc[:, data["is_safe"]].sum(axis=1).mean()
    metrics.loc["lstm", "avg_holdings_over_1pct"] = (stocks > 0.01).sum(axis=1).mean()
    return daily, weights, metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=N_SEEDS, help="Ensemble size")
    parser.add_argument("--default-params", action="store_true",
                        help="Ignore best_params.json and use DEFAULT_PARAMS")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_inputs()
    params = load_params(args.default_params)
    log.info("%d assets (%d safe havens), %d asset features, %d regime features, %d days, device %s",
             len(data["tickers"]), data["is_safe"].sum(), len(data["asset_features"]),
             len(data["regime_features"]), len(data["dates"]), DEVICE)

    # Final model: train on all but the last walk-forward chunk, stop on it
    train_days, val_days = make_folds(decision_days(data))[-1]
    log.info("Train %s -> %s (%d days) | validation %s -> %s (%d days)",
             data["dates"][train_days[0]].date(), data["dates"][train_days[-1]].date(), len(train_days),
             data["dates"][val_days[0]].date(), data["dates"][val_days[-1]].date(), len(val_days))
    ensemble, bests, t = train_ensemble(data, params, train_days, val_days, args.seeds,
                                        args.epochs, args.patience)
    val = evaluate(ensemble, data, t, val_days)
    log.info("Ensemble of %d: val net Sharpe %.3f (single models %.3f ± %.3f)",
             args.seeds, val["sharpe"], np.mean([b["sharpe"] for b in bests]),
             np.std([b["sharpe"] for b in bests]))
    save_model(ensemble, data, params)

    daily, weights, metrics = backtest(ensemble, data, t)
    daily.to_parquet(RESULTS_DIR / "backtest_returns.parquet")
    weights.to_parquet(RESULTS_DIR / "backtest_weights.parquet")
    metrics.to_csv(RESULTS_DIR / "backtest_metrics.csv")

    log.info("Out-of-sample backtest %s -> %s:\n%s",
             daily.index[0].date(), daily.index[-1].date(), metrics.round(3).to_string())
    top = weights.apply(lambda r: ", ".join(f"{k} {v:.0%}" for k, v in r.nlargest(6).items()), axis=1)
    log.info("Largest LSTM positions at each rebalance:\n%s", top.to_string())


if __name__ == "__main__":
    main()
