"""
Sharpe-optimized LSTM portfolio for the biotech universe.

Reads the scaled datasets from ``preprocessing.py`` and trains an LSTM that
maps the last ``LOOKBACK`` days of features to long-only portfolio weights.

Loss (per decision day t, holding for ``HOLD_DAYS`` trading days):

    -Sharpe(net of transaction costs)
    + turnover_penalty      * turnover(w_t, drifted w_{t-HOLD_DAYS})
    + concentration_penalty * normalized HHI(w_t)

Turnover is measured against the portfolio the model would have chosen one
rebalance earlier, drifted by realized returns, i.e. the trade it would
actually have to make. Normalized HHI is 0 for equal weight and 1 for a single
stock. Penalty strengths and other hyperparameters are chosen by
``tune_lstm.py`` and read from ``data/model/best_params.json`` if present.

Architecture: one LSTM is shared across all assets. For each asset it reads
that asset's features plus the market regime features and emits a score; a
softmax across assets (optionally capped per asset) turns the scores into
weights. Sharing weights across assets gives the model N times more training
sequences than a single LSTM over all assets' features at once.

Validation uses walk-forward folds inside the training period. The validation
score is the net-of-cost Sharpe of actual quarterly rebalancing, averaged over
every possible rebalance phase so it does not hinge on one set of dates.

Outputs:
    data/model/lstm.pt                     Trained weights + params
    data/results/backtest_returns.parquet  Daily out-of-sample returns
    data/results/backtest_weights.parquet  Weights at each rebalance
    data/results/backtest_metrics.csv      LSTM vs equal weight vs XBI

Usage:
    python lstm_model.py                    # uses best_params.json if present
    python lstm_model.py --default-params   # ignore tuned params
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
MODEL_DIR = BASE_DIR / "data" / "model"
RESULTS_DIR = BASE_DIR / "data" / "results"
PARAMS_PATH = MODEL_DIR / "best_params.json"

LOOKBACK = 63          # days of features the LSTM sees
HOLD_DAYS = 63         # quarterly rebalance; Sharpe is measured over this window
TRADING_DAYS = 252
WARMUP_DAYS = 252      # skip the first year, where 252-day features are empty
RISK_FREE = "SHY"      # cash proxy for excess returns
BENCHMARK = "XBI"
COST_BPS = 10          # transaction cost per unit of turnover
N_FOLDS = 3            # walk-forward validation folds
SEED = 451

DEFAULT_PARAMS = {
    "hidden": 32,
    "dropout": 0.2,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 64,
    "max_weight": 0.15,           # hard single-name cap (1.0 disables it)
    "turnover_penalty": 0.0,
    "concentration_penalty": 0.0,
}

log = logging.getLogger("lstm_model")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_inputs():
    """Assemble aligned arrays for the full (train + test) period.

    Returns a dict with:
        dates      (T,) trading days
        tickers    (N,) asset order
        asset_x    (T, N, Fa) scaled asset features, NaN -> 0 (the train mean)
        regime_x   (T, Fr) scaled regime features
        returns    (T, N) daily return realized on day t (NaN before listing)
        rf         (T,) daily risk-free return
        bench      (T,) daily benchmark return
        tradable   (T, N) asset has a price on day t
        cutoff_idx index of the first test date
    """
    stock = pd.concat([pd.read_parquet(MODEL_DIR / f"stock_{s}.parquet") for s in ("train", "test")])
    regime = pd.concat([pd.read_parquet(MODEL_DIR / f"regime_{s}.parquet") for s in ("train", "test")])
    prices = pd.read_parquet(RAW_DIR / "prices.parquet")
    prices["date"] = prices["date"].astype("datetime64[ns]")
    test_start = pd.read_parquet(MODEL_DIR / "stock_test.parquet", columns=["date"])["date"].min()

    tickers = sorted(stock["ticker"].unique())
    dates = pd.DatetimeIndex(sorted(stock["date"].unique()))
    asset_features = [c for c in stock.columns if c not in ("date", "ticker")]
    regime_features = [c for c in regime.columns if c != "date"]

    grid = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    asset_x = stock.set_index(["date", "ticker"]).reindex(grid)[asset_features].to_numpy()
    asset_x = asset_x.reshape(len(dates), len(tickers), len(asset_features))
    regime_x = regime.set_index("date").reindex(dates).ffill()[regime_features].to_numpy()

    px = prices.pivot(index="date", columns="ticker", values="adj_close").reindex(dates)
    rets = px.pct_change(fill_method=None)

    return {
        "dates": dates,
        "tickers": tickers,
        "asset_features": asset_features,
        "regime_features": regime_features,
        "asset_x": np.nan_to_num(asset_x).astype(np.float32),
        "regime_x": np.nan_to_num(regime_x).astype(np.float32),
        "returns": rets[tickers].to_numpy(),
        "rf": rets[RISK_FREE].fillna(0).to_numpy(),
        "bench": rets[BENCHMARK].fillna(0).to_numpy(),
        "tradable": px[tickers].notna().to_numpy(),
        "cutoff_idx": int(dates.searchsorted(test_start)),
    }


def to_tensors(data):
    return {
        "asset_x": torch.from_numpy(data["asset_x"]),
        "regime_x": torch.from_numpy(data["regime_x"]),
        "returns": torch.from_numpy(np.nan_to_num(data["returns"]).astype(np.float32)),
        "rf": torch.from_numpy(data["rf"].astype(np.float32)),
        "tradable": torch.from_numpy(data["tradable"].copy()),
    }


def window_inputs(asset_x, regime_x, idx, lookback=LOOKBACK):
    """Model input for decisions at day indices ``idx``: (B, L, N, Fa + Fr)."""
    win = idx[:, None] + torch.arange(-lookback + 1, 1)
    a = asset_x[win]                                     # (B, L, N, Fa)
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
# Model and loss
# ---------------------------------------------------------------------------

def cap_weights(weights, max_weight, iters=20):
    """Enforce a per-asset cap, redistributing the excess pro rata to the
    uncapped assets. Differentiable, so training sees the same constraint."""
    w = weights
    for _ in range(iters):
        excess = (w - max_weight).clamp(min=0).sum(1, keepdim=True)
        if excess.max() < 1e-7:
            break
        w = w.clamp(max=max_weight)
        room = w * (w < max_weight - 1e-7)
        w = w + excess * room / room.sum(1, keepdim=True).clamp_min(1e-12)
    return w


class PortfolioLSTM(nn.Module):
    def __init__(self, n_features, hidden=32, dropout=0.2, max_weight=1.0):
        super().__init__()
        self.max_weight = max_weight
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x, tradable):
        """x: (B, L, N, F), tradable: (B, N) bool -> weights (B, N)."""
        b, l, n, f = x.shape
        seq = x.permute(0, 2, 1, 3).reshape(b * n, l, f)
        _, (h, _) = self.lstm(seq)
        scores = self.head(h[-1]).view(b, n)
        weights = torch.softmax(scores.masked_fill(~tradable, float("-inf")), dim=1)
        return cap_weights(weights, self.max_weight) if self.max_weight < 1 else weights


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


def batch(t, idx):
    prev = idx - HOLD_DAYS
    fwd = idx[:, None] + torch.arange(1, HOLD_DAYS + 1)
    past = idx[:, None] + torch.arange(-HOLD_DAYS + 1, 1)   # returns over (t-H, t]
    return {
        "x": window_inputs(t["asset_x"], t["regime_x"], idx),
        "tradable": t["tradable"][idx],
        "x_prev": window_inputs(t["asset_x"], t["regime_x"], prev),
        "tradable_prev": t["tradable"][prev],
        "fwd": t["returns"][fwd],
        "rf": t["rf"][fwd],
        "past": t["returns"][past],
    }


def regularized_loss(model, b, params):
    w = model(b["x"], b["tradable"])
    w_prev = model(b["x_prev"], b["tradable_prev"])
    trade = turnover(w, w_prev, b["past"])

    port = holding_returns(w, b["fwd"])
    cost = trade * COST_BPS / 1e4
    port = torch.cat([port[:, :1] - cost[:, None], port[:, 1:]], dim=1)
    sharpe = sharpe_ratio(port, b["rf"]).mean()
    conc = concentration(w, b["tradable"]).mean()

    loss = (
        -sharpe
        + params["turnover_penalty"] * trade.mean()
        + params["concentration_penalty"] * conc
    )
    return loss, {"sharpe": sharpe.item(), "turnover": trade.mean().item(), "hhi": conc.item()}


# ---------------------------------------------------------------------------
# Validation: net Sharpe of real quarterly rebalancing
# ---------------------------------------------------------------------------

def predict_weights(model, t, days, chunk=256):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(days), chunk):
            idx = torch.from_numpy(days[i:i + chunk])
            x = window_inputs(t["asset_x"], t["regime_x"], idx)
            out.append(model(x, t["tradable"][idx]).double().numpy())
    return np.concatenate(out)


def evaluate(model, data, t, days):
    """Net-of-cost Sharpe of quarterly rebalancing over ``days``.

    Averaged over every rebalance phase (start offset 0..HOLD_DAYS-1). The
    first rebalance of each phase is treated as already held, so costs only
    reflect turnover the model chooses.
    """
    weights = predict_weights(model, t, days)
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

    n = data["tradable"][days].sum(1)
    hhi = ((weights ** 2).sum(1) - 1 / n) / (1 - 1 / n)
    return {
        "sharpe": float(np.mean(sharpes)),
        "turnover": float(np.mean(trades)),
        "hhi": float(hhi.mean()),
        "max_weight": float(weights.max(1).mean()),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(data, params, train_days, val_days, epochs=60, patience=10, seed=SEED, verbose=True):
    """Train with early stopping on validation net Sharpe.

    Returns the best model and its validation metrics.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    t = to_tensors(data)

    n_features = t["asset_x"].shape[-1] + t["regime_x"].shape[-1]
    model = PortfolioLSTM(n_features, hidden=params["hidden"], dropout=params["dropout"],
                          max_weight=params["max_weight"])
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])

    best, best_state, stale = None, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(train_days)
        stats = []
        for i in range(0, len(perm), params["batch_size"]):
            loss, s = regularized_loss(model, batch(t, torch.from_numpy(perm[i:i + params["batch_size"]])), params)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            stats.append(s)

        val = evaluate(model, data, t, val_days)
        if verbose:
            tr = pd.DataFrame(stats).mean()
            log.info(
                "epoch %3d  train Sharpe %6.3f turnover %.3f hhi %.3f | "
                "val net Sharpe %6.3f turnover %.3f hhi %.3f",
                epoch, tr["sharpe"], tr["turnover"], tr["hhi"],
                val["sharpe"], val["turnover"], val["hhi"],
            )
        if best is None or val["sharpe"] > best["sharpe"]:
            best, stale = {**val, "epoch": epoch}, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    return model.eval(), best


def load_params(use_default=False):
    if not use_default and PARAMS_PATH.exists():
        params = {**DEFAULT_PARAMS, **json.loads(PARAMS_PATH.read_text())}
        log.info("Using tuned params from %s", PARAMS_PATH)
    else:
        params = dict(DEFAULT_PARAMS)
        log.info("Using default params")
    log.info("%s", params)
    return params


def save_model(model, data, params):
    torch.save({
        "state_dict": model.state_dict(),
        "n_features": model.lstm.input_size,
        "params": params,
        "tickers": data["tickers"],
        "asset_features": data["asset_features"],
        "regime_features": data["regime_features"],
    }, MODEL_DIR / "lstm.pt")


def load_model():
    ckpt = torch.load(MODEL_DIR / "lstm.pt", weights_only=False)
    p = ckpt["params"]
    model = PortfolioLSTM(ckpt["n_features"], hidden=p["hidden"], dropout=p["dropout"],
                          max_weight=p["max_weight"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt


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


def equal_weights(tradable):
    return tradable / tradable.sum(-1, keepdims=True)


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


def backtest(model, data):
    """Quarterly-rebalanced out-of-sample backtest over the test period."""
    t = to_tensors(data)
    returns = np.nan_to_num(data["returns"])
    start, end = data["cutoff_idx"] - 1, len(data["dates"]) - 1
    n = len(data["tickers"])

    series = {"lstm": [], "equal_weight": []}
    prev = {k: np.zeros(n) for k in series}
    weight_rows, turnover_rows = [], []
    for d in range(start, end, HOLD_DAYS):
        fwd = returns[d + 1: min(d + HOLD_DAYS, end) + 1]
        w_lstm = predict_weights(model, t, np.array([d]))[0]
        targets = {"lstm": w_lstm, "equal_weight": equal_weights(data["tradable"][d].astype(float))}
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
    # Turnover excludes the initial purchase from cash
    metrics.loc["lstm", "avg_turnover"] = np.mean(turnover_rows[1:])
    metrics.loc["lstm", "avg_max_weight"] = weights.max(axis=1).mean()
    return daily, weights, metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--default-params", action="store_true",
                        help="Ignore best_params.json and use DEFAULT_PARAMS")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_inputs()
    params = load_params(args.default_params)
    log.info("%d assets, %d asset features, %d regime features, %d days",
             len(data["tickers"]), len(data["asset_features"]),
             len(data["regime_features"]), len(data["dates"]))

    # Final model: train on all but the last walk-forward chunk, stop on it
    train_days, val_days = make_folds(decision_days(data))[-1]
    log.info("Train %s -> %s (%d days) | validation %s -> %s (%d days)",
             data["dates"][train_days[0]].date(), data["dates"][train_days[-1]].date(), len(train_days),
             data["dates"][val_days[0]].date(), data["dates"][val_days[-1]].date(), len(val_days))
    model, best = train(data, params, train_days, val_days, args.epochs, args.patience)
    log.info("Best epoch %d: val net Sharpe %.3f", best["epoch"], best["sharpe"])
    save_model(model, data, params)

    daily, weights, metrics = backtest(model, data)
    daily.to_parquet(RESULTS_DIR / "backtest_returns.parquet")
    weights.to_parquet(RESULTS_DIR / "backtest_weights.parquet")
    metrics.to_csv(RESULTS_DIR / "backtest_metrics.csv")

    log.info("Out-of-sample backtest %s -> %s:\n%s",
             daily.index[0].date(), daily.index[-1].date(), metrics.round(3).to_string())
    log.info("LSTM weights at each rebalance:\n%s", weights.round(2).to_string())


if __name__ == "__main__":
    main()
