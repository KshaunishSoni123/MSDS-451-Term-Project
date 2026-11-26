"""
Exploratory data analysis for the biotech portfolio optimizer.

Reads the raw data from ``data_ingestion.py`` and the features from
``feature_engineering.py`` and writes figures and tables to ``data/eda/``:

    feature_summary.csv          Distribution stats + missing share per feature
    price_feature_dist.png       Volatility, momentum, beta, liquidity histograms
    regime_feature_dist.png      Market regime feature histograms
    volatility_over_time.png     Cross-sectional volatility vs XBI vol and VIX
    feature_coverage.png         Share of biotech rows with price features and
                                 eligible, plus tickers priced over time

Distributions use biotech (non safe-haven) rows only, and histograms are
clipped to the 1st/99th percentiles for display so a few extreme values do
not flatten the plot. The summary table uses the unclipped values.

Usage:
    python eda.py
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from feature_engineering import SAFE_HAVENS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
FEATURE_DIR = BASE_DIR / "data" / "features"
EDA_DIR = BASE_DIR / "data" / "eda"

PRICE_FEATURES = [
    "vol_21d", "vol_252d", "vol_ratio_21_252", "downside_vol_63d",
    "beta_xbi_252d", "jump_count_252d", "mom_12_1", "ret_1m",
    "dist_52w_high", "log_dollar_volume_63d",
]
FEATURE_GROUPS = {
    "momentum": ["mom_12_1", "mom_6_1", "ret_1m"],
    "volatility": ["vol_21d", "vol_252d", "downside_vol_63d"],
    "beta": ["beta_xbi_252d"],
}

CLIP_QUANTILES = (0.01, 0.99)

log = logging.getLogger("eda")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def hist_grid(df, features, path, title, ncols=4):
    """One histogram per feature, clipped to the 1st/99th percentiles."""
    features = [f for f in features if f in df]
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    for ax, f in zip(axes.flat, features):
        x = df[f].replace([np.inf, -np.inf], np.nan).dropna()
        if x.empty:
            ax.set_title(f"{f} (no data)")
            continue
        lo, hi = x.quantile(CLIP_QUANTILES)
        ax.hist(x.clip(lo, hi), bins=50, color="steelblue")
        ax.axvline(x.median(), color="darkorange", lw=1.5, label=f"median {x.median():.2f}")
        ax.set_title(f"{f}\n{x.notna().sum():,} obs, {df[f].isna().mean():.0%} missing", fontsize=9)
        ax.legend(fontsize=7)
    for ax in axes.flat[len(features):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def feature_summary(stock, regime):
    """describe() plus missing share and skew for every model feature.

    Infinite values (e.g. log of zero dollar volume) are counted in
    ``n_inf`` and excluded from the other statistics.
    """
    biotech = stock[~stock["ticker"].isin(SAFE_HAVENS)]
    rows = []
    for source, df, features in (
        ("stock", biotech, [c for c in stock.columns
                            if c not in ("date", "ticker", "close", "eligible", "is_safe_haven")]),
        ("regime", regime, [c for c in regime.columns if c != "date"]),
    ):
        for f in features:
            n_inf = np.isinf(df[f]).sum()
            x = df[f].replace([np.inf, -np.inf], np.nan)
            rows.append({
                "source": source, "feature": f, "count": x.notna().sum(), "n_inf": n_inf,
                "missing_pct": 100 * df[f].isna().mean(), "mean": x.mean(), "std": x.std(),
                "min": x.min(), "p01": x.quantile(0.01), "p25": x.quantile(0.25),
                "median": x.median(), "p75": x.quantile(0.75), "p99": x.quantile(0.99),
                "max": x.max(), "skew": x.skew(),
            })
    return pd.DataFrame(rows).round(4)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_volatility_over_time(stock, regime, path):
    biotech = stock[~stock["ticker"].isin(SAFE_HAVENS) & stock["eligible"]]
    q = biotech.groupby("date")["vol_21d"].quantile([0.25, 0.5, 0.75]).unstack()

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.fill_between(q.index, q[0.25], q[0.75], alpha=0.3, label="eligible stocks IQR")
    ax.plot(q.index, q[0.5], lw=1, label="eligible stocks median")
    r = regime.set_index("date")
    ax.plot(r.index, r["xbi_vol_21d"], lw=1, color="black", label="XBI")
    ax.set_ylabel("21d annualized volatility")
    ax2 = ax.twinx()
    ax2.plot(r.index, r["vix"], lw=0.8, color="crimson", alpha=0.6, label="VIX (right)")
    ax2.set_ylabel("VIX")
    lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(lines, labels, loc="upper left", fontsize=8)
    ax.set_title("Single-stock biotech volatility vs the sector (XBI) and VIX")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_feature_coverage(stock, path):
    """Monthly share of biotech rows where each feature group is populated."""
    biotech = stock[~stock["ticker"].isin(SAFE_HAVENS)]
    month = biotech["date"].dt.to_period("M").dt.to_timestamp()
    cov = pd.DataFrame({
        name: biotech[cols].notna().any(axis=1).groupby(month).mean()
        for name, cols in FEATURE_GROUPS.items()
    })
    cov["eligible"] = biotech["eligible"].groupby(month).mean()
    tickers = biotech.groupby(month)["ticker"].nunique()

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    for col in cov:
        ax.plot(cov.index, cov[col], lw=1.5, ls="--" if col == "eligible" else "-", label=col)
    ax.set_ylabel("share of biotech rows")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Feature availability over time (biotech names)")
    ax2.plot(tickers.index, tickers, color="gray")
    ax2.set_ylabel("tickers priced")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 160)

    stock = pd.read_parquet(FEATURE_DIR / "stock_features.parquet")
    regime = pd.read_parquet(FEATURE_DIR / "regime_features.parquet")
    biotech = stock[~stock["ticker"].isin(SAFE_HAVENS)]

    log.info(
        "Stock panel: %d rows, %d biotech tickers, %s -> %s",
        len(stock), biotech["ticker"].nunique(),
        stock["date"].min().date(), stock["date"].max().date(),
    )

    summary = feature_summary(stock, regime)
    summary.to_csv(EDA_DIR / "feature_summary.csv", index=False)
    log.info("Feature summary:\n%s", summary[
        ["source", "feature", "missing_pct", "n_inf", "mean", "median", "std", "skew"]
    ].to_string(index=False))
    for f in summary.loc[summary["n_inf"] > 0, "feature"]:
        bad = biotech.loc[np.isinf(biotech[f]), "ticker"].unique()
        log.warning("%s has infinite values for: %s", f, ", ".join(bad))

    hist_grid(biotech, PRICE_FEATURES, EDA_DIR / "price_feature_dist.png",
              "Price features (biotech rows)")
    hist_grid(regime, [c for c in regime.columns if c != "date"],
              EDA_DIR / "regime_feature_dist.png", "Market regime features", ncols=5)
    plot_volatility_over_time(stock, regime, EDA_DIR / "volatility_over_time.png")
    plot_feature_coverage(stock, EDA_DIR / "feature_coverage.png")

    log.info("Saved EDA outputs -> %s", EDA_DIR)


if __name__ == "__main__":
    main()
