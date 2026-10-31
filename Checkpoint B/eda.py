"""
Exploratory data analysis for the biotech portfolio optimizer.

Reads the raw data from ``data_ingestion.py`` and the features from
``feature_engineering.py`` and writes figures and tables to ``data/eda/``:

    feature_summary.csv          Distribution stats + missing share per feature
    sec_coverage.csv             Per-ticker SEC filing counts and metric coverage
    news_coverage.csv            Per-ticker headline counts and mean sentiment
    price_feature_dist.png       Volatility, momentum, beta, liquidity histograms
    fundamental_feature_dist.png Size, value, quality and cash runway histograms
    event_news_feature_dist.png  8-K, earnings and news sentiment histograms
    regime_feature_dist.png      Market regime feature histograms
    volatility_over_time.png     Cross-sectional volatility vs XBI vol and VIX
    feature_coverage.png         Share of biotech rows with each feature group
    sec_availability.png         Filings by year / form and metric coverage
    news_availability.png        Headlines per day / ticker and sentiment

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

from feature_engineering import SAFE_HAVENS, SEC_METRICS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
FEATURE_DIR = BASE_DIR / "data" / "features"
EDA_DIR = BASE_DIR / "data" / "eda"

PRICE_FEATURES = [
    "vol_21d", "vol_252d", "vol_ratio_21_252", "downside_vol_63d",
    "beta_xbi_252d", "jump_count_252d", "mom_12_1", "ret_1m",
    "dist_52w_high", "log_dollar_volume_63d",
]
FUNDAMENTAL_FEATURES = [
    "log_market_cap", "book_to_market", "roe", "net_margin",
    "operating_margin", "cash_to_market_cap", "cash_runway_years",
    "is_cash_flow_positive",
]
EVENT_NEWS_FEATURES = [
    "8k_count_63d", "8k_item801_count_63d", "8k_item101_count_63d",
    "8k_item502_count_252d", "days_to_earnings_est",
    "news_sentiment_5d", "news_sentiment_21d", "news_count_21d",
]
FEATURE_GROUPS = {
    "price": PRICE_FEATURES,
    "fundamentals": FUNDAMENTAL_FEATURES,
    "8-K events": ["8k_count_63d"],
    "earnings": ["days_to_earnings_est"],
    "news": ["news_sentiment_21d"],
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

    Infinite values (e.g. ratios over a zero market cap) are counted in
    ``n_inf`` and excluded from the other statistics.
    """
    biotech = stock[~stock["ticker"].isin(SAFE_HAVENS)]
    rows = []
    for source, df, features in (
        ("stock", biotech, PRICE_FEATURES + FUNDAMENTAL_FEATURES + EVENT_NEWS_FEATURES),
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


def sec_coverage(filings, facts, stock):
    """Per-ticker filing counts, first filing date and metric coverage."""
    counts = filings.pivot_table(index="ticker", columns="form", values="accession_number",
                                 aggfunc="count", fill_value=0)
    counts = counts[[c for c in ["10-K", "10-Q", "8-K"] if c in counts]]
    counts["first_filing"] = filings.groupby("ticker")["filing_date"].min()

    # Share of each ticker's trading days on which the joined SEC metric is known
    biotech = stock[~stock["ticker"].isin(SAFE_HAVENS)]
    known = biotech.groupby("ticker")[list(SEC_METRICS)].apply(lambda d: d.notna().mean())
    known.columns = [f"{c}_known_pct" for c in known.columns]
    out = counts.join(100 * known, how="outer")
    out["xbrl_facts"] = facts.groupby("ticker").size()
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(1)
    return out.reset_index()


def plot_sec_availability(filings, sec_cov, path):
    f = filings.assign(year=filings["filing_date"].dt.year)
    f["form_group"] = f["form"].str.replace("/A", "", regex=False)
    by_year = f.pivot_table(index="year", columns="form_group", values="accession_number",
                            aggfunc="count", fill_value=0)
    tickers_by_year = f.groupby("year")["ticker"].nunique()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    by_year.plot.bar(stacked=True, ax=axes[0], width=0.8)
    axes[0].set_title("SEC filings per year by form")
    axes[0].set_ylabel("filings")
    ax_t = axes[0].twinx()
    ax_t.plot(range(len(tickers_by_year)), tickers_by_year.values, color="black", marker="o", ms=3)
    ax_t.set_ylabel("tickers filing")

    known = sec_cov.filter(like="_known_pct")
    known.columns = known.columns.str.replace("_known_pct", "")
    known.median().sort_values().plot.barh(ax=axes[1], color="steelblue")
    axes[1].set_xlim(0, 100)
    axes[1].set_title("Median % of trading days\nwith each SEC metric available")

    axes[2].hist(sec_cov["8-K"].dropna(), bins=30, color="steelblue")
    axes[2].set_title("8-K filings per ticker")
    axes[2].set_xlabel("8-K count")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def news_coverage(news, sentiment):
    scored = news.merge(sentiment[["uuid", "sentiment"]], on="uuid", how="left")
    return (
        scored.groupby("ticker")
        .agg(headlines=("uuid", "nunique"), first=("published_at", "min"),
             last=("published_at", "max"), mean_sentiment=("sentiment", "mean"),
             publishers=("publisher", "nunique"))
        .sort_values("headlines", ascending=False)
        .reset_index()
        .round({"mean_sentiment": 3})
    )


def plot_news_availability(news, sentiment, news_cov, path):
    daily = news.drop_duplicates("uuid").set_index("published_at").resample("D")["uuid"].count()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].bar(daily.index, daily.values, width=1, color="steelblue")
    axes[0].set_title("Unique headlines per day")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].hist(news_cov["headlines"], bins=30, color="steelblue")
    axes[1].set_title("Headlines per ticker")
    axes[1].set_xlabel("headlines")

    axes[2].hist(sentiment["sentiment"].dropna(), bins=40, color="steelblue")
    axes[2].set_title(f"Headline sentiment ({sentiment['method'].iloc[0]})")
    axes[2].set_xlabel("positive - negative")
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
    filings = pd.read_parquet(RAW_DIR / "sec_filings.parquet")
    facts = pd.read_parquet(RAW_DIR / "sec_facts.parquet", columns=["ticker", "concept"])
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
    hist_grid(biotech, FUNDAMENTAL_FEATURES, EDA_DIR / "fundamental_feature_dist.png",
              "SEC fundamental features (biotech rows)")
    hist_grid(biotech, EVENT_NEWS_FEATURES, EDA_DIR / "event_news_feature_dist.png",
              "8-K, earnings and news features (biotech rows)")
    hist_grid(regime, [c for c in regime.columns if c != "date"],
              EDA_DIR / "regime_feature_dist.png", "Market regime features", ncols=5)
    plot_volatility_over_time(stock, regime, EDA_DIR / "volatility_over_time.png")
    plot_feature_coverage(stock, EDA_DIR / "feature_coverage.png")

    sec_cov = sec_coverage(filings, facts, stock)
    sec_cov.to_csv(EDA_DIR / "sec_coverage.csv", index=False)
    plot_sec_availability(filings, sec_cov, EDA_DIR / "sec_availability.png")
    known = sec_cov.filter(like="_known_pct")
    log.info(
        "SEC: %d filings for %d tickers (%s -> %s), %d XBRL facts\n"
        "Median %% of trading days with each metric:\n%s",
        len(filings), filings["ticker"].nunique(),
        filings["filing_date"].min().date(), filings["filing_date"].max().date(),
        len(facts), known.median().round(1).to_string(),
    )
    sparse = sec_cov.loc[known.mean(axis=1) < 50, "ticker"].tolist()
    if sparse:
        log.info("Tickers with <50%% SEC metric coverage: %s", ", ".join(sparse))

    news_path, sent_path = RAW_DIR / "news.parquet", FEATURE_DIR / "news_sentiment.parquet"
    if news_path.exists() and sent_path.exists():
        news = pd.read_parquet(news_path)
        sentiment = pd.read_parquet(sent_path)
        news_cov = news_coverage(news, sentiment)
        news_cov.to_csv(EDA_DIR / "news_coverage.csv", index=False)
        plot_news_availability(news, sentiment, news_cov, EDA_DIR / "news_availability.png")
        in_panel = biotech["news_sentiment_21d"].notna()
        log.info(
            "News: %d headlines (%d unique) for %d tickers, %s -> %s; "
            "median %d per ticker; sentiment known on %.1f%% of biotech panel rows",
            len(news), news["uuid"].nunique(), news["ticker"].nunique(),
            news["published_at"].min().date(), news["published_at"].max().date(),
            news_cov["headlines"].median(), 100 * in_panel.mean(),
        )
    else:
        log.info("No news data; skipping news EDA")

    log.info("Saved EDA outputs -> %s", EDA_DIR)


if __name__ == "__main__":
    main()
