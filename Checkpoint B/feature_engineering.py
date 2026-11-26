"""
Feature engineering for the biotech portfolio optimizer.

Reads the raw parquet files written by ``data_ingestion.py`` and builds:

    data/features/stock_features.parquet   Daily (date, ticker) panel of
                                           momentum, realized volatility,
                                           beta, drawdown and liquidity features
    data/features/regime_features.parquet  Daily market-level features for
                                           regime detection (VIX, XBI vs TLT,
                                           FOMC / CPI event calendar, ...)

Every feature comes from prices and volume (no SEC filings or news), and each
value on day t uses data up to and including t only, so backtests have no
look-ahead bias.

Usage:
    python feature_engineering.py
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from event_calendar import CPI_RELEASES, FOMC_DECISIONS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
FEATURE_DIR = BASE_DIR / "data" / "features"

TRADING_DAYS = 252
MARKET_PROXY = "XBI"   # biotech benchmark for beta

# Safe-haven assets the portfolio can rotate into; they get the same price
# features plus an is_safe_haven flag.
SAFE_HAVENS = ["SHY", "TLT"]

# Point-in-time investability rules for biotech names. Without SEC share
# counts there is no point-in-time market cap, so size is screened through
# trading volume instead.
MIN_HISTORY_DAYS = 252        # a year of trading, so 252-day features exist
MIN_DOLLAR_VOLUME = 5e6       # 63-day average daily dollar volume

EVENT_HORIZON_CAP = 63         # business days; "days to next event" is capped here

log = logging.getLogger("feature_engineering")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def stack_features(frames):
    """Combine wide (date x ticker) frames into one long (date, ticker) frame."""
    return pd.concat(
        {name: df.stack(future_stack=True) for name, df in frames.items()}, axis=1
    ).rename_axis(["date", "ticker"])


def business_days_until(dates, next_dates):
    """Business days from each date to its next event (0 on the event day)."""
    dates = pd.DatetimeIndex(dates)
    next_dates = pd.DatetimeIndex(next_dates)
    out = np.full(len(dates), np.nan)
    valid = ~next_dates.isna()
    out[valid] = np.busday_count(
        dates[valid].values.astype("datetime64[D]"),
        next_dates[valid].values.astype("datetime64[D]"),
    )
    return np.minimum(out, EVENT_HORIZON_CAP)


def scheduled_event_features(name, event_dates, calendar):
    """Day-of flag and countdown for a market-wide event published in advance."""
    event_dates = pd.DatetimeIndex(sorted(event_dates))
    idx = event_dates.searchsorted(calendar, side="left")
    has_next = idx < len(event_dates)
    next_event = pd.DatetimeIndex(
        np.where(has_next, event_dates[np.minimum(idx, len(event_dates) - 1)], np.datetime64("NaT"))
    )
    return pd.DataFrame({
        f"{name}_day": calendar.isin(event_dates).astype(float),
        f"days_to_{name}": business_days_until(calendar, next_event),
    }, index=calendar)


# ---------------------------------------------------------------------------
# Price-based features: momentum, volatility, beta, liquidity
# ---------------------------------------------------------------------------

def price_features(prices, universe):
    px = prices.pivot(index="date", columns="ticker", values="adj_close")
    close = prices.pivot(index="date", columns="ticker", values="close")
    volume = prices.pivot(index="date", columns="ticker", values="volume")
    rets = px.pct_change(fill_method=None)

    r = rets[universe]
    m = rets[MARKET_PROXY]
    p = px[universe]
    ann = np.sqrt(TRADING_DAYS)
    w252 = dict(window=TRADING_DAYS, min_periods=126)
    w63 = dict(window=63, min_periods=42)
    w21 = dict(window=21, min_periods=15)
    w5 = dict(window=5, min_periods=4)

    # Beta via rolling moments: cov(r, m) / var(m)
    cov = r.mul(m, axis=0).rolling(**w252).mean() - r.rolling(**w252).mean().mul(
        m.rolling(**w252).mean(), axis=0
    )
    beta = cov.div(m.rolling(**w252).var(ddof=0), axis=0)

    frames = {
        # Momentum (skip the most recent month to avoid short-term reversal)
        "mom_12_1": p.shift(21) / p.shift(TRADING_DAYS) - 1,
        "mom_6_1": p.shift(21) / p.shift(126) - 1,
        "ret_1m": p / p.shift(21) - 1,
        # Realized volatility at short, medium and long horizons
        "vol_5d": r.rolling(**w5).std() * ann,
        "vol_21d": r.rolling(**w21).std() * ann,
        "vol_63d": r.rolling(**w63).std() * ann,
        "vol_252d": r.rolling(**w252).std() * ann,
        # Above 1 when the stock is more volatile than usual (vol regime)
        "vol_ratio_21_252": r.rolling(**w21).std() / r.rolling(**w252).std(),
        "downside_vol_63d": (r.clip(upper=0) ** 2).rolling(**w63).mean() ** 0.5 * ann,
        f"beta_{MARKET_PROXY.lower()}_252d": beta,
        # Biotech gap risk: count of >10% daily moves in the past year
        "jump_count_252d": (r.abs() > 0.10).astype(float).where(r.notna()).rolling(**w252).sum(),
        "dist_52w_high": p / p.rolling(**w252).max() - 1,
        # Liquidity
        "log_dollar_volume_63d": np.log(
            (close[universe] * volume[universe]).rolling(**w63).mean().replace(0, np.nan)
        ),
    }
    feats = stack_features(frames)
    feats["close"] = close[universe].stack(future_stack=True)
    return feats.dropna(subset=["close"])


# ---------------------------------------------------------------------------
# Point-in-time eligibility
# ---------------------------------------------------------------------------

def eligibility(panel):
    """Flag rows that were investable on that date using only data known then.

    Biotech names need a year of history and enough liquidity; safe havens
    only need the history.
    """
    panel = panel.sort_values(["ticker", "date"])
    history = panel.groupby("ticker").cumcount() + 1
    is_safe = panel["ticker"].isin(SAFE_HAVENS)
    liquid = np.exp(panel["log_dollar_volume_63d"]) >= MIN_DOLLAR_VOLUME
    panel["is_safe_haven"] = is_safe.astype(float)
    panel["eligible"] = (history >= MIN_HISTORY_DAYS) & (is_safe | liquid)
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Market regime features
# ---------------------------------------------------------------------------

def regime_features(prices, calendar):
    px = prices.pivot(index="date", columns="ticker", values="adj_close")
    rets = px.pct_change(fill_method=None)
    vix = px["^VIX"]
    w252 = dict(window=TRADING_DAYS, min_periods=126)

    def ret(t, n=63):
        return px[t] / px[t].shift(n) - 1

    features = pd.DataFrame({
        "vix": vix,
        "vix_change_21d": vix - vix.shift(21),
        "vix_zscore_252d": (vix - vix.rolling(**w252).mean()) / vix.rolling(**w252).std(),
        "xbi_ret_63d": ret("XBI"),
        "tlt_ret_63d": ret("TLT"),
        "xbi_minus_tlt_63d": ret("XBI") - ret("TLT"),
        "xbi_minus_spy_63d": ret("XBI") - ret("SPY"),
        "xbi_vol_21d": rets["XBI"].rolling(21, min_periods=15).std() * np.sqrt(TRADING_DAYS),
        "xbi_drawdown_252d": px["XBI"] / px["XBI"].rolling(**w252).max() - 1,
        "spy_drawdown_252d": px["SPY"] / px["SPY"].rolling(**w252).max() - 1,
    }).dropna(how="all")

    # Scheduled macro events: FOMC decisions (2pm ET) and CPI (8:30am ET)
    # both land during the session, so the flag is on the release day itself
    events = pd.concat([
        scheduled_event_features("fomc", FOMC_DECISIONS, calendar),
        scheduled_event_features("cpi", CPI_RELEASES, calendar),
    ], axis=1)
    return features.join(events)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    return parser.parse_args()


def main():
    parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    prices = pd.read_parquet(RAW_DIR / "prices.parquet")
    prices["date"] = prices["date"].astype("datetime64[ns]")
    priced = set(prices["ticker"])
    biotech = sorted(set(pd.read_parquet(RAW_DIR / "company_info.parquet")["ticker"]) & priced)
    universe = biotech + SAFE_HAVENS
    calendar = pd.DatetimeIndex(sorted(prices.loc[prices["ticker"] == MARKET_PROXY, "date"]))
    # Yahoo occasionally quotes ^VIX on market holidays; those rows would
    # insert empty days into every stock's rolling windows
    prices = prices[prices["date"].isin(calendar)]

    log.info("Price features for %d biotech tickers + %s", len(biotech), SAFE_HAVENS)
    panel = price_features(prices, universe)
    panel = eligibility(panel.reset_index())
    eligible_now = panel.loc[panel["date"] == panel["date"].max(), "eligible"].sum()
    log.info("Eligible assets on the last date: %d of %d", eligible_now, len(universe))
    panel.to_parquet(FEATURE_DIR / "stock_features.parquet", index=False)
    log.info("Saved stock_features: %d rows x %d cols", *panel.shape)

    regime = regime_features(prices, calendar).reset_index()
    regime.to_parquet(FEATURE_DIR / "regime_features.parquet", index=False)
    log.info("Saved regime_features: %d rows x %d cols", *regime.shape)

    latest = panel[panel["date"] == panel["date"].max()]
    coverage = latest.drop(columns=["date", "ticker"]).notna().mean().round(2)
    log.info("Feature coverage on %s:\n%s", latest["date"].iloc[0].date(), coverage.to_string())


if __name__ == "__main__":
    main()
