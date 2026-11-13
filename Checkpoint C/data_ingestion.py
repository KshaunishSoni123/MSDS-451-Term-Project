"""
Data ingestion for the biotech portfolio optimizer.

Pulls raw data for a universe of biotechnology stocks and writes it to
parquet files under ``data/raw/`` for downstream feature engineering:

    universe.csv           XBI holdings snapshot (only with --universe xbi)
    prices.parquet         Daily OHLCV + adjusted close (yfinance), long format,
                           with Bollinger Band columns (SMA, bands, %B, bandwidth)
    company_info.parquet   Snapshot of yfinance profile/valuation fields
    news.parquet           Recent news headlines (yfinance)
    sec_companies.parquet  SEC company metadata (edgartools)
    sec_filings.parquet    10-K / 10-Q / 8-K filing index (edgartools)
    sec_facts.parquet      XBRL financial facts with filing dates (edgartools)

The SEC requires a User-Agent identifying the requester. Set it with:

    export EDGAR_IDENTITY="Your Name your.email@example.com"

Usage:
    python data_ingestion.py
    python data_ingestion.py --tickers VRTX REGN AMGN --start 2018-01-01
    python data_ingestion.py --years 10 --bb-window 20 --bb-std 2
    python data_ingestion.py --skip-sec --refresh
    python data_ingestion.py --universe xbi    # current XBI holdings instead

The default universe is the 20 names in BIOTECH_TICKERS (core), the same
stocks as Checkpoint B. Point-in-time eligibility
(listing history, market cap, liquidity) is applied later in
feature_engineering.py. Yahoo has no prices for delisted companies, so the
universe still carries survivorship bias from names that left the index.
"""

import argparse
import io
import logging
import os
import re
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from edgar import Company, set_identity

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# US-domiciled biotech names that file 10-K/10-Q with the SEC. Foreign filers
# (e.g. BNTX, ARGX) file 20-F/6-K and have sparse XBRL coverage, so they are
# left out of the default universe.
BIOTECH_TICKERS = [
    # Large cap
    "AMGN", "GILD", "VRTX", "REGN", "BIIB", "MRNA",
    # Mid cap
    "ALNY", "INCY", "BMRN", "NBIX", "UTHR", "EXEL", "INSM",
    "SRPT", "IONS", "NTRA", "HALO", "RYTM", "CRSP", "ACAD",
]

XBI_HOLDINGS_URL = (
    "https://www.ssga.com/us/en/intermediary/library-content/products/"
    "fund-data/etfs/us/holdings-daily-us-en-xbi.xlsx"
)

# Benchmarks and safe-haven assets used for relative performance and
# regime-based rotation (HMM) later on.
BENCHMARK_TICKERS = [
    "XBI",   # SPDR S&P Biotech (equal weight)
    "IBB",   # iShares Biotech (cap weight)
    "XLV",   # Health Care Select Sector SPDR
    "SPY",   # S&P 500
    "TLT",   # 20+ yr Treasuries (safe haven)
    "SHY",   # 1-3 yr Treasuries (cash proxy)
    "^VIX",  # Volatility index
]

HISTORY_YEARS = 20  # default lookback; tickers listed later start at their IPO
DEFAULT_START = (pd.Timestamp.today().normalize() - pd.DateOffset(years=HISTORY_YEARS)).strftime("%Y-%m-%d")

# Bollinger Bands: SMA(window) +/- num_std * rolling std of adjusted close
BB_WINDOW = 20
BB_NUM_STD = 2.0
SEC_FORMS = ["10-K", "10-Q", "8-K"]
DATA_DIR = Path(__file__).resolve().parent / "data" / "raw"

INFO_FIELDS = [
    "shortName", "longName", "sector", "industry", "country", "marketCap",
    "enterpriseValue", "sharesOutstanding", "floatShares", "beta",
    "trailingPE", "forwardPE", "priceToBook", "priceToSalesTrailing12Months",
    "enterpriseToRevenue", "profitMargins", "operatingMargins",
    "returnOnEquity", "returnOnAssets", "revenueGrowth", "totalCash",
    "totalDebt", "debtToEquity", "freeCashflow", "heldPercentInsiders",
    "heldPercentInstitutions", "shortPercentOfFloat", "shortRatio",
    "fullTimeEmployees",
]

log = logging.getLogger("data_ingestion")


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

def fetch_xbi_holdings():
    """Current SPDR S&P Biotech ETF (XBI) holdings as a DataFrame."""
    resp = requests.get(XBI_HOLDINGS_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    resp.raise_for_status()
    raw = pd.read_excel(io.BytesIO(resp.content), header=None)
    as_of = re.search(r"\d{2}-\w{3}-\d{4}", " ".join(map(str, raw.iloc[:4, 1])))
    header_row = raw.index[raw.iloc[:, 0].map(str).str.strip() == "Name"][0]
    holdings = raw.iloc[header_row + 1:].set_axis(list(raw.iloc[header_row]), axis=1)
    holdings = holdings.dropna(subset=["Ticker"])
    holdings["Ticker"] = holdings["Ticker"].astype(str).str.strip().str.upper()
    # Drop cash lines and anything that is not a plain US ticker
    holdings = holdings[holdings["Ticker"].str.fullmatch(r"[A-Z]{1,5}")]
    return pd.DataFrame({
        "ticker": holdings["Ticker"],
        "name": holdings["Name"],
        "xbi_weight": pd.to_numeric(holdings["Weight"], errors="coerce"),
        "as_of": pd.to_datetime(as_of.group(0) if as_of else date.today()),
    }).reset_index(drop=True)


def load_universe(refresh=False):
    """XBI holdings, cached in universe.csv; falls back to BIOTECH_TICKERS."""
    path = DATA_DIR / "universe.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path)["ticker"].tolist()
    try:
        holdings = fetch_xbi_holdings()
    except Exception as exc:
        log.warning("Could not download XBI holdings (%s); using the core list", exc)
        return list(BIOTECH_TICKERS)
    holdings.to_csv(path, index=False)
    log.info("Universe: %d XBI holdings as of %s -> %s",
             len(holdings), holdings["as_of"].iloc[0].date(), path)
    return holdings["ticker"].tolist()


def cached_tickers(name):
    path = DATA_DIR / f"{name}.parquet"
    return set(pd.read_parquet(path, columns=["ticker"])["ticker"]) if path.exists() else set()


# ---------------------------------------------------------------------------
# yfinance: market data, company info, news
# ---------------------------------------------------------------------------

def fetch_prices(tickers, start, end=None):
    """Download daily OHLCV for all tickers and return a long-format frame."""
    log.info("Downloading prices for %d tickers from %s", len(tickers), start)
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=False,  # keep both Close and Adj Close
        actions=True,       # dividends and splits
        group_by="column",
        progress=False,
        threads=True,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no price data")

    prices = (
        raw.stack(level="Ticker", future_stack=True)
        .reset_index()
        .rename(columns=lambda c: c.lower().replace(" ", "_"))
        .dropna(subset=["close"])
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )
    prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None)

    missing = set(tickers) - set(prices["ticker"])
    if missing:
        log.warning("No price data for: %s", sorted(missing))
    return prices


def add_bollinger_bands(prices, window=BB_WINDOW, num_std=BB_NUM_STD, price_col="adj_close"):
    """Add Bollinger Band columns to a long-format price frame, per ticker.

    Computed on adjusted close so splits do not create false band breaks.

        bb_sma        simple moving average over ``window`` days (middle band)
        bb_std        rolling standard deviation (population, as in Bollinger's definition)
        bb_upper      bb_sma + num_std * bb_std
        bb_lower      bb_sma - num_std * bb_std
        bb_pct_b      (price - bb_lower) / (bb_upper - bb_lower); >1 above, <0 below the bands
        bb_bandwidth  (bb_upper - bb_lower) / bb_sma; low values flag a volatility squeeze

    The first ``window - 1`` rows of each ticker are NaN. Every value uses only
    prices up to and including that date, so there is no look-ahead.
    """
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    roll = prices.groupby("ticker")[price_col].rolling(window, min_periods=window)
    sma = roll.mean().reset_index(level=0, drop=True)
    std = roll.std(ddof=0).reset_index(level=0, drop=True)
    upper = sma + num_std * std
    lower = sma - num_std * std
    width = (upper - lower).replace(0, float("nan"))
    return prices.assign(
        bb_sma=sma,
        bb_std=std,
        bb_upper=upper,
        bb_lower=lower,
        bb_pct_b=(prices[price_col] - lower) / width,
        bb_bandwidth=width / sma,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save(df, name, append_keys=None):
    """Write a frame to ``DATA_DIR/<name>.parquet``.

    With ``append_keys``, merge into the existing file and de-duplicate on
    those keys instead of overwriting (used to accumulate news history).
    """
    path = DATA_DIR / f"{name}.parquet"
    if df.empty:
        log.warning("Nothing to save for %s", name)
        return
    if append_keys and path.exists():
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
        df = df.drop_duplicates(subset=append_keys, keep="last")
    df.to_parquet(path, index=False)
    log.info("Saved %s (%d rows) -> %s", name, len(df), path)


def is_cached(*names):
    return all((DATA_DIR / f"{n}.parquet").exists() for n in names)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Biotech tickers to ingest (overrides --universe)")
    parser.add_argument("--universe", choices=["xbi", "core"], default="core",
                        help="core: built-in 20 names (default); xbi: current XBI holdings")
    parser.add_argument("--start", default=None,
                        help=f"History start date, YYYY-MM-DD (default: {HISTORY_YEARS} years ago, {DEFAULT_START})")
    parser.add_argument("--years", type=int, default=None,
                        help="Years of history to pull, ending today (ignored if --start is given)")
    parser.add_argument("--bb-window", type=int, default=BB_WINDOW,
                        help="Bollinger Band SMA window in trading days")
    parser.add_argument("--bb-std", type=float, default=BB_NUM_STD,
                        help="Bollinger Band width in standard deviations")
    parser.add_argument("--end", default=None, help="History end date, YYYY-MM-DD")
    parser.add_argument("--refresh", action="store_true",
                        help="Re-download prices and SEC data even if cached")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    elif args.universe == "core":
        tickers = list(BIOTECH_TICKERS)
    else:
        tickers = load_universe(args.refresh)
    if args.start:
        start = pd.Timestamp(args.start)
    elif args.years:
        start = pd.Timestamp.today().normalize() - pd.DateOffset(years=args.years)
    else:
        start = pd.Timestamp(DEFAULT_START)
    log.info("Ingesting %d biotech tickers from %s", len(tickers), start.date())

    # Re-download when the cache was built for a different universe or starts
    # later than requested (SPY has traded since 1993, so it marks the cache start)
    universe_changed = set(tickers) != cached_tickers("requested_tickers")
    prices_path = DATA_DIR / "prices.parquet"
    history_short = False
    if prices_path.exists():
        cached = pd.read_parquet(prices_path, columns=["ticker", "date"])
        spy_start = cached.loc[cached["ticker"] == "SPY", "date"].min()
        history_short = pd.isna(spy_start) or spy_start > start + pd.Timedelta(days=7)
    if args.refresh or universe_changed or history_short or not is_cached("prices", "company_info"):
        prices = fetch_prices(tickers + BENCHMARK_TICKERS, start.strftime("%Y-%m-%d"), args.end)
        pd.DataFrame({"ticker": tickers}).to_parquet(DATA_DIR / "requested_tickers.parquet", index=False)
    else:
        log.info("Prices cached; use --refresh to re-download")
        prices = pd.read_parquet(prices_path)

    # Bands are cheap to compute, so they are rebuilt every run; this keeps
    # them in sync with --bb-window / --bb-std even when prices are cached.
    prices = add_bollinger_bands(prices.drop(columns=prices.filter(like="bb_").columns),
                                 window=args.bb_window, num_std=args.bb_std)
    save(prices, "prices")

if __name__ == "__main__":
    main()
