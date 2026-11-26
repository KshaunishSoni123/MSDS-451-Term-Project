"""
Data ingestion for the biotech portfolio optimizer.

Pulls raw data for a universe of biotechnology stocks and writes it to
parquet files under ``data/raw/`` for downstream feature engineering:

    universe.csv           XBI holdings snapshot (the default universe)
    prices.parquet         Daily OHLCV + adjusted close (yfinance), long format
    company_info.parquet   Snapshot of yfinance profile fields

The model uses price data only: no SEC filings or news.

Usage:
    python data_ingestion.py                   # every current XBI holding
    python data_ingestion.py --refresh         # re-download holdings and prices
    python data_ingestion.py --universe core   # the 20 names in BIOTECH_TICKERS
    python data_ingestion.py --tickers VRTX REGN AMGN --start 2018-01-01

The default universe is every stock in the SPDR S&P Biotech ETF (XBI), from the
holdings file State Street publishes (cached in universe.csv; --refresh
re-downloads it). Checkpoint C trades the same names. Point-in-time
eligibility (listing history, liquidity) is applied later in
feature_engineering.py. The holdings are today's, and Yahoo has no prices for
delisted companies, so the universe carries survivorship bias from names that
left the index.
"""

import argparse
import io
import logging
import re
from datetime import date
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# A 20-name large/mid-cap subset, kept for quick runs (--universe core).
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

DEFAULT_START = "2015-01-01"
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
# yfinance: market data and company info
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


def fetch_company_info(tickers):
    """Snapshot of profile and valuation fields from yfinance ``Ticker.info``."""
    rows = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
        except Exception as exc:
            log.warning("info failed for %s: %s", ticker, exc)
            continue
        row = {"ticker": ticker, "as_of": date.today()}
        row.update({field: info.get(field) for field in INFO_FIELDS})
        rows.append(row)
    info = pd.DataFrame(rows)
    # Yahoo sometimes returns strings like 'Infinity' in numeric fields
    text_fields = {"shortName", "longName", "sector", "industry", "country"}
    for field in INFO_FIELDS:
        if field not in text_fields and field in info:
            info[field] = pd.to_numeric(info[field], errors="coerce")
    return info


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save(df, name):
    """Write a frame to ``DATA_DIR/<name>.parquet``."""
    path = DATA_DIR / f"{name}.parquet"
    if df.empty:
        log.warning("Nothing to save for %s", name)
        return
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
    parser.add_argument("--universe", choices=["xbi", "core"], default="xbi",
                        help="xbi: every current XBI holding (default); core: built-in 20 names")
    parser.add_argument("--start", default=DEFAULT_START,
                        help="History start date, YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="History end date, YYYY-MM-DD")
    parser.add_argument("--refresh", action="store_true",
                        help="Re-download XBI holdings and prices even if cached")
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
    log.info("Ingesting %d biotech tickers", len(tickers))

    # Re-download when the cache was built for a different universe
    universe_changed = set(tickers) != cached_tickers("requested_tickers")
    if args.refresh or universe_changed or not is_cached("prices", "company_info"):
        save(fetch_prices(tickers + BENCHMARK_TICKERS, args.start, args.end), "prices")
        save(fetch_company_info(tickers), "company_info")
        pd.DataFrame({"ticker": tickers}).to_parquet(DATA_DIR / "requested_tickers.parquet", index=False)
    else:
        log.info("Prices cached; use --refresh to re-download")


if __name__ == "__main__":
    main()
