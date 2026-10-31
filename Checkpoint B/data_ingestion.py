"""
Data ingestion for the biotech portfolio optimizer.

Pulls raw data for a universe of biotechnology stocks and writes it to
parquet files under ``data/raw/`` for downstream feature engineering:

    universe.csv           Biotech universe: snapshot of SPDR XBI holdings
    prices.parquet         Daily OHLCV + adjusted close (yfinance), long format
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
    python data_ingestion.py --skip-sec --refresh
    python data_ingestion.py --universe core   # the original 20-stock list

The default universe is XBI's current holdings. Point-in-time eligibility
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

DEFAULT_START = "2015-01-01"
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


def _parse_news_item(ticker, item):
    """Normalize the two news payload shapes yfinance can return."""
    # Newer Ticker.get_news payload: {"id": ..., "content": {...}}
    if "content" in item:
        content = item["content"]
        provider = content.get("provider") or {}
        url = (content.get("canonicalUrl") or {}).get("url")
        return {
            "ticker": ticker,
            "uuid": item.get("id"),
            "title": content.get("title"),
            "summary": content.get("summary"),
            "publisher": provider.get("displayName"),
            "link": url,
            "published_at": pd.to_datetime(content.get("pubDate"), utc=True),
            "related_tickers": None,
        }
    # yf.Search payload: flat dict with a unix timestamp
    return {
        "ticker": ticker,
        "uuid": item.get("uuid"),
        "title": item.get("title"),
        "summary": None,
        "publisher": item.get("publisher"),
        "link": item.get("link"),
        "published_at": pd.to_datetime(
            item.get("providerPublishTime"), unit="s", utc=True
        ),
        "related_tickers": ",".join(item.get("relatedTickers") or []),
    }


def fetch_news(tickers, count=50):
    """Fetch recent headlines per ticker.

    Yahoo only exposes recent news (roughly the last few weeks), so this is a
    rolling snapshot. Re-run the script periodically and the results are
    appended to the existing news file to build history over time.
    """
    rows = []
    for ticker in tickers:
        items = []
        try:
            items = yf.Ticker(ticker).get_news(count=count)
        except Exception as exc:
            log.debug("get_news failed for %s: %s", ticker, exc)
        if not items:
            # Ticker.get_news intermittently returns nothing; Search is more
            # reliable but can include headlines about other companies.
            try:
                items = yf.Search(ticker, news_count=count, max_results=0).news
            except Exception as exc:
                log.warning("news failed for %s: %s", ticker, exc)
                continue
        rows.extend(_parse_news_item(ticker, item) for item in items)
        time.sleep(0.25)

    news = pd.DataFrame(rows)
    if news.empty:
        return news
    news["fetched_at"] = datetime.now(timezone.utc)
    return news.drop_duplicates(subset=["ticker", "uuid"])


# ---------------------------------------------------------------------------
# edgartools: SEC company metadata, filings, XBRL financial facts
# ---------------------------------------------------------------------------

def configure_edgar_identity():
    identity = "Kshaunish Soni kshaunishsoni123@gmail.com"
    if not identity:
        raise SystemExit(
            "EDGAR_IDENTITY is not set. The SEC requires a contact identity, e.g.\n"
            '  export EDGAR_IDENTITY="Your Name your.email@example.com"\n'
            "or run with --skip-sec."
        )
    set_identity(identity)


def fetch_sec_data(tickers, start):
    """Pull company metadata, filing index, and XBRL facts for each ticker."""
    companies, filings, facts = [], [], []

    for ticker in tickers:
        log.info("SEC: %s", ticker)
        try:
            company = Company(ticker)
        except Exception as exc:
            log.warning("SEC lookup failed for %s: %s", ticker, exc)
            continue
        if company.not_found:
            log.warning("SEC has no company for %s", ticker)
            continue

        companies.append({
            "ticker": ticker,
            "cik": company.cik,
            "name": company.name,
            "sic": company.sic,
            "industry": company.industry,
            "fiscal_year_end": company.fiscal_year_end,
            "is_foreign": company.is_foreign,
            "filer_category": str(company.filer_category),
        })

        try:
            filing_df = company.get_filings(form=SEC_FORMS).to_pandas()
            filing_df = filing_df[pd.to_datetime(filing_df["filing_date"]) >= start]
            filing_df.insert(0, "ticker", ticker)
            filing_df.insert(1, "cik", company.cik)
            filings.append(filing_df)
        except Exception as exc:
            log.warning("filings failed for %s: %s", ticker, exc)

        try:
            # include_metadata adds filing_date/form_type, which lets feature
            # engineering use only facts that were public as of each date.
            fact_df = company.get_facts().to_dataframe(include_metadata=True)
            fact_df.insert(0, "ticker", ticker)
            fact_df.insert(1, "cik", company.cik)
            facts.append(fact_df)
        except Exception as exc:
            log.warning("facts failed for %s: %s", ticker, exc)

    filings_df = pd.concat(filings, ignore_index=True) if filings else pd.DataFrame()
    facts_df = pd.concat(facts, ignore_index=True) if facts else pd.DataFrame()

    for df, cols in (
        (filings_df, ["filing_date", "reportDate", "acceptanceDateTime"]),
        (facts_df, ["period_start", "period_end", "filing_date"]),
    ):
        for col in cols:
            if col in df:
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=col == "acceptanceDateTime")

    # Mixed str/float values do not serialize to parquet; numeric_value holds
    # the numbers, so keep the raw value as text.
    if "value" in facts_df:
        facts_df["value"] = facts_df["value"].astype(str)

    return pd.DataFrame(companies), filings_df, facts_df


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
    parser.add_argument("--universe", choices=["xbi", "core"], default="xbi",
                        help="xbi: current XBI holdings (default); core: built-in 20 names")
    parser.add_argument("--start", default=DEFAULT_START,
                        help="History start date, YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="History end date, YYYY-MM-DD")
    parser.add_argument("--news-count", type=int, default=50,
                        help="Max headlines per ticker")
    parser.add_argument("--skip-news", action="store_true")
    parser.add_argument("--skip-sec", action="store_true")
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
    start = pd.Timestamp(args.start)
    log.info("Ingesting %d biotech tickers", len(tickers))

    # Re-download when the cache was built for a different universe
    universe_changed = set(tickers) != cached_tickers("requested_tickers")
    if args.refresh or universe_changed or not is_cached("prices", "company_info"):
        save(fetch_prices(tickers + BENCHMARK_TICKERS, args.start, args.end), "prices")
        save(fetch_company_info(tickers), "company_info")
        pd.DataFrame({"ticker": tickers}).to_parquet(DATA_DIR / "requested_tickers.parquet", index=False)
    else:
        log.info("Prices cached; use --refresh to re-download")

    if not args.skip_news:
        # News is always fetched and appended, since Yahoo only serves recent items
        save(fetch_news(tickers, args.news_count), "news", append_keys=["ticker", "uuid"])

    if not args.skip_sec:
        sec_stale = not set(tickers) <= cached_tickers("sec_companies") | cached_tickers("sec_skipped")
        if args.refresh or sec_stale or not is_cached("sec_companies", "sec_filings", "sec_facts"):
            configure_edgar_identity()
            companies, filings, facts = fetch_sec_data(tickers, start)
            save(companies, "sec_companies")
            save(filings, "sec_filings")
            save(facts, "sec_facts")
            # Remember tickers with no SEC data so they don't force a re-download
            skipped = sorted(set(tickers) - set(companies["ticker"]))
            pd.DataFrame({"ticker": skipped}).to_parquet(DATA_DIR / "sec_skipped.parquet", index=False)
        else:
            log.info("SEC data cached; use --refresh to re-download")


if __name__ == "__main__":
    main()
