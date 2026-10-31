"""
Feature engineering for the biotech portfolio optimizer.

Reads the raw parquet files written by ``data_ingestion.py`` and builds:

    data/features/stock_features.parquet   Daily (date, ticker) panel of
                                           momentum, realized volatility,
                                           quality, size/value, cash runway,
                                           SEC event, earnings calendar and
                                           news sentiment features
    data/features/regime_features.parquet  Daily market-level features for
                                           regime detection (VIX, XBI vs TLT,
                                           FOMC / CPI event calendar, ...)
    data/features/news_sentiment.parquet   Scored headlines (cache, so re-runs
                                           only score new articles)

All fundamental and event features are point-in-time: a value only enters the
panel on the first trading day *after* it was filed with the SEC, using the
first-reported number (not later restatements). This keeps backtests free of
look-ahead bias.

Usage:
    python feature_engineering.py
    python feature_engineering.py --sentiment vader   # lightweight, no model download
    python feature_engineering.py --sentiment none
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
MIN_REVENUE = 50e6     # below this, margins are noise (pre-commercial biotech)
RUNWAY_CAP = 10.0      # years; assigned to cash-flow-positive companies
STALE_DAYS = 400       # drop fundamentals older than this (missed filings)

# Safe-haven assets the portfolio can rotate into; they get price features
# only (no fundamentals, events or news) plus an is_safe_haven flag.
SAFE_HAVENS = ["SHY", "TLT"]

# Point-in-time investability rules for biotech names
MIN_HISTORY_DAYS = 252        # a year of trading, so 252-day features exist
MIN_MARKET_CAP = 500e6        # from SEC shares outstanding x price on that day
MIN_DOLLAR_VOLUME = 5e6       # 63-day average daily dollar volume

# Metric -> (period type, XBRL concepts in priority order). Companies switch
# tags over time, so for each period the earliest-filed concept is used and
# priority only breaks ties within the same filing.
SEC_METRICS = {
    "revenue": ("duration", [
        "us-gaap:Revenues",
        "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
    ]),
    "net_income": ("duration", ["us-gaap:NetIncomeLoss"]),
    "operating_income": ("duration", ["us-gaap:OperatingIncomeLoss"]),
    "operating_cash_flow": ("duration", [
        "us-gaap:NetCashProvidedByUsedInOperatingActivities",
        "us-gaap:NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ]),
    "cash": ("instant", [
        "us-gaap:CashAndCashEquivalentsAtCarryingValue",
        "us-gaap:CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ]),
    "short_term_investments": ("instant", [
        "us-gaap:ShortTermInvestments",
        "us-gaap:MarketableSecuritiesCurrent",
        "us-gaap:AvailableForSaleSecuritiesDebtSecuritiesCurrent",
    ]),
    "equity": ("instant", [
        "us-gaap:StockholdersEquity",
        "us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ]),
    "shares_outstanding": ("instant", [
        "dei:EntityCommonStockSharesOutstanding",
        "us-gaap:CommonStockSharesOutstanding",
        # Multi-class filers (e.g. REGN) only tag share counts by class
        "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding",
    ]),
}
PERIODIC_FORMS = {"10-K", "10-Q", "10-K/A", "10-Q/A"}

# 8-K items worth tracking for biotech
EIGHT_K_ITEMS = {
    "801": "8.01",  # Other events: trial readouts, FDA decisions
    "101": "1.01",  # Material agreements: licensing, partnerships, M&A
    "502": "5.02",  # Executive / director departures and appointments
}

EARNINGS_ITEM = "2.02"         # 8-K Results of Operations and Financial Condition
EARNINGS_DEDUP_DAYS = 10       # a 10-Q/10-K this soon after an earnings 8-K is the same event
EVENT_HORIZON_CAP = 63         # business days; "days to next event" is capped here

FINBERT_MODEL = "ProsusAI/finbert"

log = logging.getLogger("feature_engineering")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def stack_features(frames):
    """Combine wide (date x ticker) frames into one long (date, ticker) frame."""
    return pd.concat(
        {name: df.stack(future_stack=True) for name, df in frames.items()}, axis=1
    ).rename_axis(["date", "ticker"])


def next_trading_day(timestamps, calendar):
    """Map each timestamp to the first trading day on or after it."""
    idx = calendar.searchsorted(pd.DatetimeIndex(timestamps).as_unit("ns"), side="left")
    out = pd.Series(pd.NaT, index=range(len(idx)), dtype="datetime64[ns]")
    valid = idx < len(calendar)
    out[valid] = calendar[idx[valid]]
    return out.to_numpy()


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
# SEC fundamentals: point-in-time TTM flows and latest balance sheet values
# ---------------------------------------------------------------------------

def first_reported(facts):
    """Keep the first-filed value for each (ticker, concept, period)."""
    return (
        facts.sort_values("filing_date")
        .drop_duplicates(["ticker", "concept", "period_start", "period_end"], keep="first")
    )


def ttm_from_durations(facts):
    """Trailing-twelve-month values from 10-K/10-Q duration facts.

    10-Qs report year-to-date figures (cash flow only reports YTD), so TTM at
    a quarter end E with fiscal-year start S is:
        YTD(S..E) + FY(prior year) - YTD(S-1y..E-1y)
    Annual facts are used directly.
    """
    tol = pd.Timedelta(days=10)
    f = facts.assign(days=(facts["period_end"] - facts["period_start"]).dt.days)
    annual = f[f["days"].between(350, 380)]
    # Longest sub-annual fact per period end is the YTD figure
    ytd = f[f["days"] < 350].sort_values("days").groupby("period_end").tail(1)
    ytd = ytd[~ytd["period_end"].isin(annual["period_end"])]

    rows = [annual[["period_end", "numeric_value", "filing_date"]]]
    for row in ytd.itertuples():
        prior_fy = annual[
            ((row.period_start - pd.Timedelta(days=1)) - annual["period_end"]).abs() <= tol
        ]
        prior_ytd = f[
            ((row.period_start - pd.DateOffset(years=1)) - f["period_start"]).abs().le(tol)
            & ((row.period_end - pd.DateOffset(years=1)) - f["period_end"]).abs().le(tol)
        ]
        if prior_fy.empty or prior_ytd.empty:
            continue
        rows.append(pd.DataFrame([{
            "period_end": row.period_end,
            "numeric_value": row.numeric_value
            + prior_fy["numeric_value"].iloc[0]
            - prior_ytd["numeric_value"].iloc[0],
            "filing_date": row.filing_date,
        }]))
    return pd.concat(rows, ignore_index=True)


def sec_fundamentals(facts):
    """Long frame of (ticker, metric, period_end, value, filing_date)."""
    facts = facts[
        facts["form_type"].isin(PERIODIC_FORMS)
        & facts["numeric_value"].notna()
        & facts["filing_date"].notna()
    ]
    facts = first_reported(facts)

    out = []
    for metric, (period_type, concepts) in SEC_METRICS.items():
        metric_rows = []
        for priority, concept in enumerate(concepts):
            cf = facts[facts["concept"] == concept]
            if period_type == "duration":
                cf = cf.dropna(subset=["period_start"])
                values = cf.groupby("ticker", group_keys=True)[
                    ["period_start", "period_end", "numeric_value", "filing_date"]
                ].apply(ttm_from_durations)
                values = values.reset_index(level=0)
            else:
                values = cf[["ticker", "period_end", "numeric_value", "filing_date"]]
            metric_rows.append(values.assign(priority=priority))

        m = pd.concat(metric_rows, ignore_index=True)
        m = (
            m.sort_values(["filing_date", "priority"])
            .drop_duplicates(["ticker", "period_end"], keep="first")
            .assign(metric=metric)
        )
        out.append(m)

    fund = pd.concat(out, ignore_index=True).rename(columns={"numeric_value": "value"})
    return fund[["ticker", "metric", "period_end", "value", "filing_date"]]


def as_of_join(panel, fund, calendar):
    """Attach the latest available value of each metric to every (date, ticker)."""
    fund = fund.copy()
    # Filings can land after the close, so values are usable the next trading day
    fund["effective_date"] = next_trading_day(
        fund["filing_date"] + pd.Timedelta(days=1), calendar
    )
    fund = fund.dropna(subset=["effective_date"])

    panel = panel.reset_index().sort_values("date")
    for metric, m in fund.groupby("metric"):
        # Ignore late-filed amendments for periods older than one already known
        m = m.sort_values(["ticker", "effective_date", "period_end"])
        m = m[m["period_end"] >= m.groupby("ticker")["period_end"].cummax()]
        m = m.drop_duplicates(["ticker", "effective_date"], keep="last")
        panel = pd.merge_asof(
            panel,
            m[["ticker", "effective_date", "value"]]
            .rename(columns={"value": metric})
            .sort_values("effective_date"),
            left_on="date",
            right_on="effective_date",
            by="ticker",
            direction="backward",
            tolerance=pd.Timedelta(days=STALE_DAYS),
        ).drop(columns="effective_date")
    return panel.set_index(["date", "ticker"]).sort_index()


def fundamental_features(panel):
    """Size, value, quality and cash runway from the joined SEC metrics."""
    p = panel
    revenue_ok = p["revenue"] >= MIN_REVENUE
    market_cap = p["shares_outstanding"] * p["close"]
    liquid_assets = p["cash"] + p["short_term_investments"].fillna(0)
    burn = -p["operating_cash_flow"]

    return pd.DataFrame({
        "log_market_cap": np.log(market_cap.where(market_cap > 0)),
        "book_to_market": p["equity"] / market_cap,
        "roe": (p["net_income"] / p["equity"]).where(p["equity"] > 0),
        "net_margin": (p["net_income"] / p["revenue"]).where(revenue_ok),
        "operating_margin": (p["operating_income"] / p["revenue"]).where(revenue_ok),
        "cash_to_market_cap": liquid_assets / market_cap,
        "cash_runway_years": (liquid_assets / burn)
        .where(burn > 0, RUNWAY_CAP)
        .clip(upper=RUNWAY_CAP)
        .where(burn.notna() & liquid_assets.notna()),
        "is_cash_flow_positive": (burn <= 0).astype(float).where(burn.notna()),
    }, index=p.index)


# ---------------------------------------------------------------------------
# SEC 8-K event features
# ---------------------------------------------------------------------------

def event_features(filings, calendar, universe):
    eight_k = filings[filings["form"] == "8-K"].copy()
    eight_k["date"] = next_trading_day(eight_k["filing_date"] + pd.Timedelta(days=1), calendar)
    eight_k = eight_k.dropna(subset=["date"])
    items = eight_k["items"].fillna("")

    daily = {"8k": eight_k.assign(n=1)}
    for code, item in EIGHT_K_ITEMS.items():
        daily[f"8k_item{code}"] = eight_k.assign(
            n=items.str.split(",").apply(lambda xs, item=item: item in [x.strip() for x in xs]).astype(int)
        )

    def counts(df):
        wide = df.pivot_table(index="date", columns="ticker", values="n", aggfunc="sum")
        return wide.reindex(index=calendar, columns=universe).fillna(0)

    frames = {}
    for name, df in daily.items():
        c = counts(df)
        frames[f"{name}_count_63d"] = c.rolling(63, min_periods=1).sum()
        if name == "8k":
            frames["8k_count_21d"] = c.rolling(21, min_periods=1).sum()
    frames["8k_item502_count_252d"] = counts(daily["8k_item502"]).rolling(TRADING_DAYS, min_periods=1).sum()

    # Counts are only meaningful once the filing history starts
    start = filings["filing_date"].min()
    return stack_features({k: v.loc[v.index >= start] for k, v in frames.items()})


# ---------------------------------------------------------------------------
# Earnings calendar
# ---------------------------------------------------------------------------

def earnings_dates(filings, calendar):
    """Trading day on which each earnings release first hits the market.

    Earnings are the 8-K item 2.02 press release; companies that skip the 8-K
    fall back to the 10-Q/10-K filing. Releases accepted after the 4pm ET
    close are assigned to the next session.
    """
    items = filings["items"].fillna("").str.split(",").apply(lambda xs: [x.strip() for x in xs])
    is_release = (filings["form"] == "8-K") & items.apply(lambda xs: EARNINGS_ITEM in xs)
    is_report = filings["form"].isin(["10-Q", "10-K"])
    ev = filings[is_release | is_report].copy()

    et = ev["acceptanceDateTime"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    session = et.dt.normalize() + pd.to_timedelta((et.dt.hour >= 16).astype(int), unit="D")
    ev["date"] = next_trading_day(session, calendar)
    ev = ev.dropna(subset=["date"]).sort_values(["ticker", "date"])

    gap = ev.groupby("ticker")["date"].diff()
    ev = ev[gap.isna() | (gap > pd.Timedelta(days=EARNINGS_DEDUP_DAYS))]
    return ev[["ticker", "date"]].reset_index(drop=True)


def earnings_features(filings, calendar, universe):
    """Earnings-day flag and estimated business days to the next release.

    The next release date is not known in advance from SEC data, so it is
    projected point-in-time: companies report on a steady annual cycle, so
    each past release + 52 weeks predicts the same quarter's next release.
    A projection is dropped once an actual release lands within 30 days
    before it (that quarter has already been reported).
    """
    ev = earnings_dates(filings, calendar)

    flag = (
        ev.assign(n=1.0)
        .pivot_table(index="date", columns="ticker", values="n", aggfunc="max")
        .reindex(index=calendar, columns=universe)
        .fillna(0)
    )

    days_to = pd.DataFrame(np.nan, index=calendar, columns=universe)
    t = calendar.values
    for ticker, actual in ev.groupby("ticker")["date"]:
        if ticker not in days_to:
            continue
        actual = np.sort(actual.values)
        proj = actual + np.timedelta64(364, "D")
        # Latest actual release on or before each date
        last_pos = actual.searchsorted(t, side="right") - 1
        last = np.where(last_pos >= 0, actual[np.maximum(last_pos, 0)], np.datetime64("NaT"))

        nxt = np.full(len(t), np.datetime64("NaT"), dtype="datetime64[ns]")
        i = proj.searchsorted(t, side="left")
        for _ in range(2):  # skip at most one already-reported projection
            ok = i < len(proj)
            cand = np.where(ok, proj[np.minimum(i, len(proj) - 1)], np.datetime64("NaT"))
            reported = ok & (last >= cand - np.timedelta64(30, "D"))
            take = ok & ~reported & np.isnat(nxt)
            nxt[take] = cand[take]
            i = i + reported
        # Projections come from releases already observed (proj - 364d <= t)
        nxt[nxt > t + np.timedelta64(364, "D")] = np.datetime64("NaT")
        days_to[ticker] = business_days_until(calendar, nxt)

    start = filings["filing_date"].min()
    return stack_features({
        "earnings_day": flag.loc[flag.index >= start],
        "days_to_earnings_est": days_to.loc[days_to.index >= start],
    })


# ---------------------------------------------------------------------------
# News sentiment
# ---------------------------------------------------------------------------

def score_finbert(texts):
    from transformers import pipeline

    clf = pipeline("text-classification", model=FINBERT_MODEL, top_k=None, truncation=True)
    scores = []
    for result in clf(list(texts), batch_size=32):
        probs = {r["label"]: r["score"] for r in result}
        scores.append(probs.get("positive", 0) - probs.get("negative", 0))
    return scores


def score_vader(texts):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    return [analyzer.polarity_scores(t)["compound"] for t in texts]


SCORERS = {"finbert": score_finbert, "vader": score_vader}


def score_news(news, method):
    """Score headlines in [-1, 1], reusing cached scores for seen articles."""
    cache_path = FEATURE_DIR / "news_sentiment.parquet"
    cached = pd.read_parquet(cache_path) if cache_path.exists() else pd.DataFrame()
    if not cached.empty:
        cached = cached[cached["method"] == method]

    # Search-based news can include stories about other companies
    related = news["related_tickers"].fillna("").str.split(",")
    relevant = [
        ticker in tickers or f"({ticker})" in str(title)
        for ticker, tickers, title in zip(news["ticker"], related, news["title"])
    ]
    news = news[relevant].drop_duplicates(["ticker", "uuid"])

    seen = set(cached["uuid"]) if not cached.empty else set()
    new = news[~news["uuid"].isin(seen)].drop_duplicates("uuid")
    if not new.empty:
        text = (new["title"].fillna("") + ". " + new["summary"].fillna("")).str.strip(". ")
        log.info("Scoring %d new headlines with %s", len(new), method)
        new = new[["uuid"]].assign(sentiment=SCORERS[method](text.tolist()), method=method)
        cached = pd.concat([cached, new], ignore_index=True)
        cached.to_parquet(cache_path, index=False)

    return news.merge(cached[["uuid", "sentiment"]], on="uuid", how="left")


def sentiment_features(news, calendar, universe, method):
    scored = score_news(news, method)

    # Articles published after the 4pm ET close count toward the next session
    et = scored["published_at"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    after_close = et.dt.hour >= 16
    session = et.dt.normalize() + pd.to_timedelta(after_close.astype(int), unit="D")
    scored["date"] = next_trading_day(session, calendar)
    scored = scored.dropna(subset=["date", "sentiment"])

    total = scored.pivot_table(index="date", columns="ticker", values="sentiment", aggfunc="sum")
    count = scored.pivot_table(index="date", columns="ticker", values="sentiment", aggfunc="count")
    total = total.reindex(index=calendar, columns=universe).fillna(0)
    count = count.reindex(index=calendar, columns=universe).fillna(0)

    frames = {}
    for window in (5, 21):
        n = count.rolling(window, min_periods=1).sum()
        frames[f"news_sentiment_{window}d"] = total.rolling(window, min_periods=1).sum() / n.replace(0, np.nan)
    frames["news_count_21d"] = count.rolling(21, min_periods=1).sum()

    # Yahoo only serves recent headlines; before coverage starts, the
    # features are unknown rather than neutral.
    coverage_start = scored["date"].min()
    log.info("News coverage starts %s", pd.Timestamp(coverage_start).date())
    return stack_features({k: v.loc[v.index >= coverage_start] for k, v in frames.items()})


# ---------------------------------------------------------------------------
# Point-in-time eligibility
# ---------------------------------------------------------------------------

def eligibility(panel):
    """Flag rows that were investable on that date using only data known then.

    Biotech names need a year of history, enough market cap and enough
    liquidity; safe havens only need the history.
    """
    panel = panel.sort_values(["ticker", "date"])
    history = panel.groupby("ticker").cumcount() + 1
    is_safe = panel["ticker"].isin(SAFE_HAVENS)
    big_enough = np.exp(panel["log_market_cap"]) >= MIN_MARKET_CAP
    liquid = np.exp(panel["log_dollar_volume_63d"]) >= MIN_DOLLAR_VOLUME
    panel["is_safe_haven"] = is_safe.astype(float)
    panel["eligible"] = (history >= MIN_HISTORY_DAYS) & (is_safe | (big_enough & liquid))
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
    parser.add_argument("--sentiment", choices=["finbert", "vader", "none"], default="finbert",
                        help="Headline sentiment model (default: finbert)")
    return parser.parse_args()


def main():
    args = parse_args()
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

    log.info("SEC fundamentals")
    fund = sec_fundamentals(pd.read_parquet(RAW_DIR / "sec_facts.parquet"))
    panel = as_of_join(panel, fund, calendar)
    panel = panel.join(fundamental_features(panel))

    log.info("SEC 8-K events")
    filings = pd.read_parquet(RAW_DIR / "sec_filings.parquet")
    panel = panel.join(event_features(filings, calendar, universe))

    log.info("Earnings calendar")
    panel = panel.join(earnings_features(filings, calendar, universe))

    news_path = RAW_DIR / "news.parquet"
    if args.sentiment != "none" and news_path.exists():
        log.info("News sentiment (%s)", args.sentiment)
        news = pd.read_parquet(news_path)
        panel = panel.join(sentiment_features(news, calendar, universe, args.sentiment))

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
