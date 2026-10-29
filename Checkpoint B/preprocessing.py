"""
Train/test split and scaling for the biotech portfolio optimizer.

Reads the feature files written by ``feature_engineering.py`` and produces
model-ready datasets under ``data/model/``:

    stock_{train,test}.parquet   Full history (2015+), all features except news
    news_{train,test}.parquet    News coverage window only, all features
                                 including sentiment
    regime_{train,test}.parquet  Market-level regime features
    scalers.joblib               Fitted clip bounds + StandardScaler per dataset

Splits are chronological by date (80% earliest dates train, 20% latest test),
never random: shuffling a time series would leak future information into
training. Each dataset's scaler is fit on its training dates only and then
applied to the test dates. Before scaling, features are clipped to the
training set's 1st/99th percentiles so that extreme values (e.g. ROE on
near-zero equity) do not dominate the mean and standard deviation.

Usage:
    python preprocessing.py
    python preprocessing.py --train-frac 0.8
"""

import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
FEATURE_DIR = BASE_DIR / "data" / "features"
MODEL_DIR = BASE_DIR / "data" / "model"

ID_COLUMNS = ["date", "ticker"]

# Raw inputs kept in the feature file for reference. Dollar amounts and share
# counts are not comparable across companies, so they are not model features.
RAW_COLUMNS = [
    "close", "cash", "equity", "net_income", "operating_cash_flow",
    "operating_income", "revenue", "shares_outstanding", "short_term_investments",
]

NEWS_COLUMNS = ["news_sentiment_5d", "news_sentiment_21d", "news_count_21d"]

CLIP_QUANTILES = (0.01, 0.99)

log = logging.getLogger("preprocessing")


# ---------------------------------------------------------------------------
# Split and scale
# ---------------------------------------------------------------------------

def chronological_split(df, train_frac, cutoff=None):
    """Split on date so that every test date is after every train date."""
    if cutoff is None:
        dates = pd.Series(df["date"].unique()).sort_values().reset_index(drop=True)
        cutoff = dates.iloc[int(len(dates) * train_frac)]
    train = df[df["date"] < cutoff].reset_index(drop=True)
    test = df[df["date"] >= cutoff].reset_index(drop=True)
    return train, test, cutoff


def fit_scaler(train, features):
    """Fit clip bounds and a StandardScaler on training rows only."""
    lower = train[features].quantile(CLIP_QUANTILES[0])
    upper = train[features].quantile(CLIP_QUANTILES[1])
    # StandardScaler ignores NaNs when fitting and keeps them when transforming
    scaler = StandardScaler().fit(train[features].clip(lower, upper, axis=1))
    return {"features": features, "lower": lower, "upper": upper, "scaler": scaler}


def transform(df, fitted):
    features = fitted["features"]
    clipped = df[features].clip(fitted["lower"], fitted["upper"], axis=1)
    scaled = pd.DataFrame(
        fitted["scaler"].transform(clipped), columns=features, index=df.index
    )
    id_cols = [c for c in ID_COLUMNS if c in df]
    return pd.concat([df[id_cols], scaled], axis=1)


def build_dataset(name, df, features, train_frac, cutoff=None):
    """Split, fit on train, scale both halves, and save."""
    train, test, cutoff = chronological_split(df, train_frac, cutoff)
    fitted = fit_scaler(train, features)
    fitted["cutoff"] = cutoff

    for split, part in (("train", train), ("test", test)):
        out = transform(part, fitted)
        out.to_parquet(MODEL_DIR / f"{name}_{split}.parquet", index=False)

    log.info(
        "%-6s train %s -> %s (%d rows) | test %s -> %s (%d rows) | %d features",
        name,
        train["date"].min().date(), train["date"].max().date(), len(train),
        test["date"].min().date(), test["date"].max().date(), len(test),
        len(features),
    )
    return fitted


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--train-frac", type=float, default=0.8,
                        help="Fraction of dates used for training (default: 0.8)")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    stock = pd.read_parquet(FEATURE_DIR / "stock_features.parquet")
    regime = pd.read_parquet(FEATURE_DIR / "regime_features.parquet")

    model_features = [c for c in stock.columns if c not in ID_COLUMNS + RAW_COLUMNS]
    news_features = [c for c in NEWS_COLUMNS if c in stock]
    base_features = [c for c in model_features if c not in news_features]
    regime_features = [c for c in regime.columns if c != "date"]

    scalers = {"stock": build_dataset("stock", stock, base_features, args.train_frac)}
    # Same cutoff as the stock panel so the two can be joined without leakage
    scalers["regime"] = build_dataset(
        "regime", regime, regime_features, args.train_frac, scalers["stock"]["cutoff"]
    )

    # Yahoo only serves recent headlines, so sentiment gets its own split
    # within the window where news exists.
    if news_features:
        news_start = stock.loc[stock[news_features[0]].notna(), "date"].min()
        news = stock[stock["date"] >= news_start]
        scalers["news"] = build_dataset("news", news, model_features, args.train_frac)

    joblib.dump(scalers, MODEL_DIR / "scalers.joblib")
    log.info("Saved scalers -> %s", MODEL_DIR / "scalers.joblib")


if __name__ == "__main__":
    main()
