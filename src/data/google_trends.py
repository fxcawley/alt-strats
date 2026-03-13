"""
Google Trends data fetcher.

Fetches search interest data for economic keywords using the pytrends
library. The signal is the z-score of search volume relative to its
own 52-week history.

Key keywords tracked:
  - "unemployment" / "jobs" -- labor market stress
  - "bankruptcy" -- corporate distress
  - "recession" -- macro fear indicator
  - "new car" / "mortgage" -- consumer spending proxy
  - "stock market crash" -- retail panic indicator

Google Trends data is normalized (0-100 scale per query), so
cross-keyword comparisons need z-scoring.

Rate limits: Google Trends has aggressive rate limiting. Use caching
and avoid hammering the API.
"""

from __future__ import annotations

import time
from pathlib import Path
from datetime import datetime

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"

# Keywords to track (economic activity indicators)
ECONOMIC_KEYWORDS = {
    "labor_stress": ["unemployment", "layoffs", "job loss"],
    "corporate_distress": ["bankruptcy", "default", "restructuring"],
    "macro_fear": ["recession", "economic crisis", "stock market crash"],
    "consumer_spending": ["new car", "mortgage rate", "home buying"],
    "investment_sentiment": ["buy stocks", "sell stocks", "market bottom"],
}

# Flatten for API calls
ALL_KEYWORDS = []
for group in ECONOMIC_KEYWORDS.values():
    ALL_KEYWORDS.extend(group)


def fetch_google_trends(
    keywords: list[str],
    timeframe: str = "today 5-y",
    geo: str = "US",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch Google Trends data for a list of keywords.

    Parameters
    ----------
    keywords : list[str]
        Max 5 keywords per API call (Google Trends limit).
    timeframe : str
        Time range (e.g., "today 5-y", "2020-01-01 2025-01-01").
    geo : str
        Geographic region (e.g., "US", "" for worldwide).
    use_cache : bool
        Cache results to parquet.

    Returns a DataFrame indexed by date with one column per keyword.
    """
    from pytrends.request import TrendReq

    cache_key = "_".join(keywords[:3]).replace(" ", "_")[:50]
    cache_path = CACHE_DIR / f"gtrends_{cache_key}_{geo}.parquet"

    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    pytrends = TrendReq(hl="en-US", tz=300)

    # Google Trends allows max 5 keywords per request
    all_data = pd.DataFrame()
    for i in range(0, len(keywords), 5):
        batch = keywords[i:i + 5]
        pytrends.build_payload(batch, cat=0, timeframe=timeframe, geo=geo)
        data = pytrends.interest_over_time()

        if not data.empty and "isPartial" in data.columns:
            data = data.drop(columns=["isPartial"])

        if all_data.empty:
            all_data = data
        else:
            all_data = all_data.join(data, how="outer")

        # Rate limit
        time.sleep(2)

    if use_cache and not all_data.empty:
        all_data.to_parquet(cache_path)

    return all_data


def compute_trends_zscore(
    trends_df: pd.DataFrame,
    lookback_weeks: int = 52,
) -> pd.DataFrame:
    """Compute z-score of search volume relative to trailing history.

    High z-score on "unemployment" = elevated labor market stress
    (bearish for equities). Low z-score = normal conditions.
    """
    rolling_mean = trends_df.rolling(lookback_weeks, min_periods=26).mean()
    rolling_std = trends_df.rolling(lookback_weeks, min_periods=26).std()
    zscore = (trends_df - rolling_mean) / rolling_std.replace(0, float("nan"))
    return zscore


def compute_aggregate_stress_index(
    trends_zscore: pd.DataFrame,
) -> pd.Series:
    """Compute an aggregate economic stress index from Google Trends.

    Averages the z-scores of stress-related keywords. Higher = more stress.
    Returns a single series that can be used as a macro timing signal.
    """
    stress_keywords = []
    for kw in ["unemployment", "layoffs", "job loss", "bankruptcy",
               "recession", "economic crisis", "stock market crash"]:
        if kw in trends_zscore.columns:
            stress_keywords.append(kw)

    if not stress_keywords:
        return pd.Series(dtype=float)

    return trends_zscore[stress_keywords].mean(axis=1).rename("stress_index")
