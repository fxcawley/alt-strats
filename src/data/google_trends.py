"""
Google Trends data fetcher.

Fetches search interest data for economic keywords using the pytrends
library. The signal is the z-score of search volume relative to its
own trailing history.

LOOK-AHEAD BIAS WARNING:
  Google retroactively normalizes historical values to a 0-100 scale
  based on the PEAK in the query's time range. If you fetch 2015-2026
  in one request, the 2015 values are scaled relative to the 2020 COVID
  spike -- information that wasn't available in 2015.

  To avoid this, we fetch data in narrow rolling windows (1 year each)
  so normalization is local to each window. The z-score computation
  then uses only trailing data, making it safe for backtesting.

Key keywords tracked:
  - "unemployment" / "jobs" -- labor market stress
  - "bankruptcy" -- corporate distress
  - "recession" -- macro fear indicator
  - "new car" / "mortgage" -- consumer spending proxy

Google Trends data is normalized (0-100 scale per query), so
cross-keyword comparisons need z-scoring.

Rate limits: Google Trends has aggressive rate limiting. Use caching
and avoid hammering the API.
"""

from __future__ import annotations

import time
from pathlib import Path
from datetime import datetime, timedelta

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


def fetch_google_trends_rolling(
    keywords: list[str],
    start_date: str = "2010-01-01",
    end_date: str | None = None,
    window_months: int = 12,
    geo: str = "US",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch Google Trends data using rolling windows to avoid normalization bias.

    Instead of fetching the full date range (which causes Google to normalize
    all values against the global peak), we fetch overlapping 1-year windows
    and concatenate. Each window's values are only normalized against that
    window's own peak, preventing future peaks from contaminating past values.

    Parameters
    ----------
    keywords : list[str]
        Max 5 keywords per API call.
    start_date : str
        Start of the date range.
    end_date : str or None
        End of the date range. Defaults to today.
    window_months : int
        Width of each rolling fetch window in months. Default 12.
    geo : str
        Geographic region.
    use_cache : bool
        Cache results to parquet.

    Returns a DataFrame indexed by date with one column per keyword.
    Each value is the raw 0-100 interest level from its local window.
    Use compute_trends_zscore() to normalize for backtesting.
    """
    from pytrends.request import TrendReq

    cache_key = "_".join(keywords[:3]).replace(" ", "_")[:50]
    cache_path = CACHE_DIR / f"gtrends_rolling_{cache_key}_{geo}.parquet"

    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if end_date is None:
        end_dt = datetime.now()
    else:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")

    pytrends = TrendReq(hl="en-US", tz=300)

    all_windows = []
    current_start = start_dt

    while current_start < end_dt:
        current_end = min(
            current_start + timedelta(days=window_months * 30),
            end_dt,
        )
        timeframe = (f"{current_start.strftime('%Y-%m-%d')} "
                     f"{current_end.strftime('%Y-%m-%d')}")

        # Google Trends allows max 5 keywords per request
        for i in range(0, len(keywords), 5):
            batch = keywords[i:i + 5]
            try:
                pytrends.build_payload(batch, cat=0, timeframe=timeframe, geo=geo)
                data = pytrends.interest_over_time()

                if not data.empty:
                    if "isPartial" in data.columns:
                        data = data.drop(columns=["isPartial"])
                    all_windows.append(data)

            except Exception as e:
                print(f"Warning: Google Trends fetch failed for "
                      f"{timeframe}: {e}")

            time.sleep(3)  # Rate limit

        current_start = current_end - timedelta(days=7)  # 1-week overlap

    if not all_windows:
        return pd.DataFrame()

    # Concatenate and deduplicate (keep last window's value for overlaps)
    combined = pd.concat(all_windows)
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()

    if use_cache and not combined.empty:
        combined.to_parquet(cache_path)

    return combined


def fetch_google_trends(
    keywords: list[str],
    timeframe: str = "today 5-y",
    geo: str = "US",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch Google Trends data for a list of keywords.

    WARNING: This function fetches the full timeframe in one request,
    which means Google normalizes ALL values against the peak in the
    entire range. Historical values are contaminated by future peaks.
    Use fetch_google_trends_rolling() for backtesting.

    This function is kept for quick exploratory analysis only.
    """
    from pytrends.request import TrendReq

    cache_key = "_".join(keywords[:3]).replace(" ", "_")[:50]
    cache_path = CACHE_DIR / f"gtrends_{cache_key}_{geo}.parquet"

    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    pytrends = TrendReq(hl="en-US", tz=300)

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

    This is safe for backtesting because it only uses trailing data.
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
