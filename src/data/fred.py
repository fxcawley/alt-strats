"""
FRED economic data API client.

Fetches macroeconomic data from the Federal Reserve Economic Data (FRED)
API. Key series include initial jobless claims, ISM PMI, retail sales,
CPI, and other indicators with known release dates.

LOOK-AHEAD BIAS SAFEGUARDS:
  1. Publication lag: FRED indexes data by observation period (e.g.,
     "January 2024" for that month's CPI), NOT by release date.
     We apply explicit publication offsets per series so that
     .loc[:date] only returns data that was actually available.
  2. Vintage correctness: The default FRED API returns *revised* values.
     GDP is revised three times; employment is revised monthly.
     For proper backtesting, use realtime_start/realtime_end params
     (ALFRED) to get initial-release values. This module uses ALFRED
     when available via fetch_fred_series_vintage().

The signal is in the surprise (actual vs expected), not the level.
Since we don't have consensus estimates for free, we proxy the
"surprise" as the difference between the initial-release value and
a trailing average of prior initial releases.

Requires a FRED API key (free from https://fred.stlouisfed.org/docs/api/).
Set the environment variable FRED_API_KEY or pass it directly.
"""

from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"

# Key macro series, FRED IDs, and publication lag in calendar days.
# The lag is the typical delay between the observation period end
# and when the data is first released to the public.
MACRO_SERIES = {
    "initial_claims":     {"id": "ICSA",    "lag_days": 5},    # Weekly, released Thursday (5d lag)
    "continued_claims":   {"id": "CCSA",    "lag_days": 12},   # Weekly, 12d lag
    "retail_sales":       {"id": "RSXFS",   "lag_days": 46},   # Monthly, released ~15th of following month
    "industrial_prod":    {"id": "INDPRO",  "lag_days": 46},   # Monthly, ~15th of following month
    "consumer_sentiment": {"id": "UMCSENT", "lag_days": 1},    # Prelim released same month
    "housing_starts":     {"id": "HOUST",   "lag_days": 50},   # Monthly, ~18th of following month
    "cpi":                {"id": "CPIAUCSL","lag_days": 40},   # Monthly, ~12th of following month
    "unemployment_rate":  {"id": "UNRATE",  "lag_days": 35},   # Monthly, first Friday of following month
    "nonfarm_payrolls":   {"id": "PAYEMS",  "lag_days": 35},   # Monthly, first Friday of following month
    "pce":                {"id": "PCE",     "lag_days": 60},   # Monthly, ~last bday of following month
    "gdp":                {"id": "GDP",     "lag_days": 90},   # Quarterly, advance estimate ~30d after Q-end
}


def _get_fred_api():
    """Get a FRED API client (lazy import to avoid mandatory dependency)."""
    from fredapi import Fred
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise ValueError(
            "FRED_API_KEY environment variable not set. "
            "Get a free key at https://fred.stlouisfed.org/docs/api/"
        )
    return Fred(api_key=api_key)


def fetch_fred_series(
    series_id: str,
    start_date: str = "2005-01-01",
    end_date: str | None = None,
    use_cache: bool = True,
) -> pd.Series:
    """Fetch a single FRED series (latest-vintage, i.e. revised values).

    WARNING: This returns revised data. For backtesting, use
    fetch_fred_series_vintage() to get initial-release values.
    This function is suitable for exploratory analysis only.

    Returns a Series indexed by observation date.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"fred_{series_id}.parquet"

    if use_cache and cache_path.exists():
        df = pd.read_parquet(cache_path)
        return df.iloc[:, 0]

    fred = _get_fred_api()
    data = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)

    if data.empty:
        raise ValueError(f"No data returned for FRED series {series_id}")

    df = data.to_frame(name=series_id)
    if use_cache:
        df.to_parquet(cache_path)

    return data


def fetch_fred_series_vintage(
    series_id: str,
    start_date: str = "2005-01-01",
    end_date: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch vintage-correct (ALFRED) data for a FRED series.

    Returns a DataFrame with columns:
      - 'value': the initial-release value (what the market saw)
      - 'observation_date': the period the data describes
      - 'realtime_start': when this vintage was first published

    Uses the FRED API's realtime_start/realtime_end to reconstruct
    the initial release values, avoiding revision look-ahead bias.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"fred_vintage_{series_id}.parquet"

    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    fred = _get_fred_api()

    # Fetch the full vintage history using get_series_all_releases
    try:
        vintages = fred.get_series_all_releases(series_id)
    except Exception:
        # Fallback: not all series support ALFRED. Use latest vintage
        # but flag it.
        data = fred.get_series(series_id, observation_start=start_date,
                               observation_end=end_date)
        if data.empty:
            raise ValueError(f"No data returned for FRED series {series_id}")
        df = pd.DataFrame({
            "observation_date": data.index,
            "value": data.values,
            "realtime_start": data.index,  # Not vintage-correct -- flagged
            "is_vintage_correct": False,
        })
        if use_cache:
            df.to_parquet(cache_path)
        return df

    if vintages.empty:
        raise ValueError(f"No vintage data returned for FRED series {series_id}")

    # Keep only the first release for each observation date
    # (the value the market saw on release day)
    vintages = vintages.sort_values("realtime_start")
    first_releases = vintages.groupby("date").first().reset_index()
    first_releases = first_releases.rename(columns={"date": "observation_date"})
    first_releases["is_vintage_correct"] = True

    if use_cache:
        first_releases.to_parquet(cache_path)

    return first_releases


def _apply_publication_lag(
    series: pd.Series,
    lag_days: int,
) -> pd.Series:
    """Shift a series forward by the publication lag.

    Converts observation-date indexing to availability-date indexing.
    After this, .loc[:date] returns only data that was actually
    published on or before `date`.
    """
    shifted_index = series.index + pd.Timedelta(days=lag_days)
    return pd.Series(series.values, index=shifted_index, name=series.name)


def fetch_all_macro_series(
    start_date: str = "2005-01-01",
    end_date: str | None = None,
    use_cache: bool = True,
    apply_pub_lag: bool = True,
    use_vintage: bool = True,
) -> pd.DataFrame:
    """Fetch all standard macro series from FRED with publication lag applied.

    Returns a DataFrame with columns for each series, indexed by
    *availability date* (not observation date). This means .loc[:date]
    is safe for point-in-time backtesting.

    Parameters
    ----------
    apply_pub_lag : bool
        If True (default), shifts each series forward by its publication
        lag so the index reflects when the data was available, not when
        the observation period ended. Set to False only for exploratory
        analysis where you understand the look-ahead implications.
    use_vintage : bool
        If True (default), uses ALFRED vintage data to get initial-release
        values instead of revised values. Falls back to revised data if
        the vintage API isn't available for a given series, but logs a
        warning. Set to False only for exploratory analysis.
    """
    series_dict = {}
    for name, meta in MACRO_SERIES.items():
        series_id = meta["id"]
        lag_days = meta["lag_days"]
        try:
            if use_vintage:
                vintage_df = fetch_fred_series_vintage(
                    series_id, start_date, end_date, use_cache
                )
                if not vintage_df["is_vintage_correct"].all():
                    print(f"Warning: {name} ({series_id}) vintage data not "
                          f"available -- using revised values (look-ahead risk)")
                data = pd.Series(
                    vintage_df["value"].values,
                    index=pd.DatetimeIndex(vintage_df["observation_date"]),
                    name=series_id,
                )
            else:
                data = fetch_fred_series(series_id, start_date, end_date, use_cache)

            if apply_pub_lag:
                data = _apply_publication_lag(data, lag_days)
            series_dict[name] = data
        except Exception as e:
            print(f"Warning: could not fetch {name} ({series_id}): {e}")

    if not series_dict:
        raise ValueError("No FRED series could be fetched")

    return pd.DataFrame(series_dict)


def compute_fred_surprise(
    macro_df: pd.DataFrame,
    lookback_periods: int = 12,
) -> pd.DataFrame:
    """Compute 'surprise' as deviation from trailing average.

    Since we don't have consensus estimates for free, we proxy the surprise as:
        surprise = (actual - trailing_mean) / trailing_std

    This z-score captures whether the release was above or below
    recent trends. Positive surprise = stronger-than-expected economy.

    NOTE: This is computed on the publication-lag-adjusted DataFrame,
    so the trailing average uses only previously-published values.
    If you pass a DataFrame without lag adjustment, this computation
    has look-ahead bias.
    """
    rolling_mean = macro_df.rolling(lookback_periods, min_periods=6).mean()
    rolling_std = macro_df.rolling(lookback_periods, min_periods=6).std()

    surprise = (macro_df - rolling_mean) / rolling_std.replace(0, float("nan"))
    return surprise
