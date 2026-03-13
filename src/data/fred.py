"""
FRED economic data API client.

Fetches macroeconomic data from the Federal Reserve Economic Data (FRED)
API. Key series include initial jobless claims, ISM PMI, retail sales,
CPI, and other indicators with known release dates.

The signal is in the surprise (actual vs expected), not the level.
Since we don't have consensus estimates for free, we proxy the
"surprise" as the difference between the actual value and a simple
trailing average.

Requires a FRED API key (free from https://fred.stlouisfed.org/docs/api/).
Set the environment variable FRED_API_KEY or pass it directly.
"""

from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"

# Key macro series and their FRED IDs
MACRO_SERIES = {
    "initial_claims": "ICSA",           # Weekly initial jobless claims
    "continued_claims": "CCSA",          # Continued claims
    "ism_pmi": "MANEMP",                # ISM Manufacturing Employment
    "retail_sales": "RSXFS",             # Advance Retail Sales
    "industrial_prod": "INDPRO",         # Industrial Production Index
    "consumer_sentiment": "UMCSENT",     # U Michigan Consumer Sentiment
    "housing_starts": "HOUST",           # Housing Starts
    "cpi": "CPIAUCSL",                   # CPI All Urban Consumers
    "unemployment_rate": "UNRATE",       # Unemployment Rate
    "nonfarm_payrolls": "PAYEMS",        # Nonfarm Payrolls
    "pce": "PCE",                        # Personal Consumption Expenditures
    "gdp": "GDP",                        # Gross Domestic Product
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
    """Fetch a single FRED series.

    Returns a Series indexed by date.
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

    # Cache as single-column DataFrame
    df = data.to_frame(name=series_id)
    if use_cache:
        df.to_parquet(cache_path)

    return data


def fetch_all_macro_series(
    start_date: str = "2005-01-01",
    end_date: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch all standard macro series from FRED.

    Returns a DataFrame with columns for each series, indexed by date.
    Missing dates are NaN (series have different frequencies).
    """
    series_dict = {}
    for name, series_id in MACRO_SERIES.items():
        try:
            series_dict[name] = fetch_fred_series(
                series_id, start_date, end_date, use_cache
            )
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

    Since we don't have consensus estimates, we proxy the surprise as:
        surprise = (actual - trailing_mean) / trailing_std

    This z-score captures whether the release was above or below
    recent trends. Positive surprise = stronger-than-expected economy.
    """
    rolling_mean = macro_df.rolling(lookback_periods, min_periods=6).mean()
    rolling_std = macro_df.rolling(lookback_periods, min_periods=6).std()

    surprise = (macro_df - rolling_mean) / rolling_std.replace(0, float("nan"))
    return surprise
