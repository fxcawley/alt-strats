"""
VIIRS nighttime light data processor.

Processes NOAA VIIRS Day/Night Band composites to extract economic
activity proxies. Month-over-month luminosity changes at the country
level correlate with GDP growth and can be used as a leading indicator
for country/sector ETFs.

LOOK-AHEAD BIAS: VIIRS composites have a 2-3 month publication lag.
The code applies a 90-day (conservative) offset so that .loc[:date]
only returns data that was actually available at that time.

In practice, this module provides a framework for processing the data.
Actual VIIRS composites are large GeoTIFF files (~500MB per month).
For this project, we use pre-aggregated country-level data.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

SATELLITE_DIR = Path(__file__).resolve().parents[2] / "data" / "satellite"

# Publication lag for VIIRS composites (conservative estimate)
VIIRS_PUBLICATION_LAG_DAYS = 90

# Country -> ETF mapping for satellite signals
COUNTRY_ETF_MAP = {
    "US": ["SPY", "QQQ", "IWM"],
    "China": ["FXI", "EEM"],
    "Japan": ["EWJ"],
    "Europe": ["VGK", "EFA"],
    "Germany": ["VGK"],
    "UK": ["EFA"],
    "India": ["EEM"],
    "Brazil": ["EEM"],
    "South Korea": ["EEM"],
}


def load_luminosity_data(filepath: str | Path | None = None) -> pd.DataFrame:
    """Load country-level luminosity data.

    Expected format: CSV with columns [date, country, luminosity].
    If no file provided, returns a template DataFrame.

    In production, this would read from processed VIIRS GeoTIFF
    aggregates. For development, we support CSV input.
    """
    if filepath is not None:
        path = Path(filepath)
        if path.exists():
            df = pd.read_csv(path, parse_dates=["date"])
            return df

    # Return empty template
    return pd.DataFrame(columns=["date", "country", "luminosity"])


def compute_luminosity_change(
    luminosity_df: pd.DataFrame,
    apply_pub_lag: bool = True,
) -> pd.DataFrame:
    """Compute month-over-month luminosity change per country.

    Returns a pivoted DataFrame: index=date, columns=country,
    values=pct_change in luminosity.

    Parameters
    ----------
    apply_pub_lag : bool
        If True (default), shifts the index forward by VIIRS_PUBLICATION_LAG_DAYS
        so that .loc[:date] only returns data available at that time.
    """
    if luminosity_df.empty:
        return pd.DataFrame()

    pivot = luminosity_df.pivot_table(
        index="date",
        columns="country",
        values="luminosity",
        aggfunc="mean",
    )
    pivot = pivot.sort_index()
    changes = pivot.pct_change()

    if apply_pub_lag:
        changes.index = changes.index + pd.Timedelta(days=VIIRS_PUBLICATION_LAG_DAYS)

    return changes


def luminosity_to_etf_signals(
    luminosity_change: pd.DataFrame,
    date: pd.Timestamp,
) -> dict[str, float]:
    """Map country luminosity changes to ETF signals.

    Positive luminosity change in a country -> bullish for associated ETFs.
    Returns a dict of {ticker: signal_value}.
    """
    if luminosity_change.empty:
        return {}

    # Get most recent data
    available = luminosity_change.loc[:date]
    if available.empty:
        return {}

    latest = available.iloc[-1]

    signals: dict[str, list[float]] = {}
    for country, change in latest.items():
        if pd.isna(change):
            continue
        etfs = COUNTRY_ETF_MAP.get(country, [])
        for etf in etfs:
            if etf not in signals:
                signals[etf] = []
            signals[etf].append(change)

    # Equal-weight average across all countries mapping to each ETF
    return {etf: sum(vals) / len(vals) for etf, vals in signals.items() if vals}



